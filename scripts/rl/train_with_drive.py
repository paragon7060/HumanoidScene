#!/usr/bin/env python3
"""Supervise one PPO run and its verified Drive backups, without using a GPU here."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import runpy
import time

from drive_backup import Rclone, ROOT, archive_file, backup_once

# CPU supervisors run in system Python without the simulator/package dependencies.
add_action_space_argument = runpy.run_path(
    str(ROOT / "src/kuavo_isaaclab_scene/rl/action_spaces.py"))["add_action_space_argument"]


def write_status(parent, **values):
    path = parent / "status.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    state.update(values, updated_at=datetime.now().astimezone().isoformat())
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(path)


def archive(source, remote_root, finished=False):
    destination = f"{remote_root.rstrip('/')}/{source.name}"
    with (source / ".drive-backup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        remote = Rclone(ROOT / "scripts/rl/gdrive.sh")
        if (source / "manifest.json").is_file():
            return backup_once(source, destination, remote, keep=2,
                               min_age=0 if finished else 120, finished=finished)
        if not finished:
            raise RuntimeError("Training has not finished writing its manifest")
        # Initialization can fail before a manifest exists. Still preserve its logs.
        for path in sorted(source.glob("*.log")):
            archive_file(path, destination, remote)
        return []


def supervise(command, parent, environment, archive_run, interval=300, poll=5,
              min_free_bytes=5 * 1024**3, stop_timeout=120, run_prefix="train_",
              resource_check=None, require_run_status=False):
    """Own only this child process group; finalize logs after all its writes stop."""
    requested_stop = []
    handlers = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
        handlers[sig] = signal.signal(sig, lambda number, frame: requested_stop.append(number))
    console = parent / "console.log"
    output = console.open("xb", buffering=0)
    child = None
    source = None
    future = None
    executor = ThreadPoolExecutor(max_workers=1)
    next_backup = 0
    stop_at = None
    reason = None

    def find_run():
        nonlocal source, console
        if source is None:
            candidates = [p for p in parent.glob(run_prefix + "*") if p.is_dir()]
            if len(candidates) > 1:
                raise RuntimeError("More than one training directory in this unique experiment")
            if candidates:
                source = candidates[0]
                destination = source / "console.log"
                if destination.exists():
                    raise RuntimeError("Refusing to replace an existing console log")
                console.rename(destination)
                console = destination
                write_status(parent, run_dir=str(source))

    def finish_future():
        nonlocal future
        try:
            removed = future.result()
            write_status(parent, last_backup_verified_at=datetime.now().astimezone().isoformat(),
                         backup_error=None, last_pruned=removed)
        except Exception as error:
            print(f"[Supervisor] Backup failed; retrying without pruning: {error}", flush=True)
            write_status(parent, backup_error=str(error))
        future = None

    try:
        if shutil.disk_usage(parent).free < min_free_bytes:
            raise RuntimeError("Insufficient local disk reserve before launch")
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        write_status(parent, phase="training", supervisor_pid=os.getpid(), training_pid=child.pid,
                     command=command, CUDA_VISIBLE_DEVICES=environment.get("CUDA_VISIBLE_DEVICES"),
                     upload_interval_seconds=interval, keep=2, disk_reserve_bytes=min_free_bytes)
        print(f"[Supervisor] Training PID {child.pid}; experiment {parent}", flush=True)
        while child.poll() is None:
            find_run()
            if future is not None and future.done():
                finish_future()
            now = time.monotonic()
            resource_reason = resource_check(child.pid) if resource_check and stop_at is None else None
            if stop_at is None and (requested_stop or shutil.disk_usage(parent).free < min_free_bytes or resource_reason):
                reason = ("requested_stop" if requested_stop else resource_reason or "low_disk_space")
                write_status(parent, phase="stopping", stop_reason=reason)
                os.killpg(child.pid, signal.SIGTERM)
                stop_at = now
            elif stop_at is not None and now - stop_at >= stop_timeout:
                os.killpg(child.pid, signal.SIGKILL)
                write_status(parent, forced_stop=True)
                child.wait()
            if (source is not None and future is None and now >= next_backup
                    and all((source / name).is_file() for name in ("manifest.json", "env.yaml", "agent.yaml"))):
                future = executor.submit(archive_run, source, False)
                next_backup = now + interval
            time.sleep(poll)
    except Exception as error:
        reason = str(error)
        print(f"[Supervisor] {error}", flush=True)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=stop_timeout)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        output.close()
        if future is not None:
            finish_future()
        executor.shutdown(wait=True)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)

    # The child has exited and the console writer is closed before --finished semantics.
    find_run()
    if source is None:
        source = parent / ("train_failed_" + parent.name)
        source.mkdir()
        console.rename(source / "console.log")
    exit_code = child.returncode if child is not None else None
    run_status = None
    if require_run_status:
        status_path = source / "status.json"
        run_status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "missing"}
        if run_status.get("status") != "complete" and exit_code == 0:
            exit_code = 1  # Kit can mask an exception with a zero process exit code.
    verification = {"training_exit_code": exit_code, "stop_reason": reason,
                    "run_status": run_status,
                    "writers_stopped_at": datetime.now().astimezone().isoformat()}
    (source / "verification.json").write_text(json.dumps(verification, indent=2))
    write_status(parent, phase="final_upload", training_exit_code=exit_code, stop_reason=reason)
    while True:
        try:
            archive_run(source, True)
            break
        except Exception as error:
            write_status(parent, backup_error=str(error))
            print(f"[Supervisor] Final upload failed; retrying in {interval}s: {error}", flush=True)
            time.sleep(interval)
    write_status(parent, phase="finished", final_upload_verified=True, backup_error=None)
    print(f"[Supervisor] Training exited ({exit_code}); final logs/checkpoints verified.", flush=True)
    return exit_code if exit_code is not None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_action_space_argument(parser)
    parser.add_argument("--experiment-dir", type=Path, required=True,
                        help="A new, unique local parent directory; must not exist")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--remote-root", default=os.environ.get(
        "RL_DRIVE_REMOTE_ROOT", "gdrive:HumanoidScene-RL"))
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--gpu-limit-mib", type=int, default=78000)
    parser.add_argument("--gpu-reserve-mib", type=int, default=2048)
    parser.add_argument("--max-seconds", type=int, default=7*24*3600)
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.save_interval, args.gpu_limit_mib, args.max_seconds) < 1 or args.gpu_reserve_mib < 0:
        parser.error("Invalid training counts or resource budget")
    parent = args.experiment_dir.resolve()
    parent.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES="1", OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="8",
                       ISAACLAB_PYTHON=str(Path.home() / "miniconda3/envs/env_isaaclab_232/bin/python"),
                       PYTHONPATH=str(ROOT / "src"))
    command = ["bash", str(ROOT / "train_flap_pick.sh"), "--num-envs", str(args.num_envs),
               "--max-iterations", str(args.max_iterations), "--save-interval", str(args.save_interval),
               "--device", "cuda:0", "--log-dir", str(parent), "--kit_args",
               "--/renderer/activeGpu=1 --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"]
    if args.checkpoint:
        command += ["--checkpoint", str(args.checkpoint.resolve())]
    if args.action_space:
        command.extend(("--action-space", args.action_space))
    from gpu_budget import GpuBudget
    budget = GpuBudget(parent, 1, args.gpu_limit_mib, args.gpu_reserve_mib,
                       args.max_seconds, run_prefix="train_")
    (parent / "launch.json").write_text(json.dumps({"command": command,
        "remote_root": args.remote_root, "args": vars(args)}, default=str, indent=2))
    raise SystemExit(supervise(command, parent, environment,
                              lambda source, finished: archive(source, args.remote_root, finished),
                              resource_check=budget, require_run_status=True))


if __name__ == "__main__":
    main()
