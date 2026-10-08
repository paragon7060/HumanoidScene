"""Explicit supplemental Drive backup and cleanup for stopped SAC raw files.

Original checkpoint/log backup scope and status are preserved. This worker
uploads only the two named raw formats, verifies size/MD5, then removes unused
local payloads. Checkpoints, logs, videos, active inputs, and other users are
excluded. Reuses the checkout's existing authenticated rclone connection.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from drive_backup import ROOT, Rclone, md5, signature
from prune_verified_payloads import process_inventory

PAYLOADS = ("staged_goal_experience.pt", "executed_transitions.hdf5")


def closed_source(parent):
    parent = Path(parent)
    if not parent.is_absolute() or parent.is_symlink() or parent.stat().st_uid != os.getuid():
        raise ValueError("Explicit owned direct managed parent required")
    for name in ("launch.json", "status.json"):
        path = parent/name
        if not path.is_file() or path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError("Owned managed metadata required")
    launch = json.loads((parent/"launch.json").read_text())
    status = json.loads((parent/"status.json").read_text())
    if status.get("phase") != "finished" or status.get("training_exit_code") != 0 \
            or not status.get("final_upload_verified"):
        raise ValueError("Normal writer exit and original final Drive verification required")
    for key in ("training_pid", "supervisor_pid"):
        if type(status.get(key)) is not int or status[key] <= 0 or Path("/proc",str(status[key])).exists():
            raise ValueError("Original writer/supervisor still present or unbound")
    recorded = Path(launch["run"])
    run = recorded.resolve()
    if run.parent != parent or not run.is_dir() or run.stat().st_uid != os.getuid() \
            or not re.fullmatch(r"batch_sac_\d{8}_\d{6}_[a-f0-9]{6}",run.name):
        raise ValueError("Resolved launch run must belong to explicit managed parent")
    if status.get("run_dir") and Path(status["run_dir"]).resolve() != run:
        raise ValueError("Managed status run differs")
    command = launch.get("command", [])
    if command.count("--output-dir") != 1 or Path(command[command.index("--output-dir")+1]).resolve() != run:
        raise ValueError("Original command output directory differs")
    manifest = run/"manifest.json"
    if not manifest.is_file() or manifest.is_symlink() or manifest.stat().st_uid != os.getuid():
        raise ValueError("Owned original run manifest required")
    json.loads(manifest.read_text())
    binding = dict(run=str(run),recorded_alias=str(recorded),launch_signature=signature(parent/"launch.json"),
        writer=status["training_pid"],supervisor=status["supervisor_pid"],original_backup_scope=launch.get("backup_scope"))
    return run,binding


def unused(path, run, binding, protected):
    value = path.lstat()
    if path.resolve() in protected or not stat.S_ISREG(value.st_mode) or value.st_uid != os.getuid() \
            or value.st_nlink != 1:
        return False
    commands, opened = process_inventory()
    if (value.st_dev,value.st_ino) in opened:
        return False
    candidates = (str(path),str(run),binding["recorded_alias"],str(Path(binding["recorded_alias"])/path.name))
    return not any(text.encode() in command for command in commands for text in candidates)


def write_owned_json(path, value):
    if path.is_symlink():
        raise ValueError("Refusing symlink journal")
    temporary = path.with_suffix(".atomic.tmp")
    temporary.write_text(json.dumps(value,indent=2)+"\n")
    temporary.replace(path)


def offload(parent, remote_root, remote, *, protected=(), event=None):
    run,binding = closed_source(parent)
    protected = {Path(p).resolve() for p in protected}
    marker = run/"supplemental_raw_Drive_offload.json"
    previous = json.loads(marker.read_text()) if marker.exists() and not marker.is_symlink() else {}
    records = {x["file"]:x for x in previous.get("files",[])}
    removed,retained = [],[]
    with (run/".drive-backup.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name in PAYLOADS:
            path = run/name
            if not path.exists():
                continue
            if not unused(path,run,binding,protected):
                retained.append(dict(file=name,reason="in_use_protected_or_nonregular"));continue
            before = signature(path)
            digest = md5(path)
            if signature(path) != before:
                raise ValueError("Closed source changed while hashing")
            destination = remote_root.rstrip("/")+"/"+run.name+"/"+name
            if event:event(dict(phase="uploading",run_directory_name=run.name,file=name,bytes=before[2]))
            remote.upload(path,destination)
            remote.verify(path,destination,digest)
            fresh,fresh_binding = closed_source(parent)
            if fresh != run or fresh_binding != binding or signature(path) != before \
                    or not unused(path,run,binding,protected):
                retained.append(dict(file=name,reason="runtime_input_or_file_changed_after_upload"));continue
            # Verify again immediately before recording/removing the local copy.
            remote.verify(path,destination,digest)
            if signature(path) != before or not unused(path,run,binding,protected):
                retained.append(dict(file=name,reason="source_changed_before_removal"));continue
            value = path.stat()
            record = dict(file=name,bytes=value.st_size,allocated_bytes=value.st_blocks*512,MD5=digest,
                existing_Drive_size_MD5_verified=True,local_removal_complete=False,
                remote_subfolder="HumanoidScene-RL/"+run.name)
            records[name] = record
            def journal():
                write_owned_json(marker,dict(recorded_at=datetime.now().astimezone().isoformat(),
                    run_directory_name=run.name,original_backup_scope=binding["original_backup_scope"],
                    original_managed_launch_and_status_NOT_changed=True,
                    supplemental_raw_backup_explicit=True,files=list(records.values()),
                    checkpoints_logs_media_and_active_inputs_retained=True,
                    restore_payload_before_resuming_or_raw_analysis=True))
            journal()
            path.unlink()
            records[name].update(local_removal_complete=True,local_removed_at=datetime.now().astimezone().isoformat())
            journal();removed.append(records[name])
            if event:event(dict(phase="verified_and_pruned",run_directory_name=run.name,file=name,allocated_bytes=value.st_blocks*512))
    return dict(run_directory_name=run.name,removed=removed,retained=retained,
                allocated_bytes_removed=sum(x["allocated_bytes"] for x in removed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--watch",type=int,default=0)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.watch and args.watch<300:
        parser.error("CPU-only worker; watch>=300 seconds required")
    if args.config.is_symlink() or args.config.stat().st_uid != os.getuid():
        raise ValueError("Owned regular explicit configuration required")
    config = json.loads(args.config.read_text())
    parents = [Path(x) for x in config["experiment_dirs"]]
    if not parents or len(parents) != len(set(parents)):
        raise ValueError("Distinct explicit managed directories required")
    remotes = subprocess.check_output(["bash",str(ROOT/"scripts/rl/gdrive.sh"),"listremotes"],text=True).splitlines()
    if len(remotes) != 1 or not re.fullmatch(r"[\w-]+:",remotes[0]):
        raise ValueError("Select the existing host-local connection")
    remote_root = remotes[0]+"HumanoidScene-RL"
    remote = Rclone(ROOT/"scripts/rl/gdrive.sh")
    # Capture privately; never display account-specific alias/auth information.
    remote.call("about",remotes[0],"--json")
    status_path = args.config.with_suffix(".status.json")
    with args.config.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        results = []
        def event(value):
            data = dict(recorded_at=datetime.now().astimezone().isoformat(),worker_pid=os.getpid(),
                GPU_mask_empty=True,current=value,results=results,no_process_signals_or_new_auth=True)
            write_owned_json(status_path,data)
            print(json.dumps(value),flush=True)
        while True:
            results = []
            for parent in parents:
                try:
                    results.append(offload(parent,remote_root,remote,protected=config.get("protected_paths",[]),event=event))
                except Exception as error:
                    results.append(dict(experiment_directory_name=parent.name,error_type=type(error).__name__,retained_unverified=True))
            remaining = any((p/name).exists() for parent in parents for p in [closed_source(parent)[0]] for name in PAYLOADS)
            event(dict(phase="waiting" if remaining else "done",remaining_raw_payloads=remaining))
            if not args.watch or not remaining:return
            time.sleep(args.watch)


if __name__ == "__main__":
    main()
