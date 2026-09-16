#!/usr/bin/env python3
"""S63 body/arm capture, bounded plan, preflight, command and telemetry logging."""
import argparse
import csv
import json
import math
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import tempfile
import xml.etree.ElementTree as ET

from remote import WORKER
import trajectory

DEFAULT_URDF = Path(__file__).resolve().parents[2] / "src/kuavo_isaaclab_scene/assets/kuavo_s63/urdf/kuavo_s63.urdf"


def load_limits(path):
    joints = {j.attrib["name"]: j for j in ET.parse(path).getroot().findall("joint")}
    return {name: (float(joints[name].find("limit").attrib["lower"]),
                   float(joints[name].find("limit").attrib["upper"])) for name in trajectory.NAMES}


def ssh_command(args, request):
    source = Path(trajectory.__file__).read_text() + "\n" + WORKER
    remote = ("set -e; source /opt/ros/noetic/setup.bash; source "
              + shlex.quote(args.workspace + "/devel/setup.bash")
              + "; export ROS_MASTER_URI=" + shlex.quote(args.ros_master)
              + "; exec python3 -u -c " + shlex.quote(source))
    return ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=2", "-o", "ServerAliveCountMax=2",
            args.host, "bash -c " + shlex.quote(remote)]


def run_remote(args, request, log):
    """No reconnection/retry after uncertain output; EOF stops the remote sequence."""
    command = ssh_command(args, request)
    done = threading.Event()
    capture = None
    summary_ok = False
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w" if request["mode"] == "capture" else "x") as output:
        output.write(json.dumps({"kind": "local_attempt", "mode": request["mode"],
                                 "unix_s": time.time(), "host": args.host,
                                 "plan": request.get("plan")}) + "\n")
        output.flush()
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   text=True, bufsize=1)
        heartbeat_stats = {"sent_count":0, "max_send_gap_s":0., "last_attempt_unix_s":None,
                           "last_success_unix_s":None, "write_error":None}

        def heartbeat():
            previous = None
            while not done.is_set():
                try:
                    heartbeat_stats["last_attempt_unix_s"] = time.time()
                    sequence = heartbeat_stats["sent_count"]+1
                    process.stdin.write(f"heartbeat {sequence}\n")
                    process.stdin.flush()
                    sent = time.monotonic()
                    if previous is not None:
                        heartbeat_stats["max_send_gap_s"] = max(
                            heartbeat_stats["max_send_gap_s"], sent-previous)
                    previous = sent
                    heartbeat_stats["sent_count"] = sequence
                    heartbeat_stats["last_success_unix_s"] = time.time()
                except (BrokenPipeError, OSError, ValueError) as exc:
                    heartbeat_stats["write_error"] = str(exc)
                    break
                done.wait(.2)

        thread = threading.Thread(target=heartbeat, daemon=True)
        try:
            process.stdin.write(json.dumps(request, allow_nan=False) + "\n")
            process.stdin.flush()
            thread.start()
            flushed = time.monotonic()
            ignored = tuple('{"kind": "'+kind+'",' for kind in
                            ("message", "metadata", "target", "timing_tick", "graph_audit", "heartbeat_rx"))
            for line in process.stdout:
                output.write(line)
                # Raw telemetry need not be parsed for console output. Batch
                # local flushes; preserve every line and flush control outcomes.
                if time.monotonic()-flushed >= .05:
                    output.flush()
                    flushed = time.monotonic()
                if line.startswith(ignored):
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print(line.rstrip(), file=sys.stderr)
                    continue
                if row.get("kind") == "capture":
                    capture = row["capture"]
                if row.get("kind") == "summary":
                    summary_ok = row.get("ok") is True
                if row.get("kind") not in ("message", "metadata", "capture", "target", "timing_tick", "graph_audit", "heartbeat_rx"):
                    output.flush()
                    print(json.dumps(row, ensure_ascii=False), flush=True)
            returncode = process.wait(timeout=10)
            if returncode or not summary_ok:
                raise RuntimeError(f"Remote worker failed (exit={returncode}); see {log}")
            return capture
        except KeyboardInterrupt:
            output.write(json.dumps({"kind": "local_interrupt", "unix_s": time.time()}) + "\n")
            raise RuntimeError("Interrupted: remote EOF watchdog stops progression; use robot stop if needed")
        finally:
            done.set()
            if thread.ident is not None:
                thread.join(timeout=1)
            output.write(json.dumps(dict(heartbeat_stats, kind="local_heartbeat_summary",
                                         unix_s=time.time()))+"\n")
            output.flush()
            process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=3)
            process.stdout.close()


def print_preview(plan, send=False):
    print(json.dumps({"send_requested": send, "preview_only": not send, "kind": plan["kind"], "joints": plan["joints"],
                      "duration_s": plan["duration_s"], "amplitude_deg": plan["amplitude_deg"],
                      "ramp_s": plan["ramp_s"], "hold_s": plan["hold_s"],
                      "max_velocity_deg_s": plan["max_velocity_deg_s"],
                      "max_acceleration_deg_s2": plan["max_acceleration_deg_s2"],
                      "baseline_source": plan["capture"]["baseline_source"],
                      "baseline_deg": dict(zip(trajectory.NAMES, map(math.degrees, plan["baseline_rad"])))},
                     ensure_ascii=False, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="lab@192.168.0.22")
    parser.add_argument("--workspace", default="/home/lab/hb/kuavo-ros-opensource")
    parser.add_argument("--ros-master", default="http://kuavo_master:11311")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture", help="Read-only capture; replaces existing capture/log; no robot command")
    cap.add_argument("--output", type=Path, required=True)
    plan_cmd = commands.add_parser("plan", help="Offline target generation; no SSH")
    plan_cmd.add_argument("--capture", type=Path, required=True)
    plan_cmd.add_argument("--kind", choices=("hold", "static", "motion"), default="hold")
    plan_cmd.add_argument("--joint", nargs="+", choices=trajectory.NAMES, default=[])
    plan_cmd.add_argument("--amplitude-deg", type=float, default=.5)
    plan_cmd.add_argument("--ramp-s", type=float, default=2.)
    plan_cmd.add_argument("--hold-s", type=float, default=5.)
    plan_cmd.add_argument("--cycles", type=int, default=3)
    plan_cmd.add_argument("--output", type=Path, required=True)
    for command in ("check", "run", "prepare"):
        p = commands.add_parser(command, help="Read-only preflight" if command == "check" else "Preview by default; --send moves robot")
        p.add_argument("--plan", type=Path, required=True)
        p.add_argument("--log", type=Path)
        p.add_argument("--tracking-deg", type=float, default=3.)
        if command == "check":
            p.add_argument("--for-prepare", action="store_true", help="Read-only seed/quick3 preparation preflight")
            p.add_argument("--watch-s", type=float, default=0., help="Read-only timing rehearsal, 1-30 seconds; no publication or mode service")
        if command in ("run", "prepare"):
            p.add_argument("--send", action="store_true")
    rec = commands.add_parser("record", help="Read-only joint_cmd/sensor/target recording")
    rec.add_argument("--duration-s", type=float, default=120.)
    rec.add_argument("--log", type=Path, required=True)
    rec.add_argument("--vr-inputs", action="store_true", help="Also passively record Quest/IK inputs; no prepare or command ownership required")
    args = parser.parse_args(argv)
    if not args.host or args.host.startswith("-"):
        parser.error("Invalid SSH destination")
    try:
        if args.command == "plan":
            if args.output.suffix != ".json":
                raise ValueError("Plan output must use .json extension")
            plan = trajectory.make_plan(json.loads(args.capture.read_text()), load_limits(args.urdf),
                                        args.kind, args.joint, args.amplitude_deg, args.ramp_s,
                                        args.hold_s, args.cycles)
            if args.output.exists() or args.output.with_suffix(".csv").exists():
                raise ValueError("Output exists; choose a new filename")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n")
            with args.output.with_suffix(".csv").open("x", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["elapsed_s", "phase"] + [n+"_rad" for n in trajectory.NAMES]
                                + [n+"_rad_s" for n in trajectory.NAMES])
                for i in range(math.ceil(plan["duration_s"]*50)+1):
                    elapsed = min(i/50, plan["duration_s"])
                    q,v,label = trajectory.sample(plan, elapsed)
                    writer.writerow([elapsed,label]+q+v)
            print_preview(plan)
            return 0
        request = {"mode": args.command, "tracking_deg": getattr(args, "tracking_deg", 3.)}
        if args.command in ("check", "run", "prepare"):
            if not math.isfinite(args.tracking_deg) or not .5 <= args.tracking_deg <= 5:
                raise ValueError("tracking-deg must be finite, [0.5,5]")
            plan = trajectory.validate_plan(json.loads(args.plan.read_text()))
            limits = load_limits(args.urdf)
            if plan["lower_rad"] != [limits[n][0] for n in trajectory.NAMES] or plan["upper_rad"] != [limits[n][1] for n in trajectory.NAMES]:
                raise ValueError("Plan URDF limits differ from current model; regenerate")
            print_preview(plan, getattr(args, "send", False))
            if args.command == "prepare":
                if plan["kind"] != "hold":
                    raise ValueError("prepare accepts hold plans only")
                print("prepare: seeds current body/arm hold, then requests WBC quickMode=3; --send required")
            if args.command in ("run", "prepare") and not args.send:
                return 0
            if args.log is None:
                raise ValueError("--log is required for check or --send")
            request.update(mode=("send" if args.command == "run" else
                                 "prepare_check" if args.command == "check" and args.for_prepare else args.command), plan=plan)
            if args.command == "check":
                if not math.isfinite(args.watch_s) or not (args.watch_s == 0 or 1 <= args.watch_s <= 30):
                    raise ValueError("watch-s must be 0 (disabled) or [1,30]")
                request['watch_s'] = args.watch_s
            log = args.log
        elif args.command == "record":
            if not math.isfinite(args.duration_s) or not 3 <= args.duration_s <= 3600:
                raise ValueError("duration-s must be finite, [3,3600]")
            request["duration_s"] = args.duration_s
            request["vr_inputs"] = args.vr_inputs
            log = args.log
        else:
            if args.output.suffix != ".json":
                raise ValueError("Capture output must use .json extension")
            log = args.output.with_suffix(".jsonl")
            if log == args.output:
                raise ValueError("Capture output must not use .jsonl extension")
        result = run_remote(args, request, log)
        if args.command == "capture":
            contents = json.dumps(result, indent=2, allow_nan=False) + "\n"
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", dir=args.output.parent,
                                                 prefix=args.output.name + ".", suffix=".tmp",
                                                 delete=False) as stream:
                    temporary = Path(stream.name)
                    stream.write(contents)
                temporary.replace(args.output)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            print(f"Captured: {args.output.resolve()} (source={result['baseline_source']})")
        return 0
    except (ValueError, KeyError, OSError, RuntimeError, ET.ParseError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
