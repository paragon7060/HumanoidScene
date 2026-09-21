#!/usr/bin/env python3
"""Preview or send one numeric S63 gripper setpoint through the existing ROS service.

--position 25 addresses both claws; --position 25 50 is left/right respectively.
Only --send invokes the control service. --check only reads service/state information.
"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


REMOTE_CONTROL = r'''
import json
import sys
import time
import rospy
import rosservice
from kuavo_msgs.msg import lejuClawState
from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

options = json.loads(sys.argv[1])
def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, receipt_unix_s=time.time(), **fields)), flush=True)

rospy.init_node("gripper_numeric_control", anonymous=True, disable_signals=True)
version = rospy.get_param("/robot_version", None)
if str(version) != "63":
    raise RuntimeError("Expected robot_version=63, got " + repr(version))
service = "/control_robot_leju_claw"
rospy.wait_for_service(service, timeout=5)
service_type = rosservice.get_service_type(service)
if service_type != "kuavo_msgs/controlLejuClaw":
    raise RuntimeError("Unexpected service type: " + repr(service_type))
before = rospy.wait_for_message("/leju_claw_state", lejuClawState, timeout=3)
emit("connection", robot_version=version, service=service, service_type=service_type,
     position=list(before.data.position), name=list(before.data.name), state=list(before.state))
if not options["send"]:
    sys.exit(0)

request = controlLejuClawRequest()
request.data.name = ["left_claw", "right_claw"]
request.data.position = options["position"]
request.data.velocity = [options["velocity"]] * 2
request.data.effort = [options["effort"]] * 2
emit("command_attempt", name=list(request.data.name), position=list(request.data.position),
     velocity=list(request.data.velocity), effort=list(request.data.effort))
result = rospy.ServiceProxy(service, controlLejuClaw)(request)
emit("service_response", success=result.success, message=result.message)
sys.exit(0 if result.success else 1)
'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--position", nargs="+", type=float, help="One value for both claws, or left right (0–100)")
    parser.add_argument("--velocity", type=float, default=25.0, help="Speed command 0–100 (default: 25)")
    parser.add_argument("--effort", type=float, default=1.0, help="Current command in A (default: 1)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--send", action="store_true", help="Send one setpoint: this moves the real gripper")
    mode.add_argument("--check", action="store_true", help="Read service type and current state only")
    parser.add_argument("--host", default=os.environ.get("KUAVO_ROBOT_SSH"))
    parser.add_argument("--workspace", default=os.environ.get("KUAVO_ROBOT_WORKSPACE"))
    parser.add_argument("--ros-master", default="http://kuavo_master:11311")
    parser.add_argument("--log", type=Path,
                        default=Path(__file__).resolve().parent / "data/real_gripper_commands.jsonl")
    args = parser.parse_args(argv)
    if not args.host or args.host.startswith("-"):
        parser.error("Set --host or KUAVO_ROBOT_SSH to an SSH destination")
    if not args.workspace:
        parser.error("Set --workspace or KUAVO_ROBOT_WORKSPACE")
    if args.position is None and not args.check:
        parser.error("--position is required unless using --check")
    if args.position is not None:
        if len(args.position) not in (1, 2) or any(not math.isfinite(p) or not 0 <= p <= 100 for p in args.position):
            parser.error("Provide one or two finite position values between 0 and 100")
        if len(args.position) == 1:
            args.position *= 2
    if not math.isfinite(args.velocity) or not 0 <= args.velocity <= 100:
        parser.error("--velocity must be between 0 and 100")
    if not math.isfinite(args.effort) or not 0 < args.effort <= 2.5:
        parser.error("--effort must be greater than 0 and at most 2.5 A")
    payload = {"send": args.send, "position": args.position,
               "velocity": args.velocity, "effort": args.effort}
    if not args.send and not args.check:
        print(json.dumps({"mode": "preview", "host": args.host,
                          "name": ["left_claw", "right_claw"], **payload}, indent=2))
        print("Preview only. Add --send to send this setpoint to the real gripper.")
        return 0

    remote = (
        "set -e; source /opt/ros/noetic/setup.bash; source "
        + shlex.quote(args.workspace + "/devel/setup.bash")
        + "; export ROS_MASTER_URI=" + shlex.quote(args.ros_master)
        + "; exec python3 -u -c " + shlex.quote(REMOTE_CONTROL)
        + " " + shlex.quote(json.dumps(payload))
    )
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host,
               "bash -c " + shlex.quote(remote)]
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("a") as log:
        log.write(json.dumps({"kind": "local_attempt", "started_utc": datetime.now(timezone.utc).isoformat(),
                              "host": args.host, "mode": "send" if args.send else "check", **payload}) + "\n")
        log.flush()
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, text=True, timeout=45)
        except subprocess.TimeoutExpired as exc:
            partial = exc.stdout or ""
            if isinstance(partial, bytes):
                partial = partial.decode(errors="replace")
            for line in partial.splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                log.write(json.dumps(row) + "\n")
            log.write(json.dumps({"kind": "timeout", "send": args.send}) + "\n")
            print("Timed out; command outcome may be unknown. No retry was sent.", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            log.write(json.dumps({"kind": "interrupted", "send": args.send}) + "\n")
            print("Interrupted; this does not cancel motion already accepted by the robot.", file=sys.stderr)
            return 130
        for line in result.stdout.splitlines():
            print(line)
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            log.write(json.dumps(row) + "\n")
        if result.returncode:
            print("Check failed or command outcome unsuccessful/unknown; no retry was sent.", file=sys.stderr)
        elif args.send:
            print("Service accepted the command. Verify settling with the recorder and real gripper.")
        return result.returncode


if __name__ == "__main__":
    sys.exit(main())
