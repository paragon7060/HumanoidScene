#!/usr/bin/env python3
"""Record S63 gripper ROS messages over SSH without sending robot commands.

Uses local standard-library Python and the remote ROS Noetic workspace.
The output is JSONL, with ROS header time and remote recorder receipt times.
"""

import argparse
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys


REMOTE_RECORDER = r'''
import json
import signal
import queue
import sys
import threading
import time
import rospy
from kuavo_msgs.msg import lejuClawCommand, lejuClawState

duration = float(sys.argv[1])
camera_sides = json.loads(sys.argv[2])
lock = threading.Lock()
counts = {"/leju_claw_command": 0, "/leju_claw_state": 0}
output_queue = queue.Queue()

def writer():
    maximum_lag = 0.0
    try:
        while True:
            row = output_queue.get()
            if row is None: break
            now = time.time()
            lag = max(0.0,now-row.get("receipt_unix_s",now))
            maximum_lag = max(maximum_lag,lag)
            row["sender_unix_s"] = now
            row["sender_queue_lag_s"] = lag
            if row["kind"] == "summary":row["max_sender_queue_lag_s"] = maximum_lag
            print(json.dumps(row),flush=True)
    except (BrokenPipeError,OSError):
        rospy.signal_shutdown("SSH output closed")

writer_thread = threading.Thread(target=writer,daemon=True)
writer_thread.start()

def emit(row):
    output_queue.put(row)

rospy.init_node("gripper_measurement_recorder", anonymous=True, disable_signals=True)
signal.signal(signal.SIGTERM, lambda *_: rospy.signal_shutdown("recorder stopped"))
signal.signal(signal.SIGINT, lambda *_: rospy.signal_shutdown("recorder stopped"))
emit({"kind": "metadata", "robot_version": rospy.get_param("/robot_version", None),
      "duration_s": duration, "receipt_clock": "remote Unix seconds and monotonic seconds"})

def receive(msg, topic):
    row = {"kind": "message", "topic": topic, "receipt_unix_s": time.time(),
           "receipt_monotonic_s": time.monotonic(),
           "header_secs": msg.header.stamp.secs, "header_nsecs": msg.header.stamp.nsecs,
           "header_seq": msg.header.seq,
           "name": list(msg.data.name), "position": list(msg.data.position),
           "velocity": list(msg.data.velocity), "effort": list(msg.data.effort)}
    if topic == "/leju_claw_state":
        row["state"] = list(msg.state)
    with lock:
        counts[topic] += 1
    emit(row)

subscriptions = [
    rospy.Subscriber("/leju_claw_command", lejuClawCommand, receive,
                     callback_args="/leju_claw_command", queue_size=1000),
    rospy.Subscriber("/leju_claw_state", lejuClawState, receive,
                     callback_args="/leju_claw_state", queue_size=1000),
]
if camera_sides:
    import base64
    from sensor_msgs.msg import CompressedImage, CameraInfo
    info_seen = set()
    def receive_image(msg, side):
        # Capture callback time before serialization or stdout lock contention.
        row = {"kind": "camera_frame", "side": side,
               "topic": "/"+side+"_wrist_camera/color/image_raw/compressed",
               "receipt_unix_s": time.time(), "receipt_monotonic_s": time.monotonic(),
               "header_secs": msg.header.stamp.secs, "header_nsecs": msg.header.stamp.nsecs,
               "header_seq": msg.header.seq, "frame_id": msg.header.frame_id,
               "format": msg.format, "encoded_bytes": len(msg.data),
               "image_base64": base64.b64encode(msg.data).decode("ascii")}
        with lock:
            counts[row["topic"]] += 1
        emit(row)
    def receive_info(msg, side):
        with lock:
            if side in info_seen: return
            info_seen.add(side)
            emit({"kind":"camera_info", "side":side,
                "topic":"/"+side+"_wrist_camera/color/camera_info",
                "receipt_unix_s":time.time(),"header_secs":msg.header.stamp.secs,
                "header_nsecs":msg.header.stamp.nsecs,"frame_id":msg.header.frame_id,
                "width":msg.width,"height":msg.height,"K":list(msg.K),"D":list(msg.D),
                "distortion_model":msg.distortion_model})
    for side in camera_sides:
        topic="/"+side+"_wrist_camera/color/image_raw/compressed"
        counts[topic]=0
        subscriptions.extend([
            rospy.Subscriber(topic,CompressedImage,receive_image,callback_args=side,queue_size=8),
            rospy.Subscriber("/"+side+"_wrist_camera/color/camera_info",CameraInfo,receive_info,
                             callback_args=side,queue_size=4)])
start = time.monotonic()
try:
    while not rospy.is_shutdown() and time.monotonic() - start < duration:
        time.sleep(0.05)
finally:
    for subscription in subscriptions:
        subscription.unregister()
    emit({"kind": "summary", "elapsed_s": time.monotonic() - start, "counts": counts})
    rospy.signal_shutdown("measurement complete")
    output_queue.put(None)
    writer_thread.join()
'''


def save_camera_frame(row, output):
    """Store original encoded bytes; JSONL keeps frame time and relative filename."""
    if row.get("kind") != "camera_frame": return row
    side = row["side"]
    if side not in ("left", "right"): raise ValueError("Unexpected camera side")
    extension = "png" if "png" in row["format"].lower() else "jpg"
    folder = output / "images" / side
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{row['header_secs']}_{row['header_nsecs']:09d}_{row['header_seq']}.{extension}"
    payload = base64.b64decode(row.pop("image_base64"), validate=True)
    if len(payload) != row["encoded_bytes"]: raise ValueError("Image payload length mismatch")
    # Preserve duplicate header frames too; do not silently overwrite evidence.
    counter = 1
    original = path
    while path.exists():
        path = original.with_name(original.stem+f"_duplicate_{counter}"+original.suffix)
        counter += 1
    path.write_bytes(payload)
    row["file"] = str(path.relative_to(output))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="lab@192.168.0.22")
    parser.add_argument("--workspace", default="/home/lab/hb/kuavo-ros-opensource")
    parser.add_argument("--ros-master", default="http://kuavo_master:11311")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--output", type=Path, required=True, help="New local output directory")
    parser.add_argument("--camera-sides", choices=("left","right"), nargs="+", default=[],
                        help="Also save native wrist JPEG/PNG frames and timestamps; read-only")
    args = parser.parse_args(argv)
    if not 0 < args.duration <= 3600:
        parser.error("--duration must be between 0 and 3600 seconds")
    if not args.host or args.host.startswith("-"):
        parser.error("--host must be an SSH destination")
    if args.output.exists():
        parser.error("Output already exists; choose a new directory")
    if len(set(args.camera_sides)) != len(args.camera_sides):
        parser.error("Do not repeat camera sides")

    remote = (
        "set -e; source /opt/ros/noetic/setup.bash; source "
        + shlex.quote(args.workspace + "/devel/setup.bash")
        + "; export ROS_MASTER_URI=" + shlex.quote(args.ros_master)
        + "; exec python3 -u -c " + shlex.quote(REMOTE_RECORDER)
        + " " + shlex.quote(str(args.duration))
        + " " + shlex.quote(json.dumps(args.camera_sides))
    )
    command = ["ssh", "-C", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", args.host,
               "bash -c " + shlex.quote(remote)]
    args.output.mkdir(parents=True)
    (args.output / "session.json").write_text(json.dumps({
        "started_utc": datetime.now(timezone.utc).isoformat(), "host": args.host,
        "workspace": args.workspace, "ros_master": args.ros_master,
        "duration_s": args.duration, "mode": "ROS subscribers only",
        "camera_sides": args.camera_sides,
        "camera_storage": "Original JPEG/PNG bytes with header/remote receipt times in messages.jsonl" if args.camera_sides else None,
    }, indent=2) + "\n")
    (args.output / "measurements.md").write_text("""# 실물 gripper 측정

개폐 방향은 실물 관찰로 기록한다. position 숫자만으로 방향을 가정하지 않는다.
벌림 폭은 같은 높이의 두 패드 내면 사이 거리로 측정한다.
정지 후 측정하고, 움직이는 손가락 사이에 측정 도구를 넣지 않는다.

## 정적 개폐 — UI/조종기 또는 control_real_gripper.py로 명령, 약 3초 정지

| 방향 | 명령 값 | 왼손 폭 mm | 오른손 폭 mm | 관찰 시각/영상 시각 | 비고 |
|---|---:|---:|---:|---|---|
| 증가 | 0 | | | | |
| 증가 | 25 | | | | |
| 증가 | 50 | | | | |
| 증가 | 75 | | | | |
| 증가 | 100 | | | | |
| 감소 | 100 | | | | |
| 감소 | 75 | | | | |
| 감소 | 50 | | | | |
| 감소 | 25 | | | | |
| 감소 | 0 | | | | |

## 시간 응답 — 가능한 경우 같은 위치 전환을 속도 25/50/75로 각각 3회

| 반복 | 손 | 시작→목표 | 속도 명령 | effort 명령 | 실제 이동 시작/도달 시각 | 비고 |
|---:|---|---|---:|---:|---|---|
| | | | | | | |

숫자 입력 UI가 없으면 real_robot_calibration/s63_gripper/control_real_gripper.py로 값을 지정한다.
--send는 실제 동작 명령이며 기본 미리보기와 --check는 동작을 보내지 않는다.
명령 기록 --log 경로를 이 출력 폴더의 commands.jsonl로 지정한다.
이 기록기는 robot command를 전송하지 않는다.
서비스를 통해 명령하면 command topic에 메시지가 없을 수 있다.
그 경우 명령 값과 시각을 표 또는 영상에 별도로 남긴다.
""")
    destination = args.output / "messages.jsonl"
    print(f"Recording to {destination.resolve()}", flush=True)
    process = None
    counts = {}
    try:
        with destination.open("x") as output:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
            for line in process.stdout:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    print("Remote non-JSON output: " + line.rstrip(), file=sys.stderr)
                    continue
                if row.get("kind") == "camera_frame":
                    row = save_camera_frame(row, args.output)
                    output.write(json.dumps(row)+"\n")
                else:
                    output.write(line)
                output.flush()
                if row.get("kind") == "summary":
                    counts = row["counts"]
            code = process.wait()
    except KeyboardInterrupt:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        print("Stopped; partial messages saved.", file=sys.stderr)
        return 130
    if code:
        print("Recording failed; partial output retained.", file=sys.stderr)
        return code
    print(json.dumps(counts, ensure_ascii=False), flush=True)
    if not counts.get("/leju_claw_state", 0):
        print("No state messages received; check ROS connection and robot state.", file=sys.stderr)
        return 1
    if not counts.get("/leju_claw_command", 0):
        print("No command-topic messages: record service/UI commands and times separately.")
    for side in args.camera_sides:
        if not counts.get("/"+side+"_wrist_camera/color/image_raw/compressed", 0):
            print(f"No {side} wrist frames received; partial logs retained.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
