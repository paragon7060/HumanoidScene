#!/usr/bin/env python3
"""Replay one validated Task1 cuMotion path and record the editor stream."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
import websockets

from kuavo_isaaclab_scene.teleop.browser_teleop_bridge import (
    PROTOCOL_VERSION,
    unpack_frame_packet,
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", type=Path, required=True)
    result.add_argument("--video-out", type=Path, required=True)
    result.add_argument("--websocket-url", default="ws://127.0.0.1:8765")
    result.add_argument("--fps", type=float, default=30.0)
    result.add_argument("--view", default="front_left")
    result.add_argument("--hold-before-s", type=float, default=1.0)
    result.add_argument("--hold-after-s", type=float, default=1.0)
    return result


def validated_plan(path: Path) -> tuple[dict, list[str], np.ndarray]:
    report = json.loads(path.expanduser().resolve().read_text())
    if report.get("status") != "SUCCESS":
        raise ValueError("refusing to replay a plan whose status is not SUCCESS")
    if report.get("sampled_world_collision") or report.get("sampled_self_collision"):
        raise ValueError("refusing to replay a plan with a sampled collision")
    joint_names = report.get("joint_names")
    waypoints = np.asarray(report.get("waypoint_q_rad"), dtype=float)
    if (
        not isinstance(joint_names, list)
        or len(joint_names) != 14
        or waypoints.ndim != 2
        or waypoints.shape[1] != len(joint_names)
        or len(waypoints) < 2
        or not np.isfinite(waypoints).all()
    ):
        raise ValueError("plan must contain finite 14-DoF waypoints")
    return report, joint_names, waypoints


def ffmpeg_process(path: Path, fps: float) -> subprocess.Popen:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to record the replay")
    path = path.expanduser().resolve()
    if path.suffix.lower() != ".mp4":
        raise ValueError("--video-out must use the .mp4 suffix")
    if path.exists():
        raise FileExistsError(f"video output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "image2pipe",
            "-framerate",
            f"{fps:.8g}",
            "-vcodec",
            "mjpeg",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


async def replay_and_record(args, joint_names: list[str], waypoints: np.ndarray) -> dict:
    if not math.isfinite(args.fps) or args.fps <= 0:
        raise ValueError("--fps must be finite and positive")
    for name in ("hold_before_s", "hold_after_s"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")

    video_path = args.video_out.expanduser().resolve()
    process = ffmpeg_process(video_path, args.fps) if shutil.which("ffmpeg") else None
    cv2 = None
    cv_writer = None
    recorder_backend = "ffmpeg/libx264" if process is not None else "opencv/mp4v"
    if process is not None and process.stdin is None:
        raise RuntimeError("ffmpeg stdin is unavailable")
    if process is None:
        import cv2 as cv2_module

        if video_path.suffix.lower() != ".mp4":
            raise ValueError("--video-out must use the .mp4 suffix")
        if video_path.exists():
            raise FileExistsError(f"video output already exists: {video_path}")
        video_path.parent.mkdir(parents=True, exist_ok=True)
        cv2 = cv2_module
    frame_count = 0
    first_frame_sequence = None
    last_frame_sequence = None
    started = time.monotonic()
    stop_at = None
    sequence = 0

    try:
        async with websockets.connect(args.websocket_url, max_size=None) as socket:
            await socket.send(
                json.dumps({"type": "client_hello", "protocol_version": PROTOCOL_VERSION})
            )
            sequence += 1
            await socket.send(
                json.dumps(
                    {
                        "type": "pose_editor",
                        "protocol_version": PROTOCOL_VERSION,
                        "sequence": sequence,
                        "action": "reset",
                    }
                )
            )
            sequence += 1
            await socket.send(
                json.dumps(
                    {
                        "type": "pose_editor",
                        "protocol_version": PROTOCOL_VERSION,
                        "sequence": sequence,
                        "action": "set_view",
                        "view": args.view,
                    }
                )
            )

            async def sender() -> None:
                nonlocal sequence, stop_at
                await asyncio.sleep(args.hold_before_s)
                interval = 1.0 / args.fps
                next_send = time.monotonic()
                for row in waypoints:
                    sequence += 1
                    await socket.send(
                        json.dumps(
                            {
                                "type": "pose_editor",
                                "protocol_version": PROTOCOL_VERSION,
                                "sequence": sequence,
                                "action": "set_pose",
                                "joint_positions": dict(zip(joint_names, row.tolist())),
                            }
                        )
                    )
                    next_send += interval
                    await asyncio.sleep(max(0.0, next_send - time.monotonic()))
                await asyncio.sleep(args.hold_after_s)
                stop_at = time.monotonic()

            send_task = asyncio.create_task(sender())
            while stop_at is None:
                message = await asyncio.wait_for(socket.recv(), timeout=5.0)
                if not isinstance(message, bytes):
                    continue
                metadata, jpeg = unpack_frame_packet(message)
                if process is not None:
                    process.stdin.write(jpeg)
                else:
                    frame = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise RuntimeError("failed to decode one editor JPEG frame")
                    if cv_writer is None:
                        height, width = frame.shape[:2]
                        cv_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            args.fps,
                            (width, height),
                        )
                        if not cv_writer.isOpened():
                            raise RuntimeError("OpenCV could not open the MP4 video writer")
                    elif frame.shape[1] != width or frame.shape[0] != height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                    cv_writer.write(frame)
                frame_count += 1
                frame_sequence = int(metadata["frame_sequence"])
                first_frame_sequence = (
                    frame_sequence if first_frame_sequence is None else first_frame_sequence
                )
                last_frame_sequence = frame_sequence
            await send_task
    finally:
        if process is not None:
            process.stdin.close()
            stderr = (
                process.stderr.read().decode("utf-8", errors="replace")
                if process.stderr
                else ""
            )
            return_code = process.wait(timeout=30)
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed with exit code {return_code}: {stderr}")
        elif cv_writer is not None:
            cv_writer.release()

    return {
        "video": str(video_path),
        "frame_count": frame_count,
        "first_frame_sequence": first_frame_sequence,
        "last_frame_sequence": last_frame_sequence,
        "wall_duration_s": time.monotonic() - started,
        "waypoint_count": len(waypoints),
        "command_fps": args.fps,
        "view": args.view,
        "recorder_backend": recorder_backend,
    }


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    report, joint_names, waypoints = validated_plan(args.plan)
    result = asyncio.run(replay_and_record(args, joint_names, waypoints))
    result.update(
        {
            "plan": str(args.plan.expanduser().resolve()),
            "terminal_error_m": report.get("terminal_error_m"),
            "terminal_closing_axis_error_deg": report.get(
                "terminal_closing_axis_error_deg"
            ),
        }
    )
    metadata_path = args.video_out.expanduser().resolve().with_suffix(".json")
    metadata_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
