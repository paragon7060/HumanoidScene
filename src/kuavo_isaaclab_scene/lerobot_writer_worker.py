#!/usr/bin/env python3
"""Isolated LeRobot Dataset v3 writer process used by Isaac Lab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import struct
import traceback
from typing import Any, BinaryIO

import numpy as np


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _read_message(stream: BinaryIO) -> Any:
    size = struct.unpack("!Q", _read_exact(stream, 8))[0]
    return pickle.loads(_read_exact(stream, size))


def _write_message(stream: BinaryIO, message: Any) -> None:
    payload = pickle.dumps(message, protocol=5)
    stream.write(struct.pack("!Q", len(payload)))
    stream.write(payload)


def _validate_schema(dataset: Any, expected_features: dict[str, dict], fps: int) -> None:
    if int(dataset.fps) != int(fps):
        raise ValueError(f"Existing dataset fps={dataset.fps}, requested fps={fps}.")
    for key, expected in expected_features.items():
        actual = dataset.features.get(key)
        if actual is None:
            raise ValueError(f"Existing dataset is missing feature {key!r}.")
        if actual["dtype"] != expected["dtype"] or tuple(actual["shape"]) != tuple(expected["shape"]):
            raise ValueError(
                f"Existing feature {key!r} is {actual}, expected {expected}. "
                "Use a new --lerobot-root after changing camera resolution or object counts."
            )


def _episode_buffer(dataset: Any) -> dict:
    writer = getattr(dataset, "writer", None)
    if writer is not None:
        return writer.episode_buffer
    return dataset.episode_buffer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--command-fd", required=True, type=int)
    parser.add_argument("--response-fd", required=True, type=int)
    args = parser.parse_args()
    commands = open(args.command_fd, "rb", buffering=0, closefd=True)
    responses = open(args.response_fd, "wb", buffering=0, closefd=True)
    dataset = None
    recording = False
    sample_count = 0
    episode_metadata: dict[str, Any] = {}
    save_failed = False
    root: Path | None = None

    try:
        while True:
            try:
                message = _read_message(commands)
            except EOFError:
                break
            op = message.get("op", "unknown")
            try:
                if op == "init":
                    import lerobot
                    from lerobot.datasets import CODEBASE_VERSION, LeRobotDataset

                    if str(CODEBASE_VERSION) != "v3.0":
                        raise RuntimeError(
                            f"This writer requires LeRobot Dataset v3.0, found {CODEBASE_VERSION} "
                            f"in lerobot {lerobot.__version__}."
                        )
                    root = Path(message["root"])
                    info_path = root / "meta" / "info.json"
                    if info_path.is_file():
                        dataset = LeRobotDataset.resume(
                            repo_id=message["repo_id"],
                            root=root,
                            batch_encoding_size=1,
                        )
                    else:
                        if root.exists():
                            raise FileExistsError(
                                f"LeRobot root exists without meta/info.json: {root}. "
                                "Select a new path or move the incomplete directory."
                            )
                        dataset = LeRobotDataset.create(
                            repo_id=message["repo_id"],
                            fps=int(message["fps"]),
                            features=message["features"],
                            root=root,
                            robot_type="kuavo_s63",
                            use_videos=bool(message["use_videos"]),
                            batch_encoding_size=1,
                        )
                    _validate_schema(dataset, message["features"], int(message["fps"]))
                    save_failed = bool(message["save_failed"])
                    _write_message(
                        responses,
                        {
                            "ok": True,
                            "op": op,
                            "total_episodes": int(dataset.meta.total_episodes),
                            "lerobot_version": str(lerobot.__version__),
                            "dataset_version": str(CODEBASE_VERSION),
                        },
                    )
                elif op == "start":
                    if dataset is None or recording:
                        raise RuntimeError("Writer is not initialized or an episode is already active.")
                    recording = True
                    sample_count = 0
                    episode_metadata = dict(message.get("metadata") or {})
                    name = f"episode_{int(dataset.meta.total_episodes):06d}"
                    _write_message(responses, {"ok": True, "op": op, "name": name})
                elif op == "append":
                    if dataset is None or not recording:
                        raise RuntimeError("No active episode.")
                    dataset.add_frame(message["frame"])
                    sample_count += 1
                elif op == "finish":
                    if dataset is None or not recording:
                        raise RuntimeError("No active episode.")
                    success = bool(message["success"])
                    saved = sample_count > 0 and (success or save_failed)
                    name = f"episode_{int(dataset.meta.total_episodes):06d}" if saved else None
                    if saved:
                        buffer = _episode_buffer(dataset)
                        buffer["next.done"][-1] = np.ones(1, dtype=np.float32)
                        buffer["next.success"][-1] = np.asarray([success], dtype=np.float32)
                        dataset.save_episode()
                        sidecar = root / "meta" / "kuavo_episode_metadata.jsonl"
                        record = {
                            "episode": name,
                            "success": success,
                            "end_reason": message["reason"],
                            "num_frames": sample_count,
                            **episode_metadata,
                        }
                        with sidecar.open("a", encoding="utf-8") as stream:
                            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                    else:
                        dataset.clear_episode_buffer(delete_images=True)
                    recording = False
                    sample_count = 0
                    episode_metadata = {}
                    _write_message(
                        responses,
                        {
                            "ok": True,
                            "op": op,
                            "name": name,
                            "total_episodes": int(dataset.meta.total_episodes),
                        },
                    )
                elif op == "close":
                    if recording:
                        dataset.clear_episode_buffer(delete_images=True)
                        recording = False
                    dataset.finalize()
                    _write_message(responses, {"ok": True, "op": op})
                    return 0
                else:
                    raise ValueError(f"Unknown writer operation: {op}")
            except Exception as exc:
                _write_message(
                    responses,
                    {
                        "ok": False,
                        "op": op,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc(),
                    },
                )
                return 1
    finally:
        if dataset is not None:
            try:
                if recording:
                    dataset.clear_episode_buffer(delete_images=True)
                dataset.finalize()
            except Exception:
                pass
        commands.close()
        responses.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
