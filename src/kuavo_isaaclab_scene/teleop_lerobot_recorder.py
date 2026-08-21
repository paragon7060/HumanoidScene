"""LeRobot Dataset v3 episode writer client for Kuavo Quest teleoperation."""

from __future__ import annotations

import os
from pathlib import Path
import pickle
import struct
import subprocess
from typing import Any, BinaryIO, Sequence

import numpy as np


ACTION_NAMES = [
    "left_dx",
    "left_dy",
    "left_dz",
    "left_droll",
    "left_dpitch",
    "left_dyaw",
    "right_dx",
    "right_dy",
    "right_dz",
    "right_droll",
    "right_dpitch",
    "right_dyaw",
    "head_yaw",
    "head_pitch",
]
POSE_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]


def _vector_feature(size: int, names: Sequence[str] | None = None, *, dtype: str = "float32") -> dict:
    return {
        "dtype": dtype,
        "shape": (size,),
        "names": list(names) if names is not None else None,
    }


def build_lerobot_features(
    *,
    joint_names: Sequence[str],
    hand_joint_names: Sequence[str],
    head_resolution: tuple[int, int],
    wrist_resolution: tuple[int, int],
    box_count: int,
    button_joint_count: int,
    record_wrist_cameras: bool,
    use_videos: bool,
) -> dict[str, dict]:
    """Return the fixed schema used for every episode in one dataset."""
    image_dtype = "video" if use_videos else "image"
    head_width, head_height = head_resolution
    wrist_width, wrist_height = wrist_resolution
    ee_names = [f"{side}_{field}" for side in ("left", "right") for field in POSE_NAMES]
    features: dict[str, dict] = {
        "observation.state": _vector_feature(len(joint_names), joint_names),
        "observation.velocity": _vector_feature(len(joint_names), joint_names),
        "observation.ee_pose": _vector_feature(14, ee_names),
        "observation.openxr.head_pose": _vector_feature(7, POSE_NAMES),
        "observation.openxr.left_hand": _vector_feature(
            len(hand_joint_names) * 7,
            [f"{joint}_{field}" for joint in hand_joint_names for field in POSE_NAMES],
        ),
        "observation.openxr.right_hand": _vector_feature(
            len(hand_joint_names) * 7,
            [f"{joint}_{field}" for joint in hand_joint_names for field in POSE_NAMES],
        ),
        "observation.pinch_distance": _vector_feature(2, ["left_m", "right_m"]),
        "observation.tracking_valid": _vector_feature(3, ["left", "right", "head"]),
        "observation.sim_time": _vector_feature(1, ["seconds"]),
        "observation.images.head": {
            "dtype": image_dtype,
            "shape": (head_height, head_width, 3),
            "names": ["height", "width", "channels"],
        },
        "action": _vector_feature(14, ACTION_NAMES),
        "next.done": _vector_feature(1, ["done"]),
        "next.success": _vector_feature(1, ["success"]),
    }
    if record_wrist_cameras:
        for side in ("left", "right"):
            features[f"observation.images.{side}_wrist"] = {
                "dtype": image_dtype,
                "shape": (wrist_height, wrist_width, 3),
                "names": ["height", "width", "channels"],
            }
    if box_count:
        features["observation.box_root_pose"] = _vector_feature(
            box_count * 7,
            [f"box_{index}_{field}" for index in range(box_count) for field in POSE_NAMES],
        )
    if button_joint_count:
        features["observation.button_joint_position"] = _vector_feature(
            button_joint_count,
            [f"joint_{index}" for index in range(button_joint_count)],
        )
    return features


def sample_to_lerobot_frame(sample: dict[str, Any], features: dict[str, dict]) -> dict[str, np.ndarray]:
    """Map the rich internal sample to policy-friendly LeRobot feature names."""
    frame: dict[str, np.ndarray] = {
        "observation.state": np.asarray(sample["robot_joint_position"], dtype=np.float32),
        "observation.velocity": np.asarray(sample["robot_joint_velocity"], dtype=np.float32),
        "observation.ee_pose": np.concatenate(
            [sample["left_end_effector_pose_w"], sample["right_end_effector_pose_w"]]
        ).astype(np.float32),
        "observation.openxr.head_pose": np.asarray(sample["openxr_head_pose"], dtype=np.float32),
        "observation.openxr.left_hand": np.asarray(sample["openxr_left_hand"], dtype=np.float32).reshape(-1),
        "observation.openxr.right_hand": np.asarray(sample["openxr_right_hand"], dtype=np.float32).reshape(-1),
        "observation.pinch_distance": np.asarray(sample["pinch_distance_m"], dtype=np.float32),
        "observation.tracking_valid": np.asarray(sample["tracking_valid"], dtype=np.float32),
        "observation.sim_time": np.asarray([sample["sim_time_s"]], dtype=np.float32),
        "observation.images.head": np.asarray(sample["head_rgb"], dtype=np.uint8),
        "action": np.asarray(sample["action"], dtype=np.float32),
        "next.done": np.zeros(1, dtype=np.float32),
        "next.success": np.zeros(1, dtype=np.float32),
    }
    optional_mappings = {
        "observation.images.left_wrist": "left_wrist_rgb",
        "observation.images.right_wrist": "right_wrist_rgb",
        "observation.box_root_pose": "box_root_pose_w",
        "observation.button_joint_position": "button_joint_position",
    }
    for feature_name, sample_name in optional_mappings.items():
        if feature_name in features:
            dtype = np.uint8 if features[feature_name]["dtype"] in {"image", "video"} else np.float32
            frame[feature_name] = np.asarray(sample[sample_name], dtype=dtype).reshape(features[feature_name]["shape"])
    return frame


class LeRobotTeleopRecorder:
    """Send samples to an isolated LeRobot Dataset v3 writer process."""

    def __init__(
        self,
        root: str | Path,
        *,
        repo_id: str,
        fps: int,
        task: str,
        joint_names: Sequence[str],
        hand_joint_names: Sequence[str],
        head_resolution: tuple[int, int],
        wrist_resolution: tuple[int, int],
        box_count: int,
        button_joint_count: int,
        record_wrist_cameras: bool = True,
        use_videos: bool = True,
        save_failed: bool = False,
        writer_python: str | Path | None = None,
    ):
        if fps <= 0:
            raise ValueError("LeRobot fps must be positive.")
        self.root = Path(root).expanduser().resolve()
        self.repo_id = repo_id
        self.fps = int(fps)
        self.task = task
        self.save_failed = bool(save_failed)
        if writer_python is None:
            writer_python = os.environ.get("LEROBOT_PYTHON")
        if writer_python is None:
            raise ValueError("Set writer_python or LEROBOT_PYTHON to a LeRobot Dataset v3 Python executable.")
        self.writer_python = Path(writer_python).expanduser().resolve()
        if not self.writer_python.is_file():
            raise FileNotFoundError(f"LeRobot v3 Python does not exist: {self.writer_python}")
        self.features = build_lerobot_features(
            joint_names=joint_names,
            hand_joint_names=hand_joint_names,
            head_resolution=head_resolution,
            wrist_resolution=wrist_resolution,
            box_count=box_count,
            button_joint_count=button_joint_count,
            record_wrist_cameras=record_wrist_cameras,
            use_videos=use_videos,
        )
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._process, self._commands, self._responses = self._start_worker()
        self._recording = False
        self._sample_count = 0
        init = self._request(
            {
                "op": "init",
                "root": str(self.root),
                "repo_id": self.repo_id,
                "fps": self.fps,
                "task": self.task,
                "features": self.features,
                "use_videos": bool(use_videos),
                "save_failed": self.save_failed,
            }
        )
        self._episode_count = int(init["total_episodes"])
        self.lerobot_version = str(init["lerobot_version"])
        self.dataset_version = str(init["dataset_version"])

    def _start_worker(self) -> tuple[subprocess.Popen, BinaryIO, BinaryIO]:
        worker_path = Path(__file__).with_name("lerobot_writer_worker.py")
        command_read_fd, command_write_fd = os.pipe()
        response_read_fd, response_write_fd = os.pipe()
        process = subprocess.Popen(
            [
                str(self.writer_python),
                "-I",
                "-u",
                str(worker_path),
                "--command-fd",
                str(command_read_fd),
                "--response-fd",
                str(response_write_fd),
            ],
            pass_fds=(command_read_fd, response_write_fd),
        )
        os.close(command_read_fd)
        os.close(response_write_fd)
        return process, os.fdopen(command_write_fd, "wb", buffering=0), os.fdopen(response_read_fd, "rb", buffering=0)

    @staticmethod
    def _write_message(stream: BinaryIO, message: Any) -> None:
        payload = pickle.dumps(message, protocol=5)
        stream.write(struct.pack("!Q", len(payload)))
        stream.write(payload)

    @staticmethod
    def _read_exact(stream: BinaryIO, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = stream.read(remaining)
            if not chunk:
                raise EOFError("LeRobot v3 writer exited before replying.")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    @classmethod
    def _read_message(cls, stream: BinaryIO) -> Any:
        size = struct.unpack("!Q", cls._read_exact(stream, 8))[0]
        return pickle.loads(cls._read_exact(stream, size))

    def _send(self, message: dict[str, Any]) -> None:
        return_code = self._process.poll()
        if return_code is not None:
            raise RuntimeError(f"LeRobot v3 writer exited with status {return_code}.")
        self._write_message(self._commands, message)

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        self._send(message)
        response = self._read_message(self._responses)
        if not response.get("ok"):
            raise RuntimeError(
                f"LeRobot v3 writer failed during {response.get('op', message['op'])}: "
                f"{response.get('error', 'unknown error')}\n{response.get('traceback', '')}"
            )
        return response

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def start_episode(self, metadata: dict[str, Any] | None = None) -> str:
        if self.recording:
            raise RuntimeError("An episode is already being recorded.")
        response = self._request({"op": "start", "metadata": dict(metadata or {})})
        self._recording = True
        self._sample_count = 0
        return str(response["name"])

    def append(self, sample: dict[str, Any]) -> None:
        if not self.recording:
            return
        frame = sample_to_lerobot_frame(sample, self.features)
        frame["task"] = self.task
        self._send({"op": "append", "frame": frame})
        self._sample_count += 1

    def finish_episode(self, *, success: bool, reason: str) -> str | None:
        if not self.recording:
            return None
        response = self._request({"op": "finish", "success": bool(success), "reason": reason})
        name = response.get("name")
        self._episode_count = int(response["total_episodes"])
        self._recording = False
        self._sample_count = 0
        return name

    def close(self) -> None:
        if self.recording:
            self.finish_episode(success=False, reason="process_closed")
        if self._process.poll() is None:
            self._request({"op": "close"})
        self._commands.close()
        self._responses.close()
        return_code = self._process.wait(timeout=30)
        if return_code:
            raise RuntimeError(f"LeRobot v3 writer exited with status {return_code}.")
