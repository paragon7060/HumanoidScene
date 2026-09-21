"""Portable, dependency-light deployment trajectory format."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from .contract import ARM_JOINT_NAMES


KINDS = {"joint_position_rad", "normalized_joint_delta"}


@dataclass(frozen=True)
class Trajectory:
    kind: str
    timestamps_s: np.ndarray
    arm_values: np.ndarray
    gripper_close: Optional[np.ndarray]
    metadata: Dict[str, Any]

    def validate(self) -> "Trajectory":
        timestamps = np.asarray(self.timestamps_s, dtype=np.float64)
        arms = np.asarray(self.arm_values, dtype=np.float64)
        if self.kind not in KINDS:
            raise ValueError("Unsupported trajectory kind: {}".format(self.kind))
        if timestamps.ndim != 1 or len(timestamps) < 1:
            raise ValueError("timestamps_s must be a non-empty vector")
        if arms.shape != (len(timestamps), len(ARM_JOINT_NAMES)):
            raise ValueError("arm_values must have shape (samples, 14)")
        if not np.isfinite(timestamps).all() or not np.isfinite(arms).all():
            raise ValueError("Trajectory values must be finite")
        if timestamps[0] < 0.0 or (len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0.0)):
            raise ValueError("Trajectory timestamps must start at >=0 and strictly increase")
        if self.kind == "normalized_joint_delta" and np.max(np.abs(arms)) > 1.0 + 1e-7:
            raise ValueError("Normalized RL arm actions must stay in [-1, 1]")
        if tuple(self.metadata.get("arm_joint_names", ())) != ARM_JOINT_NAMES:
            raise ValueError("Trajectory arm joint order is not the verified S63 order")
        if self.metadata.get("robot_model") != "s63":
            raise ValueError("Trajectory is not explicitly pinned to S63")
        if self.gripper_close is not None:
            gripper = np.asarray(self.gripper_close, dtype=np.float64)
            if gripper.shape != (len(timestamps), 2):
                raise ValueError("gripper_close must have shape (samples, 2)")
            if not np.isfinite(gripper).all() or not np.isin(gripper, (0.0, 1.0)).all():
                raise ValueError("Gripper commands must use binary 0=open, 1=close")
        return self

    @property
    def duration_s(self) -> float:
        return float(self.timestamps_s[-1])

    def absolute_targets(self, initial_arm_rad: np.ndarray, delta_scale_rad: float) -> np.ndarray:
        """Resolve deltas exactly once per policy sample, never per ROS callback."""
        self.validate()
        initial = np.asarray(initial_arm_rad, dtype=np.float64)
        if initial.shape != (14,) or not np.isfinite(initial).all():
            raise ValueError("initial_arm_rad must contain 14 finite values")
        if not math.isfinite(delta_scale_rad) or delta_scale_rad <= 0.0:
            raise ValueError("delta_scale_rad must be finite and positive")
        if self.kind == "joint_position_rad":
            return np.asarray(self.arm_values, dtype=np.float64).copy()
        return initial[None, :] + np.cumsum(self.arm_values * delta_scale_rad, axis=0)


def _metadata_text(value: np.ndarray) -> str:
    item = value.item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def load_trajectory(path: Path) -> Trajectory:
    path = Path(path)
    with np.load(str(path), allow_pickle=False) as payload:
        required = {"timestamps_s", "arm_values", "metadata_json"}
        if not required.issubset(payload.files):
            raise ValueError("Trajectory is missing {}".format(sorted(required - set(payload.files))))
        metadata = json.loads(_metadata_text(payload["metadata_json"]))
        gripper = np.asarray(payload["gripper_close"], dtype=np.float64) if "gripper_close" in payload else None
        trajectory = Trajectory(
            kind=str(metadata.get("kind", "")),
            timestamps_s=np.asarray(payload["timestamps_s"], dtype=np.float64),
            arm_values=np.asarray(payload["arm_values"], dtype=np.float64),
            gripper_close=gripper,
            metadata=metadata,
        )
    return trajectory.validate()


def save_trajectory(path: Path, trajectory: Trajectory) -> None:
    path = Path(path)
    trajectory.validate()
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "timestamps_s": np.asarray(trajectory.timestamps_s, dtype=np.float64),
        "arm_values": np.asarray(trajectory.arm_values, dtype=np.float64),
        "metadata_json": np.asarray(json.dumps(trajectory.metadata, sort_keys=True, allow_nan=False)),
    }
    if trajectory.gripper_close is not None:
        arrays["gripper_close"] = np.asarray(trajectory.gripper_close, dtype=np.uint8)
    np.savez_compressed(str(path), **arrays)
