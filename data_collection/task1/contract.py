"""Joint-order and trajectory contracts shared by the Task1 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


ARM_JOINT_NAMES = [
    f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)
]
WAIST_JOINT_NAMES = ["waist_pitch_joint", "waist_yaw_joint"]
WAIST_ARM_JOINT_NAMES = WAIST_JOINT_NAMES + ARM_JOINT_NAMES

_SAFE_WAIST_DEG = {
    "waist_pitch_joint": (-9.0, 25.0),
    "waist_yaw_joint": (-45.0, 45.0),
}


@dataclass(frozen=True)
class PlanLayout:
    joint_names: tuple[str, ...]
    waist_indices: tuple[int, ...]
    arm_indices: tuple[int, ...]


def layout_for_joint_names(
    joint_names: Sequence[str], *, allow_arm_only_baseline: bool = False
) -> PlanLayout:
    names = tuple(joint_names)
    if names == tuple(WAIST_ARM_JOINT_NAMES):
        return PlanLayout(names, (0, 1), tuple(range(2, 16)))
    if names == tuple(ARM_JOINT_NAMES):
        if not allow_arm_only_baseline:
            raise ValueError("14DoF arm-only baseline requires explicit opt-in")
        return PlanLayout(names, (), tuple(range(14)))
    raise ValueError(
        "joint_names must use the canonical waist+arms order"
        + (" or explicit arm-only baseline" if allow_arm_only_baseline else "")
    )


def validate_plan(plan: Mapping, *, allow_arm_only_baseline: bool = False) -> np.ndarray:
    layout = layout_for_joint_names(
        plan.get("joint_names", ()), allow_arm_only_baseline=allow_arm_only_baseline
    )
    key = "trajectory" if "trajectory" in plan else "waypoints"
    if key not in plan:
        raise ValueError("plan must contain trajectory or waypoints")
    trajectory = np.asarray(plan[key], dtype=float)
    if trajectory.ndim != 2 or trajectory.shape[1] != len(layout.joint_names):
        raise ValueError(f"plan trajectory must have {len(layout.joint_names)} columns")
    if not np.isfinite(trajectory).all():
        raise ValueError("plan trajectory must contain only finite values")
    return trajectory


def split_trajectory(
    joint_names: Sequence[str], trajectory: Sequence[Sequence[float]]
) -> tuple[np.ndarray, np.ndarray]:
    layout = layout_for_joint_names(joint_names, allow_arm_only_baseline=True)
    values = np.asarray(trajectory, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(layout.joint_names):
        raise ValueError(f"trajectory must have {len(layout.joint_names)} columns")
    return values[:, layout.waist_indices], values[:, layout.arm_indices]


def compose_waist_arm(
    waist: Sequence[float] | Sequence[Sequence[float]],
    arms: Sequence[Sequence[float]],
) -> np.ndarray:
    arm_values = np.asarray(arms, dtype=float)
    if arm_values.ndim != 2 or arm_values.shape[1] != len(ARM_JOINT_NAMES):
        raise ValueError("arm trajectory must have 14 columns")
    waist_values = np.asarray(waist, dtype=float)
    if waist_values.shape == (2,):
        waist_values = np.repeat(waist_values[None, :], arm_values.shape[0], axis=0)
    if waist_values.shape != (arm_values.shape[0], 2):
        raise ValueError("waist must be one 2DoF posture or one row per arm waypoint")
    result = np.concatenate((waist_values, arm_values), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("trajectory must contain only finite values")
    return result


def safe_waist_bounds(
    joint_limits: Mapping[str, Sequence[float]],
) -> tuple[np.ndarray, np.ndarray]:
    lower = []
    upper = []
    for name in WAIST_JOINT_NAMES:
        if name not in joint_limits or len(joint_limits[name]) != 2:
            raise ValueError(f"missing limits for {name}")
        raw_lower, raw_upper = map(float, joint_limits[name])
        safe_lower, safe_upper = np.deg2rad(_SAFE_WAIST_DEG[name])
        lower.append(max(raw_lower, safe_lower))
        upper.append(min(raw_upper, safe_upper))
    if any(lo > hi for lo, hi in zip(lower, upper, strict=True)):
        raise ValueError("waist joint limits do not overlap the safety envelope")
    return np.asarray(lower), np.asarray(upper)
