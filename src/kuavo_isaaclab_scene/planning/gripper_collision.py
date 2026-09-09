"""Validated mesh-fitted collision spheres for the S200062 two-finger hands."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..core.paths import PACKAGE_CONFIG_DIR, require_resource


GRIPPER_LINK_SUFFIXES = (
    "twofinger_base",
    "f_bar_1",
    "f_bar_2",
    "f_bar_3",
    "f_finger",
    "f_bar_4",
    "b_bar_1",
    "b_bar_2",
    "b_bar_3",
    "b_finger",
    "b_bar_4",
    "d405_camera_connect",
    "d405_camera_base",
    "d405_camera",
)
GRIPPER_COLLISION_FRAMES = tuple(
    f"{side}_{suffix}" for side in ("l", "r") for suffix in GRIPPER_LINK_SUFFIXES
)
SUPPORTED_MAX_OVERSHOOT_M = (0.002, 0.005, 0.010, 0.020)
SPHERE_COORDINATE_FRAME = "urdf_link_frame"
DEFAULT_GRIPPER_COLLISION_SPHERES = (
    PACKAGE_CONFIG_DIR / "task1_s200062_gripper_collision_spheres.json"
)


def _load_payload(path: Path) -> dict:
    payload = json.loads(
        require_resource(path, "S200062 gripper collision spheres").read_text()
    )
    if (
        payload.get("schema_version") != 1
        or payload.get("robot_model") != "s200062"
        or payload.get("sphere_coordinate_frame") != SPHERE_COORDINATE_FRAME
    ):
        raise ValueError(f"unsupported gripper collision-sphere config: {path}")
    return payload


def load_gripper_mesh_bounds(
    path: Path = DEFAULT_GRIPPER_COLLISION_SPHERES,
) -> dict[str, tuple[list[float], list[float]]]:
    """Load the source-mesh bounds recorded when the spheres were generated."""
    raw = _load_payload(path).get("mesh_bounds")
    if not isinstance(raw, dict) or set(raw) != set(GRIPPER_COLLISION_FRAMES):
        raise ValueError(f"gripper mesh bounds do not match S200062: {path}")
    result = {}
    for frame_name in GRIPPER_COLLISION_FRAMES:
        record = raw[frame_name]
        low = record.get("min") if isinstance(record, dict) else None
        high = record.get("max") if isinstance(record, dict) else None
        if not all(
            isinstance(values, list)
            and len(values) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)
            for values in (low, high)
        ) or any(float(high[index]) <= float(low[index]) for index in range(3)):
            raise ValueError(f"invalid mesh bounds for {frame_name}: {record}")
        result[frame_name] = (
            [float(value) for value in low],
            [float(value) for value in high],
        )
    return result


def load_gripper_collision_spheres(
    max_overshoot_m: float,
    path: Path = DEFAULT_GRIPPER_COLLISION_SPHERES,
) -> dict[str, list[dict]]:
    """Load one generated preset and reject incomplete or malformed geometry."""
    if not math.isfinite(max_overshoot_m) or not any(
        math.isclose(max_overshoot_m, value, abs_tol=1e-12, rel_tol=0.0)
        for value in SUPPORTED_MAX_OVERSHOOT_M
    ):
        raise ValueError(
            f"max overshoot must be one of {SUPPORTED_MAX_OVERSHOOT_M}: "
            f"{max_overshoot_m}"
        )
    payload = _load_payload(path)
    preset = payload.get("presets", {}).get(f"{max_overshoot_m:.3f}")
    if not isinstance(preset, dict) or not math.isclose(
        float(preset.get("max_overshoot_m", float("nan"))),
        max_overshoot_m,
        abs_tol=1e-12,
        rel_tol=0.0,
    ):
        raise ValueError(f"missing max-overshoot preset {max_overshoot_m:.3f}: {path}")
    frames = preset.get("frames")
    if not isinstance(frames, dict) or set(frames) != set(GRIPPER_COLLISION_FRAMES):
        raise ValueError(f"gripper collision frames do not match S200062: {path}")
    validated = {}
    for frame_name in GRIPPER_COLLISION_FRAMES:
        spheres = frames[frame_name]
        if not isinstance(spheres, list) or not spheres:
            raise ValueError(f"empty collision-sphere frame {frame_name}: {path}")
        frame_spheres = []
        for sphere in spheres:
            center = sphere.get("center") if isinstance(sphere, dict) else None
            radius = sphere.get("radius") if isinstance(sphere, dict) else None
            if (
                not isinstance(center, list)
                or len(center) != 3
                or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in center)
                or not isinstance(radius, (int, float))
                or not math.isfinite(radius)
                or radius <= 0
            ):
                raise ValueError(f"invalid sphere for {frame_name}: {sphere}")
            frame_spheres.append(
                {"center": [float(value) for value in center], "radius": float(radius)}
            )
        validated[frame_name] = frame_spheres
    if sum(map(len, validated.values())) != int(preset.get("sphere_count", -1)):
        raise ValueError(f"sphere count does not match preset metadata: {path}")
    return validated
