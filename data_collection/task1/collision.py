#!/usr/bin/env python3
"""Shared Task1 geometry and collision-model helpers."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import yaml

from kuavo_isaaclab_scene.robots.end_effector import (
    CENTER_FRAME_NAME,
    CENTER_TOOL_FRAMES,
    ORIGINAL_EEF_FRAMES,
    urdf_with_center_frames,
)

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
GRIPPER_COLLISION_FRAMES = {
    f"{side}_{suffix}" for side in ("l", "r") for suffix in GRIPPER_LINK_SUFFIXES
}
ROBOT_COLLISION_FRAMES = {
    "waist_yaw_link",
    "zhead_1_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_l7_link",
    "zarm_r2_link",
    "zarm_r4_link",
    "zarm_r7_link",
} | GRIPPER_COLLISION_FRAMES
DEFAULT_GRIPPER_SPHERE_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "src/kuavo_isaaclab_scene/configs/task1_s200062_gripper_collision_spheres.json"
)

SELF_COLLISION_IGNORE = {
    "waist_yaw_link": ["zhead_1_link", "zarm_l2_link", "zarm_r2_link"],
    "zarm_l2_link": ["zarm_l4_link"],
    "zarm_l4_link": ["zarm_l7_link"],
    "zarm_r2_link": ["zarm_r4_link"],
    "zarm_r4_link": ["zarm_r7_link"],
}
for _side in ("l", "r"):
    _frames = [f"{_side}_{suffix}" for suffix in GRIPPER_LINK_SUFFIXES]
    SELF_COLLISION_IGNORE[f"zarm_{_side}7_link"] = list(_frames)
    for _index, _frame in enumerate(_frames[:-1]):
        SELF_COLLISION_IGNORE[_frame] = _frames[_index + 1 :]


def pose_matrix(pose) -> np.ndarray:
    pose = np.asarray(pose, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("pose must be finite xyz+wxyz")
    quaternion = pose[3:]
    if not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0):
        raise ValueError("pose quaternion must be normalized")
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    result = np.eye(4)
    result[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    result[:3, 3] = pose[:3]
    return result


def inverse_transform(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    result = np.eye(4)
    result[:3, :3] = transform[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ transform[:3, 3]
    return result


def normalized_axis(axis, *, name: str) -> np.ndarray:
    """Return one finite unit axis."""
    axis = np.asarray(axis, dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-12:
        raise ValueError(f"{name} must be nonzero")
    return axis / norm


def axis_alignment_error_deg(rotation_matrix, tool_axis, target_axis) -> float:
    """Measure the unsigned world-frame alignment error for a local tool axis."""
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation_matrix must be a finite 3x3 matrix")
    tool = normalized_axis(tool_axis, name="tool_axis")
    target = normalized_axis(target_axis, name="target_axis")
    world_axis = normalized_axis(rotation @ tool, name="world tool axis")
    return math.degrees(math.acos(float(np.clip(world_axis @ target, -1.0, 1.0))))


def rotation_error_deg(rotation_a, rotation_b) -> float:
    """Return the shortest angular distance between two rotation matrices."""
    a = np.asarray(rotation_a, dtype=float)
    b = np.asarray(rotation_b, dtype=float)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite([a, b]).all():
        raise ValueError("rotations must be finite 3x3 matrices")
    cosine = (float(np.trace(a.T @ b)) - 1.0) / 2.0
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def tool_down_angle_deg(rotation_matrix) -> float:
    """Measure local -Z, the gripper viewing axis, from robot-base down."""
    return axis_alignment_error_deg(
        rotation_matrix,
        (0.0, 0.0, -1.0),
        (0.0, 0.0, -1.0),
    )


def tool_down_orientation_targets(
    inward_normal,
    minimum_angle_deg: float,
    maximum_angle_deg: float,
    step_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build full TCP frames spanning the permitted downward-view interval."""
    normal = normalized_axis(inward_normal, name="inward flap normal")
    if not all(map(math.isfinite, (minimum_angle_deg, maximum_angle_deg, step_deg))):
        raise ValueError("tool-down angles must be finite")
    if not 0.0 <= minimum_angle_deg <= maximum_angle_deg <= 90.0:
        raise ValueError("tool-down angle range must satisfy 0 <= min <= max <= 90")
    if step_deg <= 0.0:
        raise ValueError("tool-down angle step must be positive")

    base_forward = np.asarray((1.0, 0.0, 0.0))
    horizontal = base_forward - float(base_forward @ normal) * normal
    horizontal = normalized_axis(horizontal, name="projected base-forward axis")
    base_down = np.asarray((0.0, 0.0, -1.0))
    down = base_down - float(base_down @ normal) * normal
    down -= float(down @ horizontal) * horizontal
    down = normalized_axis(down, name="projected base-down axis")

    count = max(
        1,
        int(math.floor((maximum_angle_deg - minimum_angle_deg) / step_deg)) + 1,
    )
    angles = minimum_angle_deg + np.arange(count, dtype=float) * step_deg
    if angles[-1] < maximum_angle_deg - 1e-9:
        angles = np.r_[angles, maximum_angle_deg]
    else:
        angles[-1] = maximum_angle_deg
    rotations = []
    for angle_deg in angles:
        angle = math.radians(float(angle_deg))
        view = math.sin(angle) * horizontal + math.cos(angle) * down
        local_z = -normalized_axis(view, name="tool viewing axis")
        local_y = normalized_axis(np.cross(local_z, normal), name="tool local +Y")
        local_z = normalized_axis(np.cross(normal, local_y), name="tool local +Z")
        rotations.append(np.column_stack((normal, local_y, local_z)))
    return np.asarray(rotations), angles


def box_region_goal_points(
    center: np.ndarray,
    size_m: np.ndarray,
    points_per_axis: int,
) -> np.ndarray:
    """Sample a base-aligned box goalset, preferring its center and near points."""
    center = np.asarray(center, dtype=float)
    size_m = np.asarray(size_m, dtype=float)
    if center.shape != (3,) or size_m.shape != (3,):
        raise ValueError("region center and size must be xyz vectors")
    if not np.isfinite(center).all() or not np.isfinite(size_m).all():
        raise ValueError("region center and size must be finite")
    if np.any(size_m < 0):
        raise ValueError("region size must be nonnegative")
    if points_per_axis not in {3, 5}:
        raise ValueError("points_per_axis must be 3 or 5")
    if not np.any(size_m):
        return center[None]
    fractions = (
        (0.0, -1.0, 1.0)
        if points_per_axis == 3
        else (0.0, -0.5, 0.5, -1.0, 1.0)
    )
    half = size_m / 2
    return np.stack(
        [center + np.asarray(offset) * half for offset in product(fractions, repeat=3)]
    )


def line_goal_points(
    center: np.ndarray,
    axis: np.ndarray,
    length_m: float,
    point_count: int,
) -> np.ndarray:
    """Sample a centered line segment, trying the midpoint and nearby points first."""
    center = np.asarray(center, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("line center must be a finite xyz vector")
    axis = normalized_axis(axis, name="line axis")
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("line length must be finite and positive")
    if point_count < 3 or point_count % 2 == 0:
        raise ValueError("line point count must be an odd integer of at least three")
    offsets = np.linspace(-length_m / 2, length_m / 2, point_count)
    offsets = offsets[np.argsort(np.abs(offsets), kind="stable")]
    return center[None] + offsets[:, None] * axis[None]


def target_flap_line_geometry(snapshot: dict, runtime: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return left/right upper-edge axes and their full collider lengths in base frame."""
    root_transform_w = pose_matrix(runtime["root_pose_w"])
    world_to_base_rotation = root_transform_w[:3, :3].T
    axes = []
    lengths = []
    for flap_name in ("flap_right", "flap_left"):
        matches = [
            collider
            for collider in snapshot["colliders"]
            if not collider.get("robot", False)
            and "/MediumBox_0/" in collider.get("path", "")
            and collider.get("path", "").endswith(f"/{flap_name}")
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected one MediumBox_0 {flap_name} collider, found {len(matches)}"
            )
        collider = matches[0]
        dimensions = np.asarray(collider["dims"], dtype=float)
        if dimensions.shape != (3,) or not np.isfinite(dimensions).all():
            raise ValueError(f"{flap_name} collider dimensions must be finite xyz")
        long_axis_index = int(np.argmax(dimensions))
        rotation_w = pose_matrix(collider["pose_w"])[:3, :3]
        axes.append(
            normalized_axis(
                world_to_base_rotation @ rotation_w[:, long_axis_index],
                name=f"{flap_name} long axis",
            )
        )
        lengths.append(float(dimensions[long_axis_index]))
    return np.asarray(axes), np.asarray(lengths)


def editor_region_geometry(
    editor_state: dict, region: str
) -> tuple[np.ndarray, np.ndarray]:
    """Read region geometry from the pose editor's current snapshot schema."""
    center_key = f"{region}_region_center_b_m"
    if region == "transit" and center_key not in editor_state:
        center_key = "transit_center_b_m"
    centers = np.asarray(editor_state[center_key], dtype=float)
    size = np.asarray(editor_state[f"{region}_region_size_b_m"], dtype=float)
    if centers.shape != (2, 3):
        raise ValueError("editor region must contain left/right xyz centers")
    if size.shape != (3,):
        raise ValueError("editor region size must contain xyz dimensions")
    return centers, size


def cover_cuboid(pose, dimensions, cell_m: float) -> list[tuple[np.ndarray, float]]:
    dimensions = np.asarray(dimensions, dtype=float)
    if dimensions.shape != (3,) or np.any(dimensions <= 0):
        raise ValueError("cuboid dimensions must be positive")
    counts = np.maximum(1, np.ceil(dimensions / cell_m)).astype(int)
    step = dimensions / counts
    radius = float(np.linalg.norm(step) / 2)
    transform = pose_matrix(pose)
    return [
        (
            (transform @ np.r_[(np.asarray(index) + 0.5) * step - dimensions / 2, 1])[:3],
            radius,
        )
        for index in product(*(range(count) for count in counts))
    ]


def robot_spheres(
    snapshot,
    runtime,
    cell_m: float,
    radius_inflation_m: float,
    gripper_mesh_spheres: dict[str, list[dict]] | None = None,
) -> dict:
    body_poses = {
        name: pose_matrix(pose)
        for name, pose in zip(runtime["body_names"], runtime["body_poses_w"], strict=True)
    }
    spheres: dict[str, list[dict]] = {}
    mesh_frames = set(gripper_mesh_spheres or {})
    for collider in snapshot["colliders"]:
        if not collider["robot"]:
            continue
        owner = (collider["owner"] or "").rsplit("/", 1)[-1]
        if owner not in ROBOT_COLLISION_FRAMES or owner in mesh_frames:
            continue
        owner_pose_w = collider.get("owner_pose_w")
        local_from_world = inverse_transform(
            pose_matrix(owner_pose_w) if owner_pose_w is not None else body_poses[owner]
        )
        entries = spheres.setdefault(owner, [])
        for center_w, radius in cover_cuboid(collider["pose_w"], collider["dims"], cell_m):
            entries.append(
                {
                    "center": (local_from_world @ np.r_[center_w, 1])[:3].tolist(),
                    "radius": radius + radius_inflation_m,
                }
            )
    for owner, entries in (gripper_mesh_spheres or {}).items():
        if owner not in ROBOT_COLLISION_FRAMES:
            raise ValueError(f"unexpected mesh collision frame: {owner}")
        spheres[owner] = [
            {
                "center": list(entry["center"]),
                "radius": float(entry["radius"]) + radius_inflation_m,
            }
            for entry in entries
        ]
    missing = ROBOT_COLLISION_FRAMES - spheres.keys()
    if missing:
        raise ValueError(f"missing live robot collision frames: {sorted(missing)}")
    return spheres


def load_gripper_mesh_spheres(max_overshoot_m: float) -> dict[str, list[dict]]:
    """Load one checked-in cuMotion-generated hand-sphere preset."""
    payload = json.loads(DEFAULT_GRIPPER_SPHERE_CONFIG.read_text())
    preset = payload.get("presets", {}).get(f"{max_overshoot_m:.3f}")
    if (
        payload.get("schema_version") != 1
        or payload.get("robot_model") != "s200062"
        or payload.get("sphere_coordinate_frame") != "urdf_link_frame"
        or not isinstance(preset, dict)
        or not math.isclose(
            float(preset.get("max_overshoot_m", float("nan"))),
            max_overshoot_m,
            abs_tol=1e-12,
            rel_tol=0.0,
        )
        or set(preset.get("frames", {})) != GRIPPER_COLLISION_FRAMES
    ):
        raise ValueError(
            f"invalid gripper sphere preset {max_overshoot_m}: "
            f"{DEFAULT_GRIPPER_SPHERE_CONFIG}"
        )
    return preset["frames"]


def collision_world_config(
    snapshot: dict, world_config: dict, *, allow_target_flap_contact: bool
) -> tuple[dict, list[str]]:
    """Optionally omit the two grasped flaps while keeping the rest of the box."""
    nonrobot = [item for item in snapshot["colliders"] if not item["robot"]]
    cuboids = world_config["cuboid"]
    if len(cuboids) != len(nonrobot):
        raise ValueError("world obstacle count does not match the collider snapshot")
    kept = {}
    allowed = []
    for index, collider in enumerate(snapshot["colliders"]):
        if collider["robot"]:
            continue
        key = f"obstacle_{index}"
        path = collider["path"]
        is_target_flap = (
            "/MediumBox_0/" in path
            and path.endswith(("/flap_right", "/flap_left"))
        )
        if allow_target_flap_contact and is_target_flap:
            allowed.append(path)
        else:
            kept[key] = cuboids[key]
    return {"cuboid": kept}, allowed


def xrdf(
    *,
    side: str,
    defaults: dict[str, float],
    world_spheres: dict,
    self_spheres: dict,
) -> str:
    letter = {"left": "l", "right": "r"}[side]
    names = [f"zarm_{letter}{index}_joint" for index in range(1, 8)]
    data = {
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": defaults,
        "cspace": {
            "joint_names": names,
            "acceleration_limits": [10.0] * 7,
            "jerk_limits": [100.0] * 7,
        },
        "tool_frames": [CENTER_TOOL_FRAMES[side]],
        "world_collision": {"geometry": "kuavo_world_spheres"},
        "self_collision": {
            "geometry": "kuavo_self_spheres",
            "ignore": SELF_COLLISION_IGNORE,
        },
        "geometry": {
            "kuavo_world_spheres": {"spheres": world_spheres},
            "kuavo_self_spheres": {"spheres": self_spheres},
        },
    }
    return yaml.safe_dump(data, sort_keys=False)


def runtime_joint_defaults(runtime: dict) -> dict[str, float]:
    """Keep the captured full-body posture fixed outside the planned arm c-space."""
    arm_chain_joints = {"knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"}
    defaults = {
        name: float(value)
        for name, value in zip(
            runtime["joint_names"], runtime["joint_positions"], strict=True
        )
        if name in arm_chain_joints or name.startswith(("zarm_", "l_", "r_"))
    }
    defaults.update({
        item["name"]: float(item["value"])
        for item in runtime["pose_editor_state"]["joints"]
        if item["name"] in arm_chain_joints
        or item["name"].startswith(("zarm_", "l_", "r_"))
    })
    return defaults
