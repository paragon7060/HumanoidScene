#!/usr/bin/env python3
"""Plan collision-aware sequential Task1 bimanual targets with cuMotion."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import yaml


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
    Path(__file__).resolve().parents[1]
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
        "tool_frames": [f"zarm_{letter}7_end_effector"],
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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot-dir", type=Path, required=True)
    result.add_argument("--urdf", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--sphere-cell-m", type=float, default=0.06)
    result.add_argument(
        "--gripper-max-overshoot-m",
        type=float,
        choices=(0.002, 0.005, 0.010, 0.020),
        default=0.002,
    )
    result.add_argument("--collision-margin-m", type=float, default=0.005)
    result.add_argument("--self-pair-margin-m", type=float, default=None)
    result.add_argument("--validation-samples", type=int, default=101)
    result.add_argument("--target", choices=("pregrasp", "grasp"), default="pregrasp")
    result.add_argument(
        "--editor-region",
        choices=("pregrasp", "transit"),
        default=None,
        help="Use the editable box region stored in the planning snapshot.",
    )
    result.add_argument(
        "--target-offset-b-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    result.add_argument(
        "--target-region-size-m",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=(0.0, 0.0, 0.0),
    )
    result.add_argument(
        "--target-region-grid-points-per-axis",
        type=int,
        choices=(3, 5),
        default=5,
    )
    result.add_argument(
        "--target-flap-line-length-m",
        type=float,
        default=None,
        help="Allow the grasp target to move only along each target flap's long edge.",
    )
    result.add_argument(
        "--target-flap-line-point-count",
        type=int,
        default=21,
        help="Odd number of goal samples along the target flap line.",
    )
    result.add_argument("--target-tolerance-m", type=float, default=0.005)
    result.add_argument("--closing-axis-tolerance-deg", type=float, default=None)
    result.add_argument("--tool-down-angle-min-deg", type=float, default=None)
    result.add_argument("--tool-down-angle-max-deg", type=float, default=None)
    result.add_argument("--tool-down-angle-step-deg", type=float, default=5.0)
    result.add_argument("--terminal-orientation-tolerance-deg", type=float, default=2.0)
    result.add_argument(
        "--terminal-seed-only",
        action="store_true",
        help=(
            "Search IK endpoints without optimizer collision kernels, reject endpoints with the "
            "exact inspector, and require a downstream bimanual path planner."
        ),
    )
    result.add_argument("--independent-arm-seeds", action="store_true")
    result.add_argument("--allow-target-flap-contact", action="store_true")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not math.isfinite(args.sphere_cell_m) or args.sphere_cell_m <= 0:
        raise ValueError("--sphere-cell-m must be finite and positive")
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0:
        raise ValueError("--collision-margin-m must be finite and nonnegative")
    self_pair_margin_m = (
        args.collision_margin_m
        if args.self_pair_margin_m is None
        else args.self_pair_margin_m
    )
    if not math.isfinite(self_pair_margin_m) or self_pair_margin_m < 0:
        raise ValueError("--self-pair-margin-m must be finite and nonnegative")
    if args.validation_samples < 2:
        raise ValueError("--validation-samples must be at least two")
    if not math.isfinite(args.target_tolerance_m) or args.target_tolerance_m <= 0:
        raise ValueError("--target-tolerance-m must be finite and positive")
    if args.closing_axis_tolerance_deg is not None and (
        not math.isfinite(args.closing_axis_tolerance_deg)
        or not 0 < args.closing_axis_tolerance_deg < 180
    ):
        raise ValueError("--closing-axis-tolerance-deg must be between zero and 180")
    down_range = (args.tool_down_angle_min_deg, args.tool_down_angle_max_deg)
    if (down_range[0] is None) != (down_range[1] is None):
        raise ValueError("tool-down angle min and max must be specified together")
    if down_range[0] is not None:
        tool_down_orientation_targets(
            (0.0, 1.0, 0.0),
            down_range[0],
            down_range[1],
            args.tool_down_angle_step_deg,
        )
    if (
        not math.isfinite(args.terminal_orientation_tolerance_deg)
        or not 0.0 <= args.terminal_orientation_tolerance_deg < 180.0
    ):
        raise ValueError("--terminal-orientation-tolerance-deg must be in [0, 180)")
    target_offset_b_m = np.asarray(args.target_offset_b_m, dtype=float)
    target_region_size_m = np.asarray(args.target_region_size_m, dtype=float)
    if not np.isfinite(target_offset_b_m).all():
        raise ValueError("--target-offset-b-m must be finite")
    if not np.isfinite(target_region_size_m).all() or np.any(target_region_size_m < 0):
        raise ValueError("--target-region-size-m must be finite and nonnegative")
    if args.target_flap_line_length_m is not None:
        if (
            not math.isfinite(args.target_flap_line_length_m)
            or args.target_flap_line_length_m <= 0
        ):
            raise ValueError("--target-flap-line-length-m must be finite and positive")
        if (
            args.target_flap_line_point_count < 3
            or args.target_flap_line_point_count % 2 == 0
        ):
            raise ValueError(
                "--target-flap-line-point-count must be an odd integer of at least three"
            )
        if args.target != "grasp":
            raise ValueError("--target-flap-line-length-m requires --target grasp")
        if args.editor_region is not None or np.any(target_region_size_m):
            raise ValueError(
                "--target-flap-line-length-m cannot be combined with a box target region"
            )
    if args.editor_region is not None and (
        np.any(target_offset_b_m) or np.any(target_region_size_m)
    ):
        raise ValueError(
            "--editor-region cannot be combined with target offset/region size overrides"
        )
    output = args.output_dir or Path("/home/seonho/outputs/HumanoidScene") / (
        f"cumotion_collision_{args.target}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    import cumotion

    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    snapshot = json.loads((snapshot_dir / "collision_snapshot.json").read_text())
    runtime = json.loads((snapshot_dir / "runtime.json").read_text())
    world_config = json.loads((snapshot_dir / "world.json").read_text())
    world_config, allowed_contact_colliders = collision_world_config(
        snapshot,
        world_config,
        allow_target_flap_contact=args.allow_target_flap_contact,
    )
    urdf_text = args.urdf.expanduser().resolve().read_text()
    gripper_mesh_spheres = load_gripper_mesh_spheres(
        args.gripper_max_overshoot_m
    )

    world_spheres = robot_spheres(
        snapshot,
        runtime,
        args.sphere_cell_m,
        args.collision_margin_m,
        gripper_mesh_spheres,
    )
    self_spheres = robot_spheres(
        snapshot,
        runtime,
        args.sphere_cell_m,
        self_pair_margin_m / 2,
        gripper_mesh_spheres,
    )
    world = cumotion.create_world()
    for obstacle_data in world_config["cuboid"].values():
        obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        obstacle.set_attribute(
            cumotion.Obstacle.Attribute.SIDE_LENGTHS,
            np.asarray(obstacle_data["dims"], dtype=float),
        )
        world.add_obstacle(obstacle, cumotion.Pose3(pose_matrix(obstacle_data["pose"])))
    world_view = world.add_world_view()

    defaults = runtime_joint_defaults(runtime)
    editor_state = runtime["pose_editor_state"]
    targets = editor_state[f"{args.target}_position_b"]
    editor_region_centers = None
    if args.editor_region is not None:
        editor_region_centers, target_region_size_m = editor_region_geometry(
            editor_state, args.editor_region
        )
    inward_normals = editor_state["inward_flap_normal_b"]
    flap_line_axes = None
    flap_line_full_lengths = None
    if args.target_flap_line_length_m is not None:
        flap_line_axes, flap_line_full_lengths = target_flap_line_geometry(
            snapshot, runtime
        )
        if np.any(args.target_flap_line_length_m > flap_line_full_lengths + 1e-9):
            raise ValueError(
                "--target-flap-line-length-m exceeds a target flap collider's long edge: "
                f"requested={args.target_flap_line_length_m}, "
                f"available={flap_line_full_lengths.tolist()}"
            )
    if args.tool_down_angle_min_deg is not None:
        orientation_report = {
            "type": "terminal_full_orientation_candidate_set",
            "closing_axis": {
                "tool_frame_axis": [1.0, 0.0, 0.0],
                "target": "inward_flap_normal_b",
            },
            "view_axis": {
                "tool_frame_axis": [0.0, 0.0, -1.0],
                "angle_reference_b": [0.0, 0.0, -1.0],
            },
            "tool_down_angle_range_deg": list(down_range),
            "tool_down_angle_step_deg": args.tool_down_angle_step_deg,
            "terminal_orientation_deviation_limit_deg": (
                args.terminal_orientation_tolerance_deg
            ),
            "terminal_closing_axis_deviation_limit_deg": (
                args.closing_axis_tolerance_deg
            ),
        }
    else:
        orientation_report = (
            {"type": "none"}
            if args.closing_axis_tolerance_deg is None
            else {
                "type": "terminal_axis",
                "tool_frame_axis": [1.0, 0.0, 0.0],
                "terminal_axis_deviation_limit_deg": args.closing_axis_tolerance_deg,
            }
        )
    report = {
        "planner": "NVIDIA cuMotion 1.1.0 TrajectoryOptimizer",
        "strategy": f"sequential_bimanual_{args.target}",
        "target": args.target,
        "snapshot_dir": str(snapshot_dir),
        "initial_pose_source": runtime.get("initial_state", "snapshot runtime joint state"),
        "orientation_constraint": orientation_report,
        "target_region": {
            "reference": args.editor_region or args.target,
            "source": "pose_editor_state" if args.editor_region else "command_line",
            "center_offset_b_m": target_offset_b_m.tolist(),
            "size_b_m": target_region_size_m.tolist(),
            "goalset_grid_points_per_axis": args.target_region_grid_points_per_axis,
        },
        "target_flap_line": {
            "enabled": flap_line_axes is not None,
            "length_m": args.target_flap_line_length_m,
            "point_count": args.target_flap_line_point_count,
            "axis_b": None if flap_line_axes is None else flap_line_axes.tolist(),
            "full_collider_length_m": (
                None
                if flap_line_full_lengths is None
                else flap_line_full_lengths.tolist()
            ),
        },
        "collision_model": {
            "world_obstacles": len(world_config["cuboid"]),
            "robot_world_spheres": sum(map(len, world_spheres.values())),
            "robot_self_spheres": sum(map(len, self_spheres.values())),
            "sphere_cover_cell_m": args.sphere_cell_m,
            "gripper_mesh_max_overshoot_m": args.gripper_max_overshoot_m,
            "gripper_mesh_sphere_count": sum(
                map(len, gripper_mesh_spheres.values())
            ),
            "world_margin_m": args.collision_margin_m,
            "self_pair_margin_m": self_pair_margin_m,
            "excluded_stage_colliders": len(snapshot["excluded_enabled_colliders"]),
            "allow_target_flap_contact": args.allow_target_flap_contact,
            "allowed_contact_colliders": allowed_contact_colliders,
        },
        "arms": {},
    }

    left_terminal = None
    for side, letter, target_index in (("left", "l", 0), ("right", "r", 1)):
        arm_defaults = dict(defaults)
        if left_terminal is not None and not args.independent_arm_seeds:
            arm_defaults.update(
                {f"zarm_l{index}_joint": float(value) for index, value in enumerate(left_terminal, 1)}
            )
        names = [f"zarm_{letter}{index}_joint" for index in range(1, 8)]
        q_initial = np.asarray([arm_defaults[name] for name in names], dtype=float)
        tool = f"zarm_{letter}7_end_effector"
        xrdf_text = xrdf(
            side=side,
            defaults=arm_defaults,
            world_spheres=world_spheres,
            self_spheres=self_spheres,
        )
        (output / f"{side}.xrdf").write_text(xrdf_text)
        robot = cumotion.load_robot_from_memory(xrdf_text, urdf_text)
        inspector = cumotion.create_robot_world_inspector(robot, world_view)
        initial_world_collision = inspector.in_collision_with_obstacle(q_initial)
        initial_self_collision = inspector.in_self_collision(q_initial)
        if initial_world_collision or initial_self_collision:
            raise RuntimeError(
                f"{side} initial collision: world={initial_world_collision}, "
                f"self={initial_self_collision}, pairs={inspector.frames_in_self_collision(q_initial)}"
            )
        optimizer_config = cumotion.create_default_trajectory_optimizer_config(
            robot, tool, world_view
        )
        endpoint_filter_mode = (
            args.tool_down_angle_min_deg is not None or args.terminal_seed_only
        )
        if endpoint_filter_mode:
            if not optimizer_config.set_param("enable_self_collision", False):
                raise RuntimeError("failed to disable self collision for terminal IK search")
            if not optimizer_config.set_param("enable_world_collision", False):
                raise RuntimeError("failed to disable world collision for terminal IK search")
        optimizer = cumotion.create_trajectory_optimizer(optimizer_config)
        target_center = (
            editor_region_centers[target_index]
            if editor_region_centers is not None
            else np.asarray(targets[target_index], dtype=float) + target_offset_b_m
        )
        if flap_line_axes is None:
            target_goal_points = box_region_goal_points(
                target_center,
                target_region_size_m,
                args.target_region_grid_points_per_axis,
            )
        else:
            target_goal_points = line_goal_points(
                target_center,
                flap_line_axes[target_index],
                args.target_flap_line_length_m,
                args.target_flap_line_point_count,
            )
        inward_normal = normalized_axis(
            inward_normals[target_index], name=f"{side} inward flap normal"
        )
        orientation_target_matrices = None
        orientation_target_angles = None
        orientation_candidate_attempts = []
        selected_terminal_collision_free = None
        if args.tool_down_angle_min_deg is not None:
            orientation_target_matrices, orientation_target_angles = (
                tool_down_orientation_targets(
                    inward_normal,
                    args.tool_down_angle_min_deg,
                    args.tool_down_angle_max_deg,
                    args.tool_down_angle_step_deg,
                )
            )
            position_count = len(target_goal_points)
            orientation_count = len(orientation_target_matrices)
            target_goal_points = np.repeat(
                target_goal_points, orientation_count, axis=0
            )
            orientation_target_matrices = np.tile(
                orientation_target_matrices, (position_count, 1, 1)
            )
            orientation_target_angles = np.tile(
                orientation_target_angles, position_count
            )
        if orientation_target_matrices is not None:
            result = None
            selected_target_index = 0
            selected_terminal_collision_free = False
            for candidate_index, (candidate_position, candidate_rotation) in enumerate(
                zip(target_goal_points, orientation_target_matrices, strict=True)
            ):
                orientation_constraint = (
                    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_target(
                        cumotion.Rotation3.from_matrix(candidate_rotation),
                        math.radians(args.terminal_orientation_tolerance_deg),
                    )
                )
                target = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
                    cumotion.TrajectoryOptimizer.TranslationConstraint.target(
                        candidate_position
                    ),
                    orientation_constraint,
                )
                candidate_result = optimizer.plan_to_task_space_target(q_initial, target)
                result = candidate_result
                selected_target_index = candidate_index
                candidate_status = str(candidate_result.status()).split(".")[-1]
                attempt = {
                    "index": candidate_index,
                    "target_tool_down_angle_deg": float(
                        orientation_target_angles[candidate_index]
                    ),
                    "status": candidate_status,
                }
                orientation_candidate_attempts.append(attempt)
                if candidate_status != "SUCCESS":
                    continue
                candidate_trajectory = candidate_result.trajectory()
                candidate_terminal = np.asarray(
                    candidate_trajectory.eval(float(candidate_trajectory.domain().upper)),
                    dtype=float,
                )
                attempt["terminal_world_collision"] = bool(
                    inspector.in_collision_with_obstacle(candidate_terminal)
                )
                attempt["terminal_self_collision"] = bool(
                    inspector.in_self_collision(candidate_terminal)
                )
                attempt["terminal_min_world_distance_m"] = float(
                    inspector.min_distance_to_obstacle(candidate_terminal)
                )
                if not (
                    attempt["terminal_world_collision"]
                    or attempt["terminal_self_collision"]
                ):
                    selected_terminal_collision_free = True
                    break
        elif len(target_goal_points) == 1:
            orientation_constraint = (
                cumotion.TrajectoryOptimizer.OrientationConstraint.none()
            )
            if args.closing_axis_tolerance_deg is not None:
                orientation_constraint = (
                    cumotion.TrajectoryOptimizer.OrientationConstraint.terminal_axis(
                        np.asarray((1.0, 0.0, 0.0)),
                        inward_normal,
                        math.radians(args.closing_axis_tolerance_deg),
                    )
                )
            target = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
                cumotion.TrajectoryOptimizer.TranslationConstraint.target(target_center),
                orientation_constraint,
            )
            result = optimizer.plan_to_task_space_target(q_initial, target)
            selected_target_index = 0
        else:
            orientation_constraint = (
                cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.none()
            )
            if args.closing_axis_tolerance_deg is not None:
                count = len(target_goal_points)
                orientation_constraint = (
                    cumotion.TrajectoryOptimizer.OrientationConstraintGoalset.terminal_axis(
                        [np.asarray((1.0, 0.0, 0.0))] * count,
                        [inward_normal] * count,
                        math.radians(args.closing_axis_tolerance_deg),
                    )
                )
            target = cumotion.TrajectoryOptimizer.TaskSpaceTargetGoalset(
                cumotion.TrajectoryOptimizer.TranslationConstraintGoalset.target(
                    target_goal_points
                ),
                orientation_constraint,
            )
            result = optimizer.plan_to_task_space_goalset(q_initial, target)
            selected_target_index = result.target_index()
        status = str(result.status()).split(".")[-1]
        if orientation_target_matrices is not None and not selected_terminal_collision_free:
            status = "COLLISION_FREE_ENDPOINT_FAILURE"
        if (
            endpoint_filter_mode
            and orientation_target_matrices is None
            and status == "SUCCESS"
        ):
            candidate_trajectory = result.trajectory()
            candidate_terminal = np.asarray(
                candidate_trajectory.eval(float(candidate_trajectory.domain().upper)),
                dtype=float,
            )
            selected_terminal_collision_free = not (
                inspector.in_collision_with_obstacle(candidate_terminal)
                or inspector.in_self_collision(candidate_terminal)
            )
            if not selected_terminal_collision_free:
                status = "COLLISION_FREE_ENDPOINT_FAILURE"
        target_position = target_goal_points[selected_target_index]
        arm_report = {
            "status": status,
            "joint_names": names,
            "target_position_b_m": target_position.tolist(),
            "target_region_center_b_m": target_center.tolist(),
            "target_region_size_b_m": target_region_size_m.tolist(),
            "target_flap_line_axis_b": (
                None
                if flap_line_axes is None
                else flap_line_axes[target_index].tolist()
            ),
            "target_flap_line_length_m": args.target_flap_line_length_m,
            "selected_goalset_index": int(selected_target_index),
            "target_inward_flap_normal_b": inward_normal.tolist(),
            "selected_tool_down_target_angle_deg": (
                None
                if orientation_target_angles is None
                else float(orientation_target_angles[selected_target_index])
            ),
            "orientation_candidate_attempts": orientation_candidate_attempts,
            "terminal_ik_collision_mode": (
                "collision_unaware_search_exact_endpoint_filter"
                if endpoint_filter_mode
                else "collision_aware_trajectory_optimizer"
            ),
            "fixed_other_arm": (
                "main_initial"
                if side == "left" or args.independent_arm_seeds
                else "left_terminal"
            ),
        }
        report["arms"][side] = arm_report
        if status != "SUCCESS":
            break

        trajectory = result.trajectory()
        domain = trajectory.domain()
        sample_times = np.linspace(
            float(domain.lower), float(domain.upper), args.validation_samples
        )
        samples = np.stack(
            [np.asarray(trajectory.eval(float(sample_time)), dtype=float) for sample_time in sample_times]
        )
        terminal = samples[-1]
        terminal_position = np.asarray(robot.kinematics().position(terminal, tool), dtype=float)
        terminal_rotation = np.asarray(
            robot.kinematics().orientation(terminal, tool).matrix(), dtype=float
        )
        terminal_closing_axis = terminal_rotation @ np.asarray((1.0, 0.0, 0.0))
        closing_axis_error_deg = axis_alignment_error_deg(
            terminal_rotation, (1.0, 0.0, 0.0), inward_normal
        )
        terminal_tool_down = tool_down_angle_deg(terminal_rotation)
        terminal_orientation_error = (
            None
            if orientation_target_matrices is None
            else rotation_error_deg(
                terminal_rotation,
                orientation_target_matrices[selected_target_index],
            )
        )
        world_collisions = [inspector.in_collision_with_obstacle(q) for q in samples]
        self_collisions = [inspector.in_self_collision(q) for q in samples]
        distances = [inspector.min_distance_to_obstacle(q) for q in samples]
        arm_report.update(
            {
                "duration_s": float(domain.span()),
                "terminal_q_rad": terminal.tolist(),
                "terminal_tcp_position_b_m": terminal_position.tolist(),
                "terminal_error_m": float(np.linalg.norm(terminal_position - target_position)),
                "terminal_tcp_local_x_b": terminal_closing_axis.tolist(),
                "terminal_closing_axis_error_deg": closing_axis_error_deg,
                "terminal_tool_down_angle_deg": terminal_tool_down,
                "terminal_orientation_error_deg": terminal_orientation_error,
                "sample_times_s": sample_times.tolist(),
                "sample_q_rad": samples.tolist(),
                "sampled_world_collision": any(world_collisions),
                "sampled_self_collision": any(self_collisions),
                "terminal_world_collision": bool(world_collisions[-1]),
                "terminal_self_collision": bool(self_collisions[-1]),
                "sampled_min_world_distance_m": float(min(distances)),
                "terminal_self_collision_pairs": inspector.frames_in_self_collision(terminal),
            }
        )
        if side == "left":
            left_terminal = terminal.copy()

    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "arms": {
                    side: {
                        key: arm.get(key)
                        for key in (
                            "status",
                            "duration_s",
                            "terminal_error_m",
                            "terminal_closing_axis_error_deg",
                            "terminal_tool_down_angle_deg",
                            "terminal_orientation_error_deg",
                            "sampled_world_collision",
                            "sampled_self_collision",
                            "sampled_min_world_distance_m",
                        )
                    }
                    for side, arm in report["arms"].items()
                },
            },
            indent=2,
        )
    )
    success = len(report["arms"]) == 2 and all(
        arm["status"] == "SUCCESS"
        and arm["terminal_error_m"] <= args.target_tolerance_m
        and (
            args.closing_axis_tolerance_deg is None
            or arm["terminal_closing_axis_error_deg"]
            <= args.closing_axis_tolerance_deg + 1e-3
        )
        and (
            args.tool_down_angle_min_deg is None
            or args.tool_down_angle_min_deg
            - args.terminal_orientation_tolerance_deg
            - 1e-3
            <= arm["terminal_tool_down_angle_deg"]
            <= args.tool_down_angle_max_deg
            + args.terminal_orientation_tolerance_deg
            + 1e-3
        )
        and (
            endpoint_filter_mode
            or not arm["sampled_world_collision"]
        )
        and (
            endpoint_filter_mode
            or not arm["sampled_self_collision"]
        )
        and not arm["terminal_world_collision"]
        and not arm["terminal_self_collision"]
        for arm in report["arms"].values()
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
