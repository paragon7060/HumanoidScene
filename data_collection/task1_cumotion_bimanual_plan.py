#!/usr/bin/env python3
"""Plan one simultaneous collision-aware 14-DoF Task1 bimanual path."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

import numpy as np
import yaml

from data_collection.task1_cumotion_collision_plan import (
    SELF_COLLISION_IGNORE,
    axis_alignment_error_deg,
    collision_world_config,
    load_gripper_mesh_spheres,
    normalized_axis,
    pose_matrix,
    robot_spheres,
)


ARM_JOINT_NAMES = [
    f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)
]
TOOL_FRAMES = ["zarm_l7_end_effector", "zarm_r7_end_effector"]


def bimanual_xrdf(
    defaults: dict[str, float], world_spheres: dict, self_spheres: dict
) -> str:
    """Build a single 14-DoF robot description with both TCP frames."""
    data = {
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": defaults,
        "cspace": {
            "joint_names": ARM_JOINT_NAMES,
            "acceleration_limits": [10.0] * len(ARM_JOINT_NAMES),
            "jerk_limits": [100.0] * len(ARM_JOINT_NAMES),
        },
        "tool_frames": TOOL_FRAMES,
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


def planner_yaml(
    joint_count: int,
    *,
    seed: int = 123456,
    step_size: float = 0.05,
    shoulder_sweep_weight: float = 8.0,
    task_space_limits: list[list[float]] | None = None,
) -> str:
    """Return deterministic cuMotion graph-planner parameters for the workcell."""
    distance_weights = [1.0] * joint_count
    for index in (0, 7):
        if index < joint_count:
            distance_weights[index] = shoulder_sweep_weight
    data = {
        "seed": seed,
        "step_size": step_size,
        "max_iterations": 100000,
        "max_sampling": 30000,
        "distance_metric_weights": distance_weights,
        "task_space_limits": task_space_limits
        or [[-1.5, 1.5], [-1.5, 1.5], [-0.5, 2.5]],
        "cuda_tree_params": {
            "max_buffer_size": 30,
            "num_nodes_cpu_gpu_crossover": 3000,
        },
        "cspace_planning_params": {"exploration_fraction": 0.5},
        "task_space_planning_params": {
            "translation_target_zone_tolerance": 0.05,
            "orientation_target_zone_tolerance": 0.09,
            "translation_target_final_tolerance": 0.0001,
            "orientation_target_final_tolerance": 0.005,
            "translation_gradient_weight": 1.0,
            "orientation_gradient_weight": 0.125,
            "nn_translation_distance_weight": 1.0,
            "nn_orientation_distance_weight": 0.125,
            "task_space_exploitation_fraction": 0.4,
            "task_space_exploration_fraction": 0.1,
            "max_extension_substeps_away_from_target": 6,
            "max_extension_substeps_near_target": 50,
            "extension_substep_target_region_scale_factor": 2.0,
            "unexploited_nodes_culling_scalar": 1.0,
            "gradient_substep_size": 0.025,
        },
    }
    return yaml.safe_dump(data, sort_keys=False)


def densify_path(path: np.ndarray, max_joint_step_rad: float) -> np.ndarray:
    """Linearly sample every graph edge for an independent collision gate."""
    if path.ndim != 2 or len(path) < 2:
        raise ValueError("path must contain at least two c-space waypoints")
    if not math.isfinite(max_joint_step_rad) or max_joint_step_rad <= 0:
        raise ValueError("max_joint_step_rad must be finite and positive")
    dense = [path[0].copy()]
    for start, end in zip(path[:-1], path[1:], strict=True):
        count = max(1, int(math.ceil(np.max(np.abs(end - start)) / max_joint_step_rad)))
        dense.extend(start + (end - start) * (index / count) for index in range(1, count + 1))
    return np.asarray(dense)


def joint_space_path_length(path: np.ndarray) -> float:
    """Return Euclidean length through the 14-DoF joint space."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("path must contain at least two finite c-space waypoints")
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())


def segment_is_collision_free(
    start: np.ndarray,
    end: np.ndarray,
    max_joint_step_rad: float,
    in_collision,
) -> bool:
    """Validate one joint-space line segment at the requested resolution."""
    dense = densify_path(np.stack((start, end)), max_joint_step_rad)
    return not any(bool(in_collision(row)) for row in dense)


def shortcut_path(
    path: np.ndarray,
    max_joint_step_rad: float,
    in_collision,
) -> np.ndarray:
    """Remove graph detours while preserving sampled collision freedom."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or len(path) < 2 or not np.isfinite(path).all():
        raise ValueError("path must contain at least two finite c-space waypoints")
    knots = [path[0]]
    start_index = 0
    while start_index < len(path) - 1:
        for end_index in range(len(path) - 1, start_index, -1):
            if segment_is_collision_free(
                path[start_index],
                path[end_index],
                max_joint_step_rad,
                in_collision,
            ):
                knots.append(path[end_index])
                start_index = end_index
                break
        else:
            raise RuntimeError("graph path contains no collision-free next segment")
    return np.asarray(knots)


def synchronized_seed_path(
    seed_report: dict,
    q_initial: np.ndarray,
    q_terminal: np.ndarray,
) -> np.ndarray:
    """Combine the two optimizer trajectories at equal normalized progress."""
    q_initial = np.asarray(q_initial, dtype=float)
    q_terminal = np.asarray(q_terminal, dtype=float)
    arm_paths = [
        np.asarray(seed_report["arms"][side]["sample_q_rad"], dtype=float)
        for side in ("left", "right")
    ]
    if (
        q_initial.shape != (14,)
        or q_terminal.shape != (14,)
        or any(path.ndim != 2 or path.shape[1] != 7 or len(path) < 2 for path in arm_paths)
        or not all(np.isfinite(path).all() for path in arm_paths)
    ):
        raise ValueError("seed plan must contain two finite sampled 7-DoF trajectories")
    progress = np.linspace(0.0, 1.0, max(map(len, arm_paths)))
    synchronized = []
    for path in arm_paths:
        source = np.linspace(0.0, 1.0, len(path))
        synchronized.append(
            np.stack([np.interp(progress, source, path[:, joint]) for joint in range(7)], axis=1)
        )
    combined = np.concatenate(synchronized, axis=1)
    combined[0] = q_initial
    combined[-1] = q_terminal
    return combined


def constrained_rrt_connect(
    start: np.ndarray,
    goal: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    in_collision,
    *,
    seed: int,
    max_iterations: int,
    step_size_rad: float,
    edge_step_rad: float,
    distance_weights: np.ndarray | None = None,
) -> tuple[np.ndarray | None, int]:
    """Bidirectional joint-space RRT whose edges obey the full path gate."""
    start = np.asarray(start, dtype=float)
    goal = np.asarray(goal, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    weights = (
        np.ones_like(start)
        if distance_weights is None
        else np.asarray(distance_weights, dtype=float)
    )
    if (
        start.ndim != 1
        or goal.shape != start.shape
        or lower.shape != start.shape
        or upper.shape != start.shape
        or weights.shape != start.shape
        or not all(np.isfinite(value).all() for value in (start, goal, lower, upper, weights))
        or np.any(lower >= upper)
        or np.any(weights <= 0)
        or max_iterations <= 0
        or not math.isfinite(step_size_rad)
        or step_size_rad <= 0
        or not math.isfinite(edge_step_rad)
        or edge_step_rad <= 0
    ):
        raise ValueError("invalid constrained RRT inputs")
    if in_collision(start) or in_collision(goal):
        raise ValueError("constrained RRT endpoints must satisfy every path gate")

    rng = np.random.default_rng(seed)
    trees = [
        {"nodes": [start.copy()], "parents": [-1], "root": "start"},
        {"nodes": [goal.copy()], "parents": [-1], "root": "goal"},
    ]

    def nearest(tree, target) -> int:
        nodes = np.asarray(tree["nodes"])
        return int(np.argmin(np.sum((nodes - target) ** 2 * weights, axis=1)))

    def steer(source, target) -> np.ndarray:
        delta = target - source
        distance = float(np.sqrt(np.sum(delta * delta * weights)))
        if distance <= step_size_rad:
            return target.copy()
        return source + delta * (step_size_rad / distance)

    def append_toward(tree, target) -> tuple[int | None, bool]:
        parent = nearest(tree, target)
        candidate = np.clip(steer(tree["nodes"][parent], target), lower, upper)
        if np.allclose(candidate, tree["nodes"][parent], atol=1e-12, rtol=0):
            return None, False
        if not segment_is_collision_free(
            tree["nodes"][parent], candidate, edge_step_rad, in_collision
        ):
            return None, False
        tree["nodes"].append(candidate)
        tree["parents"].append(parent)
        return len(tree["nodes"]) - 1, np.allclose(candidate, target, atol=1e-9, rtol=0)

    def root_path(tree, node_index) -> list[np.ndarray]:
        path = []
        while node_index >= 0:
            path.append(tree["nodes"][node_index])
            node_index = tree["parents"][node_index]
        return list(reversed(path))

    for iteration in range(1, max_iterations + 1):
        if rng.random() < 0.65:
            progress = rng.random()
            sample = (1.0 - progress) * start + progress * goal
            sample += rng.normal(0.0, 0.30, size=start.shape)
            sample = np.clip(sample, lower, upper)
        else:
            sample = rng.uniform(lower, upper)
        active_index, reached = append_toward(trees[0], sample)
        if active_index is not None:
            meeting = trees[0]["nodes"][active_index]
            other_index = None
            while True:
                other_index, reached = append_toward(trees[1], meeting)
                if other_index is None or reached:
                    break
            if reached and other_index is not None:
                active_path = root_path(trees[0], active_index)
                other_path = root_path(trees[1], other_index)
                if trees[0]["root"] == "start":
                    path = active_path + list(reversed(other_path[:-1]))
                else:
                    path = other_path + list(reversed(active_path[:-1]))
                return np.asarray(path), iteration
        trees.reverse()
    return None, max_iterations


def rack_width_constraint_in_base(
    root_pose_w: np.ndarray,
    rack_constraint: dict,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Express the configured rack-width slab in the robot-base frame."""
    root_pose_w = np.asarray(root_pose_w, dtype=float)
    center_w = np.asarray(rack_constraint["center_w_m"], dtype=float)
    axis_w = normalized_axis(rack_constraint["axis_w"], name="rack width axis")
    half_width_m = float(rack_constraint["half_width_m"])
    if root_pose_w.shape != (7,) or center_w.shape != (3,):
        raise ValueError("rack constraint requires one root pose and one world center")
    if not math.isfinite(half_width_m) or half_width_m <= 0:
        raise ValueError("rack half width must be finite and positive")
    world_from_base = pose_matrix(root_pose_w)
    base_from_world_rotation = world_from_base[:3, :3].T
    center_b = base_from_world_rotation @ (center_w - world_from_base[:3, 3])
    axis_b = normalized_axis(
        base_from_world_rotation @ axis_w, name="rack width axis in robot base"
    )
    return center_b, axis_b, half_width_m


def rack_width_coordinates_m(
    positions_b: np.ndarray,
    center_b: np.ndarray,
    axis_b: np.ndarray,
) -> np.ndarray:
    """Project one or more base-frame EEF positions onto the rack width axis."""
    positions_b = np.asarray(positions_b, dtype=float)
    center_b = np.asarray(center_b, dtype=float)
    axis_b = normalized_axis(axis_b, name="rack width axis in robot base")
    if positions_b.shape[-1] != 3 or center_b.shape != (3,):
        raise ValueError("EEF positions and rack center must have xyz coordinates")
    if not np.isfinite(positions_b).all() or not np.isfinite(center_b).all():
        raise ValueError("EEF positions and rack center must be finite")
    return np.einsum("...i,i->...", positions_b - center_b, axis_b)


def rack_width_max_violation_m(coordinates_m: np.ndarray, half_width_m: float) -> float:
    """Return the largest EEF excess beyond either physical rack side."""
    coordinates_m = np.asarray(coordinates_m, dtype=float)
    if not np.isfinite(coordinates_m).all():
        raise ValueError("rack-width coordinates must be finite")
    if not math.isfinite(half_width_m) or half_width_m <= 0:
        raise ValueError("rack half width must be finite and positive")
    return float(max(0.0, np.max(np.abs(coordinates_m)) - half_width_m))


def rack_width_task_space_limits(
    center_b: np.ndarray,
    axis_b: np.ndarray,
    half_width_m: float,
) -> list[list[float]] | None:
    """Return cuMotion limits when the rack-width axis matches a base axis."""
    center_b = np.asarray(center_b, dtype=float)
    axis_b = normalized_axis(axis_b, name="rack width axis in robot base")
    dominant_axis = int(np.argmax(np.abs(axis_b)))
    if abs(axis_b[dominant_axis]) < 1.0 - 1e-6:
        return None
    limits = [[-1.5, 1.5], [-1.5, 1.5], [-0.5, 2.5]]
    limits[dominant_axis] = [
        float(center_b[dominant_axis] - half_width_m),
        float(center_b[dominant_axis] + half_width_m),
    ]
    return limits


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot-dir", type=Path, required=True)
    result.add_argument("--urdf", type=Path, required=True)
    result.add_argument("--terminal-seed-plan", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--sphere-cell-m", type=float, default=0.06)
    result.add_argument(
        "--gripper-max-overshoot-m",
        type=float,
        choices=(0.002, 0.005, 0.010, 0.020),
        default=0.002,
    )
    result.add_argument("--collision-margin-m", type=float, default=0.002)
    result.add_argument("--self-pair-margin-m", type=float, default=None)
    result.add_argument("--validation-step-rad", type=float, default=0.01)
    result.add_argument("--target-tolerance-m", type=float, default=0.005)
    result.add_argument("--planner-seed", type=int, default=123456)
    result.add_argument("--planner-step-size", type=float, default=0.05)
    result.add_argument("--shoulder-sweep-weight", type=float, default=8.0)
    result.add_argument("--constrained-rrt-iterations", type=int, default=20000)
    result.add_argument("--constrained-rrt-step-rad", type=float, default=0.15)
    result.add_argument("--constrained-rrt-edge-step-rad", type=float, default=0.03)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for name, value in (
        ("--sphere-cell-m", args.sphere_cell_m),
        ("--validation-step-rad", args.validation_step_rad),
        ("--target-tolerance-m", args.target_tolerance_m),
        ("--planner-step-size", args.planner_step_size),
        ("--shoulder-sweep-weight", args.shoulder_sweep_weight),
        ("--constrained-rrt-step-rad", args.constrained_rrt_step_rad),
        ("--constrained-rrt-edge-step-rad", args.constrained_rrt_edge_step_rad),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0:
        raise ValueError("--collision-margin-m must be finite and nonnegative")
    if args.planner_seed <= 0:
        raise ValueError("--planner-seed must be positive")
    if args.constrained_rrt_iterations <= 0:
        raise ValueError("--constrained-rrt-iterations must be positive")
    self_pair_margin_m = (
        args.collision_margin_m
        if args.self_pair_margin_m is None
        else args.self_pair_margin_m
    )
    if not math.isfinite(self_pair_margin_m) or self_pair_margin_m < 0:
        raise ValueError("--self-pair-margin-m must be finite and nonnegative")

    import cumotion

    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    snapshot = json.loads((snapshot_dir / "collision_snapshot.json").read_text())
    runtime = json.loads((snapshot_dir / "runtime.json").read_text())
    world_config = json.loads((snapshot_dir / "world.json").read_text())
    seed_path = args.terminal_seed_plan.expanduser().resolve()
    seed_report = json.loads(seed_path.read_text())
    target_kind = seed_report.get("target", "pregrasp")
    if target_kind not in {"pregrasp", "grasp"}:
        raise ValueError(f"unsupported seed target: {target_kind}")
    orientation_constraint = seed_report.get("orientation_constraint", {"type": "none"})
    allow_target_flap_contact = bool(
        seed_report.get("collision_model", {}).get("allow_target_flap_contact", False)
    )
    world_config, allowed_contact_colliders = collision_world_config(
        snapshot,
        world_config,
        allow_target_flap_contact=allow_target_flap_contact,
    )
    closing_axis_tolerance_deg = None
    if (
        isinstance(orientation_constraint, dict)
        and orientation_constraint.get("type") == "terminal_axis"
    ):
        closing_axis_tolerance_deg = float(
            orientation_constraint["terminal_axis_deviation_limit_deg"]
        )
    output = args.output_dir or Path("/home/seonho/outputs/HumanoidScene") / (
        f"cumotion_bimanual_{target_kind}_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)

    defaults = {
        name: float(value)
        for name, value in zip(runtime["joint_names"], runtime["joint_positions"], strict=True)
        if name.startswith(("zarm_", "l_", "r_"))
    }
    defaults.update(
        {item["name"]: float(item["value"]) for item in runtime["pose_editor_state"]["joints"]}
    )
    q_initial = np.asarray([defaults[name] for name in ARM_JOINT_NAMES], dtype=float)
    joint_limits = {
        name: limits
        for name, limits in zip(
            runtime["joint_names"], runtime["joint_limits"], strict=True
        )
    }
    q_lower = np.asarray([joint_limits[name][0] for name in ARM_JOINT_NAMES], dtype=float)
    q_upper = np.asarray([joint_limits[name][1] for name in ARM_JOINT_NAMES], dtype=float)
    q_terminal = np.concatenate(
        [
            np.asarray(seed_report["arms"][side]["terminal_q_rad"], dtype=float)
            for side in ("left", "right")
        ]
    )
    if q_terminal.shape != q_initial.shape or not np.isfinite(q_terminal).all():
        raise ValueError("terminal seed plan does not contain one finite 14-DoF target")
    editor_state = runtime["pose_editor_state"]
    targets = np.asarray(editor_state[f"{target_kind}_position_b"], dtype=float)
    inward_normals = np.asarray(
        [
            normalized_axis(axis, name=f"{side} inward flap normal")
            for side, axis in zip(
                ("left", "right"), editor_state["inward_flap_normal_b"], strict=True
            )
        ]
    )
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
    xrdf_text = bimanual_xrdf(defaults, world_spheres, self_spheres)
    (output / "bimanual.xrdf").write_text(xrdf_text)

    robot = cumotion.load_robot_from_memory(
        xrdf_text, args.urdf.expanduser().resolve().read_text()
    )
    rack_constraint = runtime.get("rack_width_constraint")
    if not isinstance(rack_constraint, dict):
        raise ValueError(
            "runtime.json has no rack_width_constraint; capture a new planning snapshot"
        )
    rack_center_b, rack_axis_b, rack_half_width_m = rack_width_constraint_in_base(
        runtime["root_pose_w"], rack_constraint
    )
    task_space_limits = rack_width_task_space_limits(
        rack_center_b, rack_axis_b, rack_half_width_m
    )
    planner_text = planner_yaml(
        len(ARM_JOINT_NAMES),
        seed=args.planner_seed,
        step_size=args.planner_step_size,
        shoulder_sweep_weight=args.shoulder_sweep_weight,
        task_space_limits=task_space_limits,
    )
    (output / "planner.yaml").write_text(planner_text)

    world = cumotion.create_world()
    for obstacle_data in world_config["cuboid"].values():
        obstacle = cumotion.create_obstacle(cumotion.Obstacle.Type.CUBOID)
        obstacle.set_attribute(
            cumotion.Obstacle.Attribute.SIDE_LENGTHS,
            np.asarray(obstacle_data["dims"], dtype=float),
        )
        world.add_obstacle(obstacle, cumotion.Pose3(pose_matrix(obstacle_data["pose"])))
    world_view = world.add_world_view()
    inspector = cumotion.create_robot_world_inspector(robot, world_view)
    initial_world_collision = inspector.in_collision_with_obstacle(q_initial)
    initial_self_collision = inspector.in_self_collision(q_initial)
    terminal_world_collision = inspector.in_collision_with_obstacle(q_terminal)
    terminal_self_collision = inspector.in_self_collision(q_terminal)
    if any(
        (
            initial_world_collision,
            initial_self_collision,
            terminal_world_collision,
            terminal_self_collision,
        )
    ):
        raise RuntimeError(
            "invalid endpoint collision: "
            f"initial_world={initial_world_collision}, "
            f"initial_self={initial_self_collision}, "
            f"terminal_world={terminal_world_collision}, "
            f"terminal_self={terminal_self_collision}, "
            f"terminal_min_world_distance_m={inspector.min_distance_to_obstacle(q_terminal)}, "
            f"terminal_self_pairs={inspector.frames_in_self_collision(q_terminal)}"
        )

    def in_collision(q) -> bool:
        tcp_positions = np.asarray(
            [robot.kinematics().position(q, frame) for frame in TOOL_FRAMES]
        )
        rack_coordinates = rack_width_coordinates_m(
            tcp_positions, rack_center_b, rack_axis_b
        )
        return bool(
            inspector.in_collision_with_obstacle(q)
            or inspector.in_self_collision(q)
            or rack_width_max_violation_m(rack_coordinates, rack_half_width_m) > 1e-9
        )

    started = time.perf_counter()
    direct_path = np.stack((q_initial, q_terminal))
    direct_path_clear = segment_is_collision_free(
        q_initial,
        q_terminal,
        args.validation_step_rad,
        in_collision,
    )
    optimizer_seed_path = synchronized_seed_path(seed_report, q_initial, q_terminal)
    optimizer_seed_path_clear = not any(
        in_collision(row)
        for row in densify_path(optimizer_seed_path, args.validation_step_rad)
    )
    graph_waypoint_count = None
    constrained_rrt_iterations = 0
    constrained_rrt_waypoint_count = None
    shortcut_knot_count = 2
    if direct_path_clear:
        path_found = True
        path = densify_path(direct_path, args.planner_step_size)
        selected_strategy = "direct_joint_space_interpolation"
    elif optimizer_seed_path_clear:
        path_found = True
        shortcut_knots = shortcut_path(
            optimizer_seed_path, args.validation_step_rad, in_collision
        )
        shortcut_knot_count = len(shortcut_knots)
        path = densify_path(shortcut_knots, args.planner_step_size)
        selected_strategy = "synchronized_optimizer_seed_shortcut"
    else:
        distance_weights = np.ones(len(ARM_JOINT_NAMES))
        distance_weights[[0, 7]] = args.shoulder_sweep_weight
        rrt_path, constrained_rrt_iterations = constrained_rrt_connect(
            q_initial,
            q_terminal,
            q_lower,
            q_upper,
            in_collision,
            seed=args.planner_seed,
            max_iterations=args.constrained_rrt_iterations,
            step_size_rad=args.constrained_rrt_step_rad,
            edge_step_rad=args.constrained_rrt_edge_step_rad,
            distance_weights=distance_weights,
        )
        if rrt_path is not None:
            constrained_rrt_waypoint_count = len(rrt_path)
            shortcut_knot_count = len(rrt_path)
            path_found = True
            path = densify_path(rrt_path, args.planner_step_size)
            selected_strategy = "rack_width_constrained_rrt_connect"
        else:
            path_found = False
            path = None
            selected_strategy = "constrained_rrt_failed"
    if not path_found:
        config = cumotion.create_motion_planner_config_from_file(
            output / "planner.yaml", robot, TOOL_FRAMES[0], world_view
        )
        planner = cumotion.create_motion_planner(config)
        result = planner.plan_to_cspace_target(q_initial, q_terminal, True)
        path_found = bool(result.path_found)
        path = None
        selected_strategy = "graph_plan_failed"
        if path_found:
            graph_path = np.asarray(result.interpolated_path, dtype=float)
            graph_waypoint_count = len(graph_path)
            try:
                shortcut_knots = shortcut_path(
                    graph_path,
                    args.validation_step_rad,
                    in_collision,
                )
            except RuntimeError:
                path = graph_path
                shortcut_knot_count = None
                selected_strategy = "graph_plan_unshortened_validation"
            else:
                shortcut_knot_count = len(shortcut_knots)
                path = densify_path(shortcut_knots, args.planner_step_size)
                selected_strategy = "collision_aware_graph_shortcut"
    planning_wall_s = time.perf_counter() - started

    report = {
        "planner": "NVIDIA cuMotion 1.1.0 MotionPlanner",
        "strategy": selected_strategy,
        "target": target_kind,
        "status": "PLANNING_FAILURE",
        "snapshot_dir": str(snapshot_dir),
        "terminal_seed_plan": str(seed_path),
        "joint_names": ARM_JOINT_NAMES,
        "tool_frames": TOOL_FRAMES,
        "target_positions_b_m": targets.tolist(),
        "target_inward_flap_normals_b": inward_normals.tolist(),
        "orientation_constraint": orientation_constraint,
        "planning_wall_s": planning_wall_s,
        "planner_seed": args.planner_seed,
        "planner_step_size": args.planner_step_size,
        "shoulder_sweep_weight": args.shoulder_sweep_weight,
        "rack_width_constraint": {
            **rack_constraint,
            "center_b_m": rack_center_b.tolist(),
            "axis_b": rack_axis_b.tolist(),
            "allowed_coordinate_b_m": [-rack_half_width_m, rack_half_width_m],
            "applies_to": TOOL_FRAMES,
            "planner_left_eef_task_space_limits_b_m": task_space_limits,
        },
        "direct_path_collision_free": direct_path_clear,
        "synchronized_optimizer_seed_collision_free": optimizer_seed_path_clear,
        "synchronized_optimizer_seed_waypoint_count": len(optimizer_seed_path),
        "constrained_rrt_iterations": constrained_rrt_iterations,
        "constrained_rrt_waypoint_count": constrained_rrt_waypoint_count,
        "constrained_rrt_step_rad": args.constrained_rrt_step_rad,
        "constrained_rrt_edge_step_rad": args.constrained_rrt_edge_step_rad,
        "direct_joint_space_path_length_rad": joint_space_path_length(direct_path),
        "graph_waypoint_count": graph_waypoint_count,
        "shortcut_knot_count": shortcut_knot_count,
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
            "allow_target_flap_contact": allow_target_flap_contact,
            "allowed_contact_colliders": allowed_contact_colliders,
        },
    }
    if path_found:
        dense = densify_path(path, args.validation_step_rad)
        world_collisions = [inspector.in_collision_with_obstacle(row) for row in dense]
        self_collisions = [inspector.in_self_collision(row) for row in dense]
        distances = [inspector.min_distance_to_obstacle(row) for row in dense]
        tcp_path_positions = np.asarray(
            [
                [robot.kinematics().position(row, frame) for frame in TOOL_FRAMES]
                for row in dense
            ]
        )
        rack_width_coordinates = rack_width_coordinates_m(
            tcp_path_positions, rack_center_b, rack_axis_b
        )
        rack_width_violation = rack_width_max_violation_m(
            rack_width_coordinates, rack_half_width_m
        )
        terminal_positions = np.asarray(
            [robot.kinematics().position(path[-1], frame) for frame in TOOL_FRAMES]
        )
        terminal_errors = np.linalg.norm(terminal_positions - targets, axis=1)
        terminal_rotations = [
            np.asarray(robot.kinematics().orientation(path[-1], frame).matrix(), dtype=float)
            for frame in TOOL_FRAMES
        ]
        terminal_closing_axes = np.asarray(
            [rotation @ np.asarray((1.0, 0.0, 0.0)) for rotation in terminal_rotations]
        )
        closing_axis_errors_deg = np.asarray(
            [
                axis_alignment_error_deg(rotation, (1.0, 0.0, 0.0), normal)
                for rotation, normal in zip(terminal_rotations, inward_normals, strict=True)
            ]
        )
        left_delta = np.linalg.norm(np.diff(path[:, :7], axis=0), axis=1)
        right_delta = np.linalg.norm(np.diff(path[:, 7:], axis=0), axis=1)
        moving = (left_delta > 1e-7) | (right_delta > 1e-7)
        overlap = (left_delta > 1e-7) & (right_delta > 1e-7)
        overlap_fraction = float(overlap.sum() / max(1, moving.sum()))
        path_length = joint_space_path_length(path)
        success = bool(
            np.all(terminal_errors <= args.target_tolerance_m)
            and (
                closing_axis_tolerance_deg is None
                or np.all(closing_axis_errors_deg <= closing_axis_tolerance_deg + 1e-3)
            )
            and not any(world_collisions)
            and not any(self_collisions)
            and rack_width_violation <= 1e-9
            and overlap_fraction > 0.5
        )
        report.update(
            {
                "status": "SUCCESS" if success else "VALIDATION_FAILURE",
                "waypoint_q_rad": path.tolist(),
                "waypoint_count": len(path),
                "validation_q_rad": dense.tolist(),
                "validation_sample_count": len(dense),
                "validation_max_joint_step_rad": args.validation_step_rad,
                "terminal_q_rad": path[-1].tolist(),
                "terminal_tcp_positions_b_m": terminal_positions.tolist(),
                "terminal_error_m": terminal_errors.tolist(),
                "terminal_tcp_local_x_b": terminal_closing_axes.tolist(),
                "terminal_closing_axis_error_deg": closing_axis_errors_deg.tolist(),
                "sampled_world_collision": any(world_collisions),
                "sampled_self_collision": any(self_collisions),
                "sampled_min_world_distance_m": float(min(distances)),
                "terminal_self_collision_pairs": inspector.frames_in_self_collision(path[-1]),
                "simultaneous_motion_overlap_fraction": overlap_fraction,
                "joint_space_path_length_rad": path_length,
                "path_length_over_direct": path_length
                / joint_space_path_length(direct_path),
                "eef_rack_width_coordinate_range_b_m": [
                    [
                        float(rack_width_coordinates[:, arm].min()),
                        float(rack_width_coordinates[:, arm].max()),
                    ]
                    for arm in range(2)
                ],
                "eef_rack_width_max_violation_m": rack_width_violation,
                "tcp_path_axis_range_b_m": [
                    [
                        [float(tcp_path_positions[:, arm, axis].min()),
                         float(tcp_path_positions[:, arm, axis].max())]
                        for axis in range(3)
                    ]
                    for arm in range(2)
                ],
            }
        )
    else:
        success = False

    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "status": report["status"],
                "planning_wall_s": planning_wall_s,
                "waypoint_count": report.get("waypoint_count"),
                "validation_sample_count": report.get("validation_sample_count"),
                "terminal_error_m": report.get("terminal_error_m"),
                "terminal_closing_axis_error_deg": report.get(
                    "terminal_closing_axis_error_deg"
                ),
                "sampled_world_collision": report.get("sampled_world_collision"),
                "sampled_self_collision": report.get("sampled_self_collision"),
                "sampled_min_world_distance_m": report.get("sampled_min_world_distance_m"),
                "simultaneous_motion_overlap_fraction": report.get(
                    "simultaneous_motion_overlap_fraction"
                ),
                "eef_rack_width_max_violation_m": report.get(
                    "eef_rack_width_max_violation_m"
                ),
            },
            indent=2,
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
