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
    joint_count: int, *, seed: int = 123456, step_size: float = 0.05
) -> str:
    """Return deterministic cuMotion graph-planner parameters for the workcell."""
    data = {
        "seed": seed,
        "step_size": step_size,
        "max_iterations": 100000,
        "max_sampling": 30000,
        "distance_metric_weights": [1.0] * joint_count,
        "task_space_limits": [[-1.5, 1.5], [-1.5, 1.5], [-0.5, 2.5]],
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
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for name, value in (
        ("--sphere-cell-m", args.sphere_cell_m),
        ("--validation-step-rad", args.validation_step_rad),
        ("--target-tolerance-m", args.target_tolerance_m),
        ("--planner-step-size", args.planner_step_size),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0:
        raise ValueError("--collision-margin-m must be finite and nonnegative")
    if args.planner_seed <= 0:
        raise ValueError("--planner-seed must be positive")
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
    planner_text = planner_yaml(
        len(ARM_JOINT_NAMES),
        seed=args.planner_seed,
        step_size=args.planner_step_size,
    )
    (output / "bimanual.xrdf").write_text(xrdf_text)
    (output / "planner.yaml").write_text(planner_text)

    robot = cumotion.load_robot_from_memory(
        xrdf_text, args.urdf.expanduser().resolve().read_text()
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
        return bool(
            inspector.in_collision_with_obstacle(q)
            or inspector.in_self_collision(q)
        )

    started = time.perf_counter()
    direct_path = np.stack((q_initial, q_terminal))
    direct_path_clear = segment_is_collision_free(
        q_initial,
        q_terminal,
        args.validation_step_rad,
        in_collision,
    )
    graph_waypoint_count = None
    shortcut_knot_count = 2
    if direct_path_clear:
        path_found = True
        path = densify_path(direct_path, args.planner_step_size)
        selected_strategy = "direct_joint_space_interpolation"
    else:
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
            shortcut_knots = shortcut_path(
                graph_path,
                args.validation_step_rad,
                in_collision,
            )
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
        "direct_path_collision_free": direct_path_clear,
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
            },
            indent=2,
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
