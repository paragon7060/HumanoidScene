#!/usr/bin/env python3
"""Generate a simultaneous, collision-validated Task1 bimanual cuMotion path."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
import yaml

from data_collection.task1_cumotion_collision_plan import (
    SELF_COLLISION_IGNORE,
    pose_matrix,
    robot_spheres,
)


ARM_JOINT_NAMES = [
    f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)
]
TOOL_FRAMES = ["zarm_l7_end_effector", "zarm_r7_end_effector"]


def bimanual_xrdf(
    defaults: dict[str, float], world_spheres: dict, self_spheres: dict
) -> tuple[str, list[dict]]:
    """Build one 14-DoF robot and point controllers at every live collision sphere."""
    modifiers = []
    controllers = []
    for owner, spheres in world_spheres.items():
        for index, sphere in enumerate(spheres):
            frame = f"rmp_collision_{owner}_{index:03d}"
            modifiers.append(
                {
                    "add_frame": {
                        "frame_name": frame,
                        "parent_frame_name": owner,
                        "joint_name": f"{frame}_joint",
                        "joint_type": "fixed",
                        "fixed_transform": {
                            "position": sphere["center"],
                            "orientation": {"quaternion": [1.0, 0.0, 0.0, 0.0]},
                        },
                    }
                }
            )
            controllers.append({"name": frame, "radius": float(sphere["radius"])})

    data = {
        "format": "xrdf",
        "format_version": 2.0,
        "modifiers": modifiers,
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
    return yaml.safe_dump(data, sort_keys=False), controllers


def rmpflow_yaml(joint_count: int, controllers: list[dict]) -> str:
    """Return a conservative multi-target RMPflow policy configuration."""
    data = {
        "format": "rmpflow",
        "api_version": 2.0,
        "joint_limit_buffers": [0.01] * joint_count,
        "rmp_params": {
            "cspace_target_rmp": {
                "metric_scalar": 2.0,
                "position_gain": 80.0,
                "damping_gain": 40.0,
                "robust_position_term_thresh": 0.5,
                "inertia": 1.0,
            },
            "cspace_trajectory_rmp": {
                "p_gain": 80.0,
                "d_gain": 10.0,
                "ff_gain": 0.25,
                "weight": 50.0,
            },
            "cspace_affine_rmp": {
                "final_handover_time_std_dev": 0.25,
                "weight": 2000.0,
            },
            "joint_limit_rmp": {
                "metric_scalar": 1000.0,
                "metric_length_scale": 0.01,
                "metric_exploder_eps": 0.001,
                "metric_velocity_gate_length_scale": 0.01,
                "accel_damper_gain": 200.0,
                "accel_potential_gain": 1.0,
                "accel_potential_exploder_length_scale": 0.1,
                "accel_potential_exploder_eps": 0.01,
            },
            "joint_velocity_cap_rmp": {
                "max_velocity": 1.5,
                "velocity_damping_region": 0.35,
                "damping_gain": 300.0,
                "metric_weight": 100.0,
            },
            "target_rmp": {
                "accel_p_gain": 500.0,
                "accel_d_gain": 300.0,
                "accel_norm_eps": 0.075,
                "metric_alpha_length_scale": 0.05,
                "min_metric_alpha": 0.01,
                "max_metric_scalar": 10000.0,
                "min_metric_scalar": 2500.0,
                "proximity_metric_boost_scalar": 20.0,
                "proximity_metric_boost_length_scale": 0.02,
                "accept_user_weights": False,
            },
            "axis_target_rmp": {
                "accel_p_gain": 200.0,
                "accel_d_gain": 40.0,
                "metric_scalar": 10.0,
                "proximity_metric_boost_scalar": 3000.0,
                "proximity_metric_boost_length_scale": 0.05,
                "accept_user_weights": False,
            },
            "collision_rmp": {
                "damping_gain": 50.0,
                "damping_std_dev": 0.04,
                "damping_robustness_eps": 0.01,
                "damping_velocity_gate_length_scale": 0.01,
                "repulsion_gain": 1000.0,
                "repulsion_std_dev": 0.01,
                "metric_modulation_radius": 0.5,
                "metric_scalar": 500.0,
                "metric_exploder_std_dev": 0.02,
                "metric_exploder_eps": 0.001,
            },
            "damping_rmp": {
                "accel_d_gain": 30.0,
                "metric_scalar": 50.0,
                "inertia": 100.0,
            },
        },
        "canonical_resolve": {
            "max_acceleration_norm": 30.0,
            "projection_tolerance": 0.01,
            "verbose": False,
        },
        "body_capsules": [],
        "body_collision_controllers": controllers,
    }
    return yaml.safe_dump(data, sort_keys=False)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot-dir", type=Path, required=True)
    result.add_argument("--urdf", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--sphere-cell-m", type=float, default=0.06)
    result.add_argument("--collision-margin-m", type=float, default=0.002)
    result.add_argument("--dt", type=float, default=0.005)
    result.add_argument("--duration-s", type=float, default=12.0)
    result.add_argument("--validation-samples", type=int, default=121)
    result.add_argument("--target-tolerance-m", type=float, default=0.005)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for name, value in (
        ("--sphere-cell-m", args.sphere_cell_m),
        ("--dt", args.dt),
        ("--duration-s", args.duration_s),
        ("--target-tolerance-m", args.target_tolerance_m),
    ):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0:
        raise ValueError("--collision-margin-m must be finite and nonnegative")
    if args.validation_samples < 2:
        raise ValueError("--validation-samples must be at least two")

    import cumotion

    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    snapshot = json.loads((snapshot_dir / "collision_snapshot.json").read_text())
    runtime = json.loads((snapshot_dir / "runtime.json").read_text())
    world_config = json.loads((snapshot_dir / "world.json").read_text())
    output = args.output_dir or Path("/home/seonho/outputs/HumanoidScene") / (
        "cumotion_bimanual_pregrasp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    targets = np.asarray(runtime["pose_editor_state"]["pregrasp_position_b"], dtype=float)
    world_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.collision_margin_m
    )
    self_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.collision_margin_m / 2
    )
    xrdf_text, controllers = bimanual_xrdf(defaults, world_spheres, self_spheres)
    rmp_text = rmpflow_yaml(len(ARM_JOINT_NAMES), controllers)
    (output / "bimanual.xrdf").write_text(xrdf_text)
    (output / "rmpflow.yaml").write_text(rmp_text)

    robot = cumotion.load_robot_from_memory(xrdf_text, args.urdf.expanduser().resolve().read_text())
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
    if inspector.in_collision_with_obstacle(q_initial) or inspector.in_self_collision(q_initial):
        raise RuntimeError("initial 14-DoF state is in collision")

    config = cumotion.create_rmpflow_config_from_file(output / "rmpflow.yaml", robot, world_view)
    policy = cumotion.create_rmpflow(config)
    for frame, target in zip(TOOL_FRAMES, targets, strict=True):
        policy.add_target_frame(frame)
        policy.set_position_target(frame, target)
    policy.set_cspace_attractor(q_initial)

    q = q_initial.copy()
    qd = np.zeros_like(q)
    qdd = np.zeros_like(q)
    steps = int(math.ceil(args.duration_s / args.dt))
    history = [q.copy()]
    for _ in range(steps):
        policy.eval_accel(q, qd, qdd)
        qd += args.dt * qdd
        q += args.dt * qd
        history.append(q.copy())

    history = np.asarray(history)
    sample_indices = np.linspace(0, len(history) - 1, args.validation_samples).round().astype(int)
    samples = history[sample_indices]
    times = sample_indices * args.dt
    world_collisions = [inspector.in_collision_with_obstacle(row) for row in samples]
    self_collisions = [inspector.in_self_collision(row) for row in samples]
    distances = [inspector.min_distance_to_obstacle(row) for row in samples]
    terminal_positions = np.asarray(
        [robot.kinematics().position(samples[-1], frame) for frame in TOOL_FRAMES]
    )
    terminal_errors = np.linalg.norm(terminal_positions - targets, axis=1)
    left_delta = np.linalg.norm(np.diff(samples[:, :7], axis=0), axis=1)
    right_delta = np.linalg.norm(np.diff(samples[:, 7:], axis=0), axis=1)
    moving = (left_delta > 1e-6) | (right_delta > 1e-6)
    overlap = (left_delta > 1e-6) & (right_delta > 1e-6)
    overlap_fraction = float(overlap.sum() / max(1, moving.sum()))

    success = bool(
        np.all(terminal_errors <= args.target_tolerance_m)
        and not any(world_collisions)
        and not any(self_collisions)
        and overlap_fraction > 0.5
    )
    report = {
        "planner": "NVIDIA cuMotion 1.1.0 RMPflow multi-target",
        "strategy": "simultaneous_bimanual_14dof",
        "status": "SUCCESS" if success else "VALIDATION_FAILURE",
        "snapshot_dir": str(snapshot_dir),
        "joint_names": ARM_JOINT_NAMES,
        "tool_frames": TOOL_FRAMES,
        "target_positions_b_m": targets.tolist(),
        "orientation_constraint": "none",
        "duration_s": float(times[-1]),
        "sample_times_s": times.tolist(),
        "sample_q_rad": samples.tolist(),
        "terminal_q_rad": samples[-1].tolist(),
        "terminal_tcp_positions_b_m": terminal_positions.tolist(),
        "terminal_error_m": terminal_errors.tolist(),
        "sampled_world_collision": any(world_collisions),
        "sampled_self_collision": any(self_collisions),
        "sampled_min_world_distance_m": float(min(distances)),
        "terminal_self_collision_pairs": inspector.frames_in_self_collision(samples[-1]),
        "simultaneous_motion_overlap_fraction": overlap_fraction,
        "collision_model": {
            "world_obstacles": len(world_config["cuboid"]),
            "robot_world_spheres": sum(map(len, world_spheres.values())),
            "robot_self_spheres": sum(map(len, self_spheres.values())),
            "rmp_collision_controllers": len(controllers),
            "sphere_cover_cell_m": args.sphere_cell_m,
            "world_margin_m": args.collision_margin_m,
            "self_pair_margin_m": args.collision_margin_m,
        },
    }
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "status": report["status"],
                "terminal_error_m": report["terminal_error_m"],
                "sampled_world_collision": report["sampled_world_collision"],
                "sampled_self_collision": report["sampled_self_collision"],
                "sampled_min_world_distance_m": report["sampled_min_world_distance_m"],
                "simultaneous_motion_overlap_fraction": overlap_fraction,
            },
            indent=2,
        )
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
