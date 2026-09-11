#!/usr/bin/env python3
"""Solve simultaneous Task1 bimanual grasp endpoints with cuMotion RMPflow."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path
from typing import Callable

import numpy as np
import yaml

from kuavo_isaaclab_scene.robots.end_effector import urdf_with_center_frames
from data_collection.task1.approach import TOOL_FRAMES
from data_collection.task1.contract import (
    ARM_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    safe_waist_bounds,
)
from data_collection.task1.collision import (
    SELF_COLLISION_IGNORE,
    axis_alignment_error_deg,
    collision_world_config,
    load_gripper_mesh_spheres,
    pose_matrix,
    robot_spheres,
    rotation_error_deg,
    runtime_joint_defaults,
    target_flap_line_geometry,
    tool_down_angle_deg,
    tool_down_orientation_targets,
)


def center_out_offsets(length_m: float, point_count: int) -> np.ndarray:
    """Return evenly spaced offsets ordered from the segment center outward."""
    if not math.isfinite(length_m) or length_m <= 0:
        raise ValueError("line length must be finite and positive")
    if point_count < 3 or point_count % 2 == 0:
        raise ValueError("line point count must be an odd integer of at least three")
    offsets = np.linspace(-length_m / 2, length_m / 2, point_count)
    return offsets[np.argsort(np.abs(offsets), kind="stable")]


def candidate_angles(
    maximum_deg: float, step_deg: float, minimum_deg: float = 0.0
) -> np.ndarray:
    """Return ordered angle candidates including a non-step-aligned maximum."""
    if not all(map(math.isfinite, (minimum_deg, maximum_deg, step_deg))):
        raise ValueError("angles must be finite")
    if not 0 <= minimum_deg <= maximum_deg <= 90 or step_deg <= 0:
        raise ValueError(
            "angles require 0 <= minimum <= maximum <= 90 and a positive step"
        )
    count = int(math.floor((maximum_deg - minimum_deg) / step_deg)) + 1
    angles = minimum_deg + np.arange(count, dtype=float) * step_deg
    if angles[-1] < maximum_deg - 1e-9:
        angles = np.r_[angles, maximum_deg]
    else:
        angles[-1] = maximum_deg
    return angles


def front_target_candidates(
    x_offsets_m: list[float] | tuple[float, ...], z_offset_m: float
) -> np.ndarray:
    """Return robot-base offsets ordered from the front-most grasp inward."""
    x_values = np.asarray(x_offsets_m, dtype=float)
    if x_values.ndim != 1 or not len(x_values) or not np.isfinite(x_values).all():
        raise ValueError("target x offsets must be a non-empty finite list")
    if np.any(x_values < 0) or not math.isfinite(z_offset_m):
        raise ValueError("front target offsets must be nonnegative and finite")
    return np.column_stack((x_values, np.zeros_like(x_values), np.full_like(x_values, z_offset_m)))


def rmpflow_xrdf(
    cspace_joint_names: list[str],
    defaults: dict[str, float],
    world_spheres: dict,
    self_spheres: dict,
) -> str:
    """Build one robot description whose c-space contains both arms and optional waist."""
    data = {
        "format": "xrdf",
        "format_version": 2.0,
        "default_joint_positions": defaults,
        "cspace": {
            "joint_names": cspace_joint_names,
            "acceleration_limits": [10.0] * len(cspace_joint_names),
            "jerk_limits": [100.0] * len(cspace_joint_names),
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


def rmpflow_config(cspace_size: int) -> str:
    """Return the NVIDIA G1-style RMPflow gains for the requested c-space."""
    data = {
        "format": "rmpflow",
        "api_version": 2.0,
        "joint_limit_buffers": [0.01] * cspace_size,
        "rmp_params": {
            "cspace_target_rmp": {
                "metric_scalar": 50.0,
                "position_gain": 100.0,
                "damping_gain": 50.0,
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
                "metric_exploder_eps": 1e-3,
                "metric_velocity_gate_length_scale": 0.01,
                "accel_damper_gain": 200.0,
                "accel_potential_gain": 1.0,
                "accel_potential_exploder_length_scale": 0.1,
                "accel_potential_exploder_eps": 1e-2,
            },
            "joint_velocity_cap_rmp": {
                "max_velocity": 2.15,
                "velocity_damping_region": 0.5,
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
                "damping_robustness_eps": 1e-2,
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
            "max_acceleration_norm": 50.0,
            "projection_tolerance": 0.01,
            "verbose": False,
        },
        # Endpoint collision is checked with the full XRDF inspector. Empty
        # controller lists keep this solve focused on simultaneous task IK.
        "body_capsules": [],
        "body_collision_controllers": [],
    }
    return yaml.safe_dump(data, sort_keys=False)


def rotation_vector(rotation_matrix: np.ndarray) -> np.ndarray:
    """Return the shortest SO(3) rotation vector represented by a matrix."""
    rotation = np.asarray(rotation_matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("rotation matrix must be finite 3x3")
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ]
    )
    if angle < 1e-8:
        return skew / 2.0
    if math.pi - angle < 1e-5:
        diagonal = np.maximum(0.0, (np.diag(rotation) + 1.0) / 2.0)
        axis = np.sqrt(diagonal)
        axis *= np.sign(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ]
        )
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            axis = np.asarray((1.0, 0.0, 0.0))
        else:
            axis /= norm
        return angle * axis
    return angle * skew / (2.0 * math.sin(angle))


def simultaneous_pose_errors(
    kinematics,
    q: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Evaluate both TCP pose errors and their cuMotion Jacobians."""
    positions = np.asarray(
        [kinematics.position(q, frame) for frame in TOOL_FRAMES], dtype=float
    )
    rotations = [
        np.asarray(kinematics.orientation(q, frame).matrix(), dtype=float)
        for frame in TOOL_FRAMES
    ]
    position_errors = target_positions - positions
    rotation_errors = [
        rotation_vector(target @ actual.T)
        for target, actual in zip(target_rotations, rotations, strict=True)
    ]
    jacobians = [
        np.asarray(kinematics.jacobian(q, frame), dtype=float)
        for frame in TOOL_FRAMES
    ]
    return position_errors, np.asarray(rotation_errors), rotations, jacobians


def solve_simultaneous_jacobian(
    kinematics,
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_positions: np.ndarray,
    target_rotations: list[np.ndarray],
    *,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    max_iterations: int = 250,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Solve both TCP poses in one c-space with damped cuMotion Jacobians."""
    q = np.asarray(q_seed, dtype=float).copy()
    position_scale = max(position_tolerance_m, 0.01)
    orientation_scale = max(math.radians(orientation_tolerance_deg), math.radians(5.0))
    damping = 1e-3
    for iteration in range(max_iterations):
        pos_e, rot_e, _, jacobians = simultaneous_pose_errors(
            kinematics, q, target_positions, target_rotations
        )
        if (
            np.all(np.linalg.norm(pos_e, axis=1) <= position_tolerance_m)
            and np.all(
                np.linalg.norm(rot_e, axis=1)
                <= math.radians(orientation_tolerance_deg)
            )
        ):
            return q, np.zeros_like(q), iteration + 1
        residual = np.concatenate(
            [
                value
                for pair in zip(
                    pos_e / position_scale,
                    rot_e / orientation_scale,
                    strict=True,
                )
                for value in pair
            ]
        )
        jacobian = np.vstack(
            [
                value
                for pair in (
                    (
                        full[:3] / position_scale,
                        full[3:] / orientation_scale,
                    )
                    for full in jacobians
                )
                for value in pair
            ]
        )
        normal = jacobian.T @ jacobian + damping * np.eye(len(q))
        delta = np.linalg.solve(normal, jacobian.T @ residual)
        max_delta = float(np.max(np.abs(delta)))
        if max_delta > 0.15:
            delta *= 0.15 / max_delta
        old_cost = float(residual @ residual)
        accepted = False
        for step_scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
            candidate = np.clip(q + step_scale * delta, lower + 1e-6, upper - 1e-6)
            candidate_pos_e, candidate_rot_e, _, _ = simultaneous_pose_errors(
                kinematics, candidate, target_positions, target_rotations
            )
            candidate_residual = np.concatenate(
                [
                    value
                    for pair in zip(
                        candidate_pos_e / position_scale,
                        candidate_rot_e / orientation_scale,
                        strict=True,
                    )
                    for value in pair
                ]
            )
            if float(candidate_residual @ candidate_residual) < old_cost:
                q = candidate
                damping = max(1e-6, damping / 2.0)
                accepted = True
                break
        if not accepted:
            damping = min(1e3, damping * 10.0)
            if damping >= 1e3 or np.linalg.norm(delta) < 1e-8:
                return q, np.zeros_like(q), iteration + 1
    return q, np.zeros_like(q), max_iterations


def integrate_rmpflow(
    flow,
    q_seed: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    timestep_s: float,
    duration_s: float,
    convergence_check: Callable[[np.ndarray], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Integrate the same semi-implicit RMPflow update used by NVIDIA's controller."""
    q = np.asarray(q_seed, dtype=float).copy()
    velocity = np.zeros_like(q)
    accel = np.zeros_like(q)
    step_count = int(math.ceil(duration_s / timestep_s))
    for step in range(step_count):
        flow.eval_accel(q, velocity, accel)
        velocity += timestep_s * accel
        q += timestep_s * velocity
        clipped = np.clip(q, lower + 1e-6, upper - 1e-6)
        velocity[clipped != q] = 0.0
        q = clipped
        if not np.isfinite(q).all() or not np.isfinite(velocity).all():
            raise RuntimeError("RMPflow integration produced a nonfinite state")
        if (
            convergence_check is not None
            and step > 100
            and step % 20 == 0
            and np.linalg.norm(velocity) < 0.02
            and convergence_check(q)
        ):
            return q, velocity, step + 1
        if (
            convergence_check is None
            and step > 100
            and np.linalg.norm(velocity) < 1e-5
            and np.linalg.norm(accel) < 1e-4
        ):
            return q, velocity, step + 1
    return q, velocity, step_count


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot-dir", type=Path, required=True)
    result.add_argument("--urdf", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument(
        "--seed-plan",
        type=Path,
        default=None,
        help="Initialize the simultaneous solve from an existing bimanual endpoint plan.",
    )
    result.add_argument(
        "--arm-only-baseline",
        action="store_true",
        help="Use the historical 14DoF arms-only solve instead of the default 16DoF solve.",
    )
    result.add_argument(
        "--waist-seed-deg",
        type=float,
        nargs=2,
        metavar=("PITCH", "YAW"),
        default=None,
        help="Override the initial waist pitch/yaw used by a 16-DoF solve.",
    )
    result.add_argument(
        "--solver", choices=("rmpflow", "jacobian"), default="rmpflow"
    )
    result.add_argument(
        "--target-x-offsets-m", type=float, nargs="+", default=[0.03, 0.04, 0.05]
    )
    result.add_argument("--target-z-offset-m", type=float, default=0.015)
    result.add_argument(
        "--tool-down-angles-deg", type=float, nargs="+", default=[45.0, 50.0, 56.0]
    )
    result.add_argument(
        "--target-offset-b-m",
        type=float,
        nargs=3,
        default=None,
        help="Explicit single XYZ override for baseline reproduction.",
    )
    result.add_argument("--flap-line-length-m", type=float, default=0.20)
    result.add_argument("--flap-line-point-count", type=int, default=21)
    result.add_argument("--position-tolerance-m", type=float, default=0.003)
    result.add_argument("--orientation-tolerance-deg", type=float, default=1.0)
    result.add_argument("--timestep-s", type=float, default=0.002)
    result.add_argument("--duration-s", type=float, default=8.0)
    result.add_argument("--sphere-cell-m", type=float, default=0.055)
    result.add_argument("--gripper-max-overshoot-m", type=float, default=0.002)
    result.add_argument("--collision-margin-m", type=float, default=0.004)
    result.add_argument("--self-pair-margin-m", type=float, default=0.002)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    for name in (
        "position_tolerance_m",
        "orientation_tolerance_deg",
        "timestep_s",
        "duration_s",
        "sphere_cell_m",
        "gripper_max_overshoot_m",
    ):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive")
    for name in ("collision_margin_m", "self_pair_margin_m"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")

    offsets = center_out_offsets(args.flap_line_length_m, args.flap_line_point_count)
    angles = np.asarray(args.tool_down_angles_deg, dtype=float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError("--tool-down-angles-deg must be a non-empty finite list")
    if np.any((angles < 0) | (angles > 90)):
        raise ValueError("--tool-down-angles-deg values must lie within [0, 90]")
    if args.target_offset_b_m is None:
        target_offsets = front_target_candidates(
            args.target_x_offsets_m, args.target_z_offset_m
        )
    else:
        target_offsets = np.asarray([args.target_offset_b_m], dtype=float)
        if target_offsets.shape != (1, 3) or not np.isfinite(target_offsets).all():
            raise ValueError("--target-offset-b-m must be finite xyz")

    import cumotion

    snapshot_dir = args.snapshot_dir.expanduser().resolve()
    snapshot = json.loads((snapshot_dir / "collision_snapshot.json").read_text())
    runtime = json.loads((snapshot_dir / "runtime.json").read_text())
    world_config = json.loads((snapshot_dir / "world.json").read_text())
    world_config, _ = collision_world_config(
        snapshot, world_config, allow_target_flap_contact=False
    )
    defaults = runtime_joint_defaults(runtime)
    include_waist = not args.arm_only_baseline
    cspace_names = WAIST_JOINT_NAMES + ARM_JOINT_NAMES if include_waist else ARM_JOINT_NAMES
    joint_limits = {
        name: limits
        for name, limits in zip(
            runtime["joint_names"], runtime["joint_limits"], strict=True
        )
    }
    lower = np.asarray([joint_limits[name][0] for name in cspace_names], dtype=float)
    upper = np.asarray([joint_limits[name][1] for name in cspace_names], dtype=float)
    if include_waist:
        lower[:2], upper[:2] = safe_waist_bounds(joint_limits)
    q_initial = np.asarray([defaults[name] for name in cspace_names], dtype=float)
    if args.waist_seed_deg is not None:
        if not include_waist:
            raise ValueError("--waist-seed-deg is unavailable with --arm-only-baseline")
        waist_seed = np.radians(np.asarray(args.waist_seed_deg, dtype=float))
        if not np.isfinite(waist_seed).all():
            raise ValueError("--waist-seed-deg must be finite")
        q_initial[:2] = waist_seed
        if np.any(q_initial[:2] <= lower[:2]) or np.any(q_initial[:2] >= upper[:2]):
            raise ValueError("--waist-seed-deg lies outside captured joint limits")
    q_seed = q_initial.copy()
    seed_source = runtime.get("initial_state")
    if args.seed_plan is not None:
        seed_path = args.seed_plan.expanduser().resolve()
        seed_report = json.loads(seed_path.read_text())
        seed_arms = seed_report.get("terminal_arm_q_rad")
        if seed_arms is None:
            seed_arms = np.concatenate(
                [seed_report["arms"][side]["terminal_q_rad"] for side in ("left", "right")]
            )
        seed_arms = np.asarray(seed_arms, dtype=float)
        if seed_arms.shape != (14,) or not np.isfinite(seed_arms).all():
            raise ValueError("--seed-plan must contain one finite 14-DoF arm endpoint")
        q_seed[-14:] = seed_arms
        seed_source = str(seed_path)

    mesh_spheres = load_gripper_mesh_spheres(args.gripper_max_overshoot_m)
    world_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.collision_margin_m, mesh_spheres
    )
    self_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.self_pair_margin_m / 2, mesh_spheres
    )
    xrdf_text = rmpflow_xrdf(cspace_names, defaults, world_spheres, self_spheres)
    urdf_text = urdf_with_center_frames(
        args.urdf.expanduser().resolve().read_text()
    )
    robot = cumotion.load_robot_from_memory(xrdf_text, urdf_text)

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
    flow_config_text = rmpflow_config(len(cspace_names))
    flow_config = cumotion.create_rmpflow_config_from_memory(
        flow_config_text, robot, world_view
    )
    flow = cumotion.create_rmpflow(flow_config)
    for frame in TOOL_FRAMES:
        flow.add_target_frame(frame)
    flow.set_cspace_attractor(q_seed)

    editor_state = runtime["pose_editor_state"]
    nominal_centers = np.asarray(editor_state["grasp_position_b"], dtype=float)
    normals = np.asarray(editor_state["inward_flap_normal_b"], dtype=float)
    line_axes, full_lengths = target_flap_line_geometry(snapshot, runtime)
    if args.flap_line_length_m > float(np.min(full_lengths)) + 1e-9:
        raise ValueError(
            f"flap line length {args.flap_line_length_m} exceeds {full_lengths.tolist()}"
        )

    output = args.output_dir or Path("/home/seonho/outputs/HumanoidScene") / (
        "cumotion_bimanual_rmpflow_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    output = output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.mkdir(parents=True)
    (output / "robot.xrdf").write_text(xrdf_text)
    (output / "rmpflow.yaml").write_text(flow_config_text)

    kinematics = robot.kinematics()
    attempts = []
    best = None
    selected = None
    for target_offset, angle, offset in product(target_offsets, angles, offsets):
        centers = nominal_centers + target_offset
        rotations = [
            tool_down_orientation_targets(normal, angle, angle, 1.0)[0][0]
            for normal in normals
        ]
        positions = centers + offset * line_axes
        for frame, position, rotation in zip(
            TOOL_FRAMES, positions, rotations, strict=True
        ):
            flow.set_pose_target(
                frame,
                cumotion.Pose3(
                    cumotion.Rotation3.from_matrix(rotation), position
                ),
            )

        def target_is_converged(q_value: np.ndarray) -> bool:
            actual_positions = np.asarray(
                [kinematics.position(q_value, frame) for frame in TOOL_FRAMES],
                dtype=float,
            )
            actual_rotations = [
                np.asarray(
                    kinematics.orientation(q_value, frame).matrix(), dtype=float
                )
                for frame in TOOL_FRAMES
            ]
            return bool(
                np.all(
                    np.linalg.norm(actual_positions - positions, axis=1)
                    <= args.position_tolerance_m
                )
                and all(
                    rotation_error_deg(actual, target)
                    <= args.orientation_tolerance_deg
                    for actual, target in zip(
                        actual_rotations, rotations, strict=True
                    )
                )
            )

        if args.solver == "rmpflow":
            q, velocity, steps = integrate_rmpflow(
                flow,
                q_seed,
                lower,
                upper,
                timestep_s=args.timestep_s,
                duration_s=args.duration_s,
                convergence_check=target_is_converged,
            )
        else:
            q, velocity, steps = solve_simultaneous_jacobian(
                kinematics,
                q_seed,
                lower,
                upper,
                positions,
                rotations,
                position_tolerance_m=args.position_tolerance_m,
                orientation_tolerance_deg=args.orientation_tolerance_deg,
            )
        terminal_positions = np.asarray(
            [kinematics.position(q, frame) for frame in TOOL_FRAMES], dtype=float
        )
        terminal_rotations = [
            np.asarray(kinematics.orientation(q, frame).matrix(), dtype=float)
            for frame in TOOL_FRAMES
        ]
        position_errors = np.linalg.norm(terminal_positions - positions, axis=1)
        orientation_errors = np.asarray(
            [
                rotation_error_deg(actual, target)
                for actual, target in zip(
                    terminal_rotations, rotations, strict=True
                )
            ]
        )
        world_collision = bool(inspector.in_collision_with_obstacle(q))
        self_collision = bool(inspector.in_self_collision(q))
        score = float(
            np.max(position_errors) / args.position_tolerance_m
            + np.max(orientation_errors) / args.orientation_tolerance_deg
            + 1000.0 * (world_collision or self_collision)
        )
        attempt = {
            "target_offset_b_m": target_offset.tolist(),
            "angle_deg": float(angle),
            "flap_line_offset_m": float(offset),
            "steps": steps,
            "velocity_norm": float(np.linalg.norm(velocity)),
            "position_error_m": position_errors.tolist(),
            "orientation_error_deg": orientation_errors.tolist(),
            "tool_down_angle_deg": [
                tool_down_angle_deg(rotation) for rotation in terminal_rotations
            ],
            "closing_axis_error_deg": [
                axis_alignment_error_deg(rotation, (1, 0, 0), normal)
                for rotation, normal in zip(
                    terminal_rotations, normals, strict=True
                )
            ],
            "world_collision": world_collision,
            "self_collision": self_collision,
            "min_world_distance_m": float(inspector.min_distance_to_obstacle(q)),
            "score": score,
        }
        attempts.append(attempt)
        candidate = (score, q.copy(), positions.copy(), attempt)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if (
            np.all(position_errors <= args.position_tolerance_m)
            and np.all(orientation_errors <= args.orientation_tolerance_deg)
            and not world_collision
            and not self_collision
        ):
            selected = candidate
            break

    chosen = selected or best
    assert chosen is not None
    _, terminal_q, target_positions, chosen_attempt = chosen
    arm_start = len(WAIST_JOINT_NAMES) if include_waist else 0
    report = {
        "schema_version": 2,
        "planner": (
            "NVIDIA cuMotion 1.1.0 RMPflow simultaneous bimanual endpoint"
            if args.solver == "rmpflow"
            else "cuMotion kinematics simultaneous bimanual Jacobian endpoint"
        ),
        "solver": args.solver,
        "status": "SUCCESS" if selected is not None else "IK_FAILURE",
        "target": "grasp",
        "snapshot_dir": str(snapshot_dir),
        "initial_pose_source": runtime.get("initial_state"),
        "rmpflow_seed_source": seed_source,
        "cspace_joint_names": cspace_names,
        "cspace_dof": len(cspace_names),
        "include_waist": include_waist,
        "arm_only_baseline": args.arm_only_baseline,
        "torso_height_m": runtime.get("pose_editor_state", {}).get("torso_height_m"),
        "waist_seed_deg": args.waist_seed_deg,
        "target_positions_b_m": target_positions.tolist(),
        "target_flap_line": {
            "axis_b": line_axes.tolist(),
            "length_m": args.flap_line_length_m,
            "full_collider_length_m": full_lengths.tolist(),
        },
        "angle_candidates_deg": angles.tolist(),
        "selected": chosen_attempt,
        "terminal_q_rad": terminal_q.tolist(),
        "terminal_waist_q_rad": (
            terminal_q[:2].tolist() if include_waist else None
        ),
        "terminal_arm_q_rad": terminal_q[arm_start:].tolist(),
        "arms": {
            side: {
                "terminal_q_rad": terminal_q[
                    arm_start + index * 7 : arm_start + (index + 1) * 7
                ].tolist(),
                "target_position_b_m": target_positions[index].tolist(),
            }
            for index, side in enumerate(("left", "right"))
        },
        "attempts": attempts,
    }
    plan_path = output / "plan.json"
    plan_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "plan": str(plan_path),
                "status": report["status"],
                "cspace_dof": report["cspace_dof"],
                "include_waist": report["include_waist"],
                "selected": report["selected"],
                "terminal_waist_q_rad": report["terminal_waist_q_rad"],
            },
            indent=2,
        )
    )
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
