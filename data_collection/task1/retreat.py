#!/usr/bin/env python3
"""Build a synchronized Task1 lift/pull path without attached-object collision."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from data_collection.task1.contract import (
    ARM_JOINT_NAMES,
    WAIST_ARM_JOINT_NAMES,
    compose_waist_arm,
)


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from kuavo_isaaclab_scene.planning.robot_model import UrdfModel
from kuavo_isaaclab_scene.robots.end_effector import (
    CENTER_FRAME_NAME,
    CENTER_TOOL_FRAMES,
    ORIGINAL_EEF_FRAMES,
    center_offset,
    center_position_from_original_pose,
)


ARM_NAMES = tuple(
    f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)
)
TOOL_FRAMES = ORIGINAL_EEF_FRAMES


def waist_preserving_retreat(
    waist_q_rad: list[float] | np.ndarray,
    arm_waypoints: list[list[float]] | np.ndarray,
) -> tuple[list[str], np.ndarray]:
    """Attach one selected waist posture to every lift/pull arm waypoint."""
    return list(WAIST_ARM_JOINT_NAMES), compose_waist_arm(waist_q_rad, arm_waypoints)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--grasp-plan", type=Path, required=True)
    parser.add_argument("--urdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lift-m", type=float, default=0.08)
    parser.add_argument("--nominal-pull-m", type=float, default=0.28)
    parser.add_argument("--lift-steps", type=int, default=20)
    parser.add_argument("--pull-steps", type=int, default=48)
    parser.add_argument("--rack-front-x-b-m", type=float, default=0.415)
    parser.add_argument("--pull-x-weight", type=float, default=150.0)
    parser.add_argument("--pull-yz-weight", type=float, default=500.0)
    parser.add_argument(
        "--preserve-pull-tool-pose",
        action="store_true",
        help="Keep both grasp-time tool rotations fixed throughout the pull.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.lift_steps <= 0 or args.pull_steps < 0:
        raise ValueError("lift steps must be positive and pull steps nonnegative")
    runtime = json.loads((args.snapshot_dir / "runtime.json").read_text())
    grasp_plan = json.loads(args.grasp_plan.read_text())
    plan_joint_names = list(
        grasp_plan.get("joint_names", grasp_plan.get("cspace_joint_names", ARM_JOINT_NAMES))
    )
    q_grasp_full = np.asarray(grasp_plan["terminal_q_rad"], dtype=float)
    if plan_joint_names == WAIST_ARM_JOINT_NAMES:
        waist_q = q_grasp_full[:2]
        q_grasp = q_grasp_full[2:]
    elif plan_joint_names == ARM_JOINT_NAMES:
        waist_q = None
        q_grasp = q_grasp_full
    else:
        raise ValueError("grasp plan has unsupported joint order")
    if q_grasp.shape != (14,) or not np.isfinite(q_grasp).all():
        raise ValueError("grasp plan must contain a finite arm endpoint")
    defaults = dict(zip(runtime["joint_names"], runtime["joint_positions"], strict=True))
    defaults.update(zip(plan_joint_names, q_grasp_full, strict=True))
    inward_normals = np.asarray(
        runtime["pose_editor_state"]["inward_flap_normal_b"], dtype=float
    )
    model = UrdfModel(args.urdf)
    tcp_offsets = tuple(np.asarray(center_offset(side), dtype=float) for side in ("left", "right"))

    def center_fk(arm_index: int, q_dict: dict[str, float]) -> np.ndarray:
        pose = model.fk(TOOL_FRAMES[arm_index], q_dict)
        pose = pose.copy()
        pose[:3, 3] = center_position_from_original_pose(
            pose[:3, 3], pose[:3, :3], tcp_offsets[arm_index]
        )
        return pose

    specs = []
    for arm_index, side in enumerate(("l", "r")):
        names = tuple(f"zarm_{side}{index}_joint" for index in range(1, 8))
        q_start = q_grasp[arm_index * 7 : (arm_index + 1) * 7]
        q_dict = dict(defaults)
        q_dict.update(zip(names, q_start, strict=True))
        start_pose = center_fk(arm_index, q_dict)
        chain = {joint.name: joint for joint in model.chain(TOOL_FRAMES[arm_index])}
        lower = np.asarray([chain[name].lower for name in names]) + 1.0e-8
        upper = np.asarray([chain[name].upper for name in names]) - 1.0e-8
        specs.append((names, q_start, start_pose, lower, upper))

    def fk(arm_index: int, q: np.ndarray) -> np.ndarray:
        q_dict = dict(defaults)
        q_dict.update(zip(specs[arm_index][0], q, strict=True))
        return center_fk(arm_index, q_dict)

    def solve_full_pose(
        arm_index: int,
        q_previous: np.ndarray,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = specs[arm_index][3:5]

        def residual(q):
            pose = fk(arm_index, q)
            rotation_error = Rotation.from_matrix(
                target_rotation.T @ pose[:3, :3]
            ).as_rotvec()
            return np.r_[
                350.0 * (pose[:3, 3] - target_position),
                60.0 * rotation_error,
                0.003 * (q - q_previous),
            ]

        result = least_squares(
            residual,
            np.clip(q_previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=1600,
            ftol=1.0e-11,
            xtol=1.0e-11,
            gtol=1.0e-11,
        )
        return result.x, fk(arm_index, result.x)

    def solve_axis_pose(
        arm_index: int,
        q_previous: np.ndarray,
        target_position: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lower, upper = specs[arm_index][3:5]

        def residual(q):
            pose = fk(arm_index, q)
            closing_axis = pose[:3, :3] @ np.asarray((1.0, 0.0, 0.0))
            position_error = pose[:3, 3] - target_position
            return np.r_[
                args.pull_x_weight * position_error[0],
                args.pull_yz_weight * position_error[1:],
                75.0 * (closing_axis - inward_normals[arm_index]),
                0.015 * (q - q_previous),
            ]

        result = least_squares(
            residual,
            np.clip(q_previous, lower, upper),
            bounds=(lower, upper),
            max_nfev=1800,
            ftol=1.0e-10,
            xtol=1.0e-10,
            gtol=1.0e-10,
        )
        return result.x, fk(arm_index, result.x)

    q_previous = [q_grasp[:7].copy(), q_grasp[7:].copy()]
    lift_rows = [q_grasp.tolist()]
    for step in range(1, args.lift_steps + 1):
        row = []
        for arm_index in range(2):
            start_pose = specs[arm_index][2]
            target_position = start_pose[:3, 3] + np.asarray(
                (0.0, 0.0, args.lift_m * step / args.lift_steps)
            )
            q, _ = solve_full_pose(
                arm_index, q_previous[arm_index], target_position, start_pose[:3, :3]
            )
            q_previous[arm_index] = q
            row.extend(q.tolist())
        lift_rows.append(row)

    lift_poses = [fk(index, q_previous[index]) for index in range(2)]
    initial_separation = lift_poses[0][:3, 3] - lift_poses[1][:3, 3]
    pull_rows = [lift_rows[-1]]
    pull_samples = []
    for step in range(1, args.pull_steps + 1):
        fraction = step / args.pull_steps
        right_target = lift_poses[1][:3, 3] + np.asarray(
            (-args.nominal_pull_m * fraction, 0.0, 0.0)
        )
        if args.preserve_pull_tool_pose:
            right_q, right_pose = solve_full_pose(
                1,
                q_previous[1],
                right_target,
                lift_poses[1][:3, :3],
            )
        else:
            right_q, right_pose = solve_axis_pose(1, q_previous[1], right_target)
        achieved_delta = right_pose[:3, 3] - lift_poses[1][:3, 3]
        left_target = lift_poses[0][:3, 3] + achieved_delta
        left_q, left_pose = solve_full_pose(
            0, q_previous[0], left_target, lift_poses[0][:3, :3]
        )
        q_previous = [left_q, right_q]
        pull_rows.append(left_q.tolist() + right_q.tolist())

        separation = left_pose[:3, 3] - right_pose[:3, 3]
        axis_errors = []
        down_angles = []
        for arm_index, pose in enumerate((left_pose, right_pose)):
            closing_axis = pose[:3, :3] @ np.asarray((1.0, 0.0, 0.0))
            axis_errors.append(
                math.degrees(
                    math.acos(
                        np.clip(closing_axis @ inward_normals[arm_index], -1.0, 1.0)
                    )
                )
            )
            view_axis = pose[:3, :3] @ np.asarray((0.0, 0.0, -1.0))
            down_angles.append(
                math.degrees(math.acos(np.clip(-view_axis[2], -1.0, 1.0)))
            )
        pull_samples.append(
            {
                "step": step,
                "achieved_delta_b_m": achieved_delta.tolist(),
                "separation_error_m": float(
                    np.linalg.norm(separation - initial_separation)
                ),
                "closing_axis_error_deg": axis_errors,
                "tool_down_angle_deg": down_angles,
            }
        )

    if pull_samples:
        final = pull_samples[-1]
    else:
        axis_errors = []
        down_angles = []
        for arm_index, pose in enumerate(lift_poses):
            closing_axis = pose[:3, :3] @ np.asarray((1.0, 0.0, 0.0))
            axis_errors.append(
                math.degrees(
                    math.acos(
                        np.clip(closing_axis @ inward_normals[arm_index], -1.0, 1.0)
                    )
                )
            )
            view_axis = pose[:3, :3] @ np.asarray((0.0, 0.0, -1.0))
            down_angles.append(
                math.degrees(math.acos(np.clip(-view_axis[2], -1.0, 1.0)))
            )
        final = {
            "achieved_delta_b_m": [0.0, 0.0, 0.0],
            "separation_error_m": 0.0,
            "closing_axis_error_deg": axis_errors,
            "tool_down_angle_deg": down_angles,
        }
    arm_rows = np.asarray(lift_rows + pull_rows[1:], dtype=float)
    if waist_q is None:
        output_joint_names = list(ARM_JOINT_NAMES)
        rows = arm_rows
    else:
        output_joint_names, rows = waist_preserving_retreat(waist_q, arm_rows)
    max_joint_step = max(
        max(abs(a - b) for a, b in zip(rows[index - 1], rows[index], strict=True))
        for index in range(1, len(rows))
    )
    box_center_x = float(runtime["pose_editor_state"]["target_box_body_position_b"][0])
    report = {
        "planner": "Cartesian common-motion regional IK without attached-object collision",
        "strategy": (
            "lift_exact_then_pull_to_rack_front_fixed_bimanual_tool_pose"
            if args.preserve_pull_tool_pose
            else "lift_exact_then_pull_to_rack_front_common_translation"
        ),
        "status": "SUCCESS",
        "joint_names": output_joint_names,
        "tool_frames": [CENTER_TOOL_FRAMES[side] for side in ("left", "right")],
        "tcp_frame": CENTER_FRAME_NAME,
        "kinematic_parent_frames": list(TOOL_FRAMES),
        "tcp_offsets_in_parent_m": [offset.tolist() for offset in tcp_offsets],
        "waypoint_q_rad": rows.tolist(),
        "held_waist_q_rad": None if waist_q is None else waist_q.tolist(),
        "waypoint_count": len(rows),
        "lift_waypoint_count": len(lift_rows),
        "pull_waypoint_count": len(pull_rows),
        "lift_achieved_delta_b_m": [0.0, 0.0, args.lift_m],
        "pull_achieved_delta_b_m": final["achieved_delta_b_m"],
        "rack_front_x_b_m": args.rack_front_x_b_m,
        "expected_box_center_final_x_b_m": box_center_x
        + final["achieved_delta_b_m"][0],
        "attached_object_collision_checked": False,
        "max_joint_step_rad": max_joint_step,
        "final_separation_error_m": final["separation_error_m"],
        "terminal_closing_axis_error_deg": final["closing_axis_error_deg"],
        "terminal_tool_down_angle_deg": final["tool_down_angle_deg"],
        "pull_samples": pull_samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key not in {"waypoint_q_rad", "pull_samples"}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
