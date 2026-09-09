#!/usr/bin/env python3
"""Plan a collision-aware sequential Task1 bimanual pregrasp with cuMotion."""

from __future__ import annotations

import argparse
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path

import numpy as np
import yaml


ROBOT_COLLISION_FRAMES = {
    "waist_yaw_link",
    "zhead_1_link",
    "zarm_l2_link",
    "zarm_l4_link",
    "zarm_l7_link",
    "l_twofinger_base",
    "l_f_finger",
    "l_b_finger",
    "zarm_r2_link",
    "zarm_r4_link",
    "zarm_r7_link",
    "r_twofinger_base",
    "r_f_finger",
    "r_b_finger",
}

SELF_COLLISION_IGNORE = {
    "waist_yaw_link": ["zhead_1_link", "zarm_l2_link", "zarm_r2_link"],
    "zarm_l2_link": ["zarm_l4_link"],
    "zarm_l4_link": ["zarm_l7_link"],
    "zarm_l7_link": ["l_twofinger_base", "l_f_finger", "l_b_finger"],
    "l_twofinger_base": ["l_f_finger", "l_b_finger"],
    "l_f_finger": ["l_b_finger"],
    "zarm_r2_link": ["zarm_r4_link"],
    "zarm_r4_link": ["zarm_r7_link"],
    "zarm_r7_link": ["r_twofinger_base", "r_f_finger", "r_b_finger"],
    "r_twofinger_base": ["r_f_finger", "r_b_finger"],
    "r_f_finger": ["r_b_finger"],
}


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


def robot_spheres(snapshot, runtime, cell_m: float, radius_inflation_m: float) -> dict:
    body_poses = {
        name: pose_matrix(pose)
        for name, pose in zip(runtime["body_names"], runtime["body_poses_w"], strict=True)
    }
    spheres: dict[str, list[dict]] = {}
    for collider in snapshot["colliders"]:
        if not collider["robot"]:
            continue
        owner = (collider["owner"] or "").rsplit("/", 1)[-1]
        if owner not in ROBOT_COLLISION_FRAMES:
            continue
        local_from_world = inverse_transform(body_poses[owner])
        entries = spheres.setdefault(owner, [])
        for center_w, radius in cover_cuboid(collider["pose_w"], collider["dims"], cell_m):
            entries.append(
                {
                    "center": (local_from_world @ np.r_[center_w, 1])[:3].tolist(),
                    "radius": radius + radius_inflation_m,
                }
            )
    missing = ROBOT_COLLISION_FRAMES - spheres.keys()
    if missing:
        raise ValueError(f"missing live robot collision frames: {sorted(missing)}")
    return spheres


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


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--snapshot-dir", type=Path, required=True)
    result.add_argument("--urdf", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, default=None)
    result.add_argument("--sphere-cell-m", type=float, default=0.06)
    result.add_argument("--collision-margin-m", type=float, default=0.005)
    result.add_argument("--validation-samples", type=int, default=101)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if not math.isfinite(args.sphere_cell_m) or args.sphere_cell_m <= 0:
        raise ValueError("--sphere-cell-m must be finite and positive")
    if not math.isfinite(args.collision_margin_m) or args.collision_margin_m < 0:
        raise ValueError("--collision-margin-m must be finite and nonnegative")
    if args.validation_samples < 2:
        raise ValueError("--validation-samples must be at least two")
    output = args.output_dir or Path("/home/seonho/outputs/HumanoidScene") / (
        "cumotion_collision_pregrasp_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
    urdf_text = args.urdf.expanduser().resolve().read_text()

    world_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.collision_margin_m
    )
    self_spheres = robot_spheres(
        snapshot, runtime, args.sphere_cell_m, args.collision_margin_m / 2
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

    defaults = {
        name: float(value)
        for name, value in zip(
            runtime["joint_names"], runtime["joint_positions"], strict=True
        )
        if name.startswith(("zarm_", "l_", "r_"))
    }
    editor_joints = {
        item["name"]: float(item["value"])
        for item in runtime["pose_editor_state"]["joints"]
    }
    defaults.update(editor_joints)
    targets = runtime["pose_editor_state"]["pregrasp_position_b"]
    report = {
        "planner": "NVIDIA cuMotion 1.1.0 TrajectoryOptimizer",
        "strategy": "sequential_bimanual_noncontact_approach",
        "snapshot_dir": str(snapshot_dir),
        "initial_pose_source": "origin/main KuavoQuestTeleopEnvCfg",
        "orientation_constraint": "none",
        "collision_model": {
            "world_obstacles": len(world_config["cuboid"]),
            "robot_world_spheres": sum(map(len, world_spheres.values())),
            "robot_self_spheres": sum(map(len, self_spheres.values())),
            "sphere_cover_cell_m": args.sphere_cell_m,
            "world_margin_m": args.collision_margin_m,
            "self_pair_margin_m": args.collision_margin_m,
            "excluded_stage_colliders": len(snapshot["excluded_enabled_colliders"]),
        },
        "arms": {},
    }

    left_terminal = None
    for side, letter, target_index in (("left", "l", 0), ("right", "r", 1)):
        arm_defaults = dict(defaults)
        if left_terminal is not None:
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
        optimizer = cumotion.create_trajectory_optimizer(
            cumotion.create_default_trajectory_optimizer_config(robot, tool, world_view)
        )
        target_position = np.asarray(targets[target_index], dtype=float)
        target = cumotion.TrajectoryOptimizer.TaskSpaceTarget(
            cumotion.TrajectoryOptimizer.TranslationConstraint.target(target_position),
            cumotion.TrajectoryOptimizer.OrientationConstraint.none(),
        )
        result = optimizer.plan_to_task_space_target(q_initial, target)
        status = str(result.status()).split(".")[-1]
        arm_report = {
            "status": status,
            "joint_names": names,
            "target_position_b_m": target_position.tolist(),
            "fixed_other_arm": "main_initial" if side == "left" else "left_terminal",
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
        world_collisions = [inspector.in_collision_with_obstacle(q) for q in samples]
        self_collisions = [inspector.in_self_collision(q) for q in samples]
        distances = [inspector.min_distance_to_obstacle(q) for q in samples]
        arm_report.update(
            {
                "duration_s": float(domain.span()),
                "terminal_q_rad": terminal.tolist(),
                "terminal_tcp_position_b_m": terminal_position.tolist(),
                "terminal_error_m": float(np.linalg.norm(terminal_position - target_position)),
                "sample_times_s": sample_times.tolist(),
                "sample_q_rad": samples.tolist(),
                "sampled_world_collision": any(world_collisions),
                "sampled_self_collision": any(self_collisions),
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
        and not arm["sampled_world_collision"]
        and not arm["sampled_self_collision"]
        for arm in report["arms"].values()
    )
    return 0 if success else 2


if __name__ == "__main__":
    raise SystemExit(main())
