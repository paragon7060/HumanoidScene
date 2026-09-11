#!/usr/bin/env python3
"""Physically replay a Task1 cuMotion approach, close, and pull retreat.

Unlike the pose editor, this runner never pins the target box.  The arm path is
sent through the articulation position servos, the integrated gripper motors
close under physics, and the target box is free to move on the rack.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_collection.task1.contract import ARM_JOINT_NAMES, WAIST_ARM_JOINT_NAMES


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-plan", type=Path, required=True)
    parser.add_argument("--retreat-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--video-out", type=Path, required=True)
    parser.add_argument(
        "--rack-box-poses",
        type=Path,
        default=PROJECT_DIR / "configs" / "rack_box_poses.json",
        help="Captured rack-relative box poses used to build the physical scene.",
    )
    parser.add_argument(
        "--initial-state",
        default="second_rack_pose",
        help=(
            "Named initial state, or 'meta_default' to keep the native "
            "KuavoQuestTeleopEnvCfg reset pose."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--torso-height-m",
        type=float,
        default=None,
        help="Override only the upright torso lift after applying the initial pose.",
    )
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--approach-steps-per-waypoint", type=int, default=4)
    parser.add_argument("--close-steps", type=int, default=180)
    parser.add_argument("--closed-hold-steps", type=int, default=60)
    parser.add_argument("--retreat-steps-per-waypoint", type=int, default=8)
    parser.add_argument("--final-hold-steps", type=int, default=120)
    parser.add_argument("--capture-stride", type=int, default=4)
    parser.add_argument("--snapshot-position-tolerance-m", type=float, default=0.003)
    parser.add_argument("--snapshot-orientation-tolerance-deg", type=float, default=2.0)
    parser.add_argument("--waypoint-tracking-tolerance-rad", type=float, default=0.01)
    parser.add_argument("--waypoint-max-steps", type=int, default=120)
    parser.add_argument("--tracking-integral-gain", type=float, default=0.08)
    parser.add_argument("--tracking-compensation-limit-rad", type=float, default=0.12)
    parser.add_argument("--approach-box-motion-max-m", type=float, default=0.001)
    parser.add_argument("--retreat-box-motion-min-m", type=float, default=0.03)
    parser.add_argument("--retention-drift-max-m", type=float, default=0.05)
    parser.add_argument("--motor-obstruction-min-rad", type=float, default=0.005)
    parser.add_argument("--approach-only", action="store_true")
    parser.add_argument("--terminal-hold-only", action="store_true")
    parser.add_argument("--direct-arm-replay", action="store_true")
    parser.add_argument(
        "--direct-approach-replay",
        action="store_true",
        help="Replay only the collision-checked approach exactly; keep close and retreat physical.",
    )
    parser.add_argument(
        "--kinematic-direct-approach-render",
        action="store_true",
        help=(
            "Render direct-approach waypoints without advancing physics, then resume "
            "normal physics for close and retreat."
        ),
    )
    parser.add_argument("--overwrite-video", action="store_true")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.headless:
        parser.error("This smoke runner requires --headless.")
    if args.device != "cuda:0":
        parser.error("Use --device cuda:0 with CUDA_VISIBLE_DEVICES selecting the physical GPU.")
    if args.kinematic_direct_approach_render and not args.direct_approach_replay:
        parser.error("--kinematic-direct-approach-render requires --direct-approach-replay.")
    if args.torso_height_m is not None and (
        not math.isfinite(args.torso_height_m) or not 0.0 <= args.torso_height_m <= 0.40
    ):
        parser.error("--torso-height-m must be finite and within [0, 0.40].")
    for name in (
        "settle_steps",
        "approach_steps_per_waypoint",
        "close_steps",
        "closed_hold_steps",
        "retreat_steps_per_waypoint",
        "final_hold_steps",
        "capture_stride",
        "waypoint_max_steps",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "snapshot_position_tolerance_m",
        "snapshot_orientation_tolerance_deg",
        "waypoint_tracking_tolerance_rad",
        "tracking_integral_gain",
        "tracking_compensation_limit_rad",
        "approach_box_motion_max_m",
        "retreat_box_motion_min_m",
        "retention_drift_max_m",
        "motor_obstruction_min_rad",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    args.enable_cameras = True
    return args


def _load_plan(path: Path) -> dict:
    data = json.loads(path.expanduser().resolve().read_text())
    if data.get("status") != "SUCCESS":
        raise ValueError(f"Plan is not successful: {path} status={data.get('status')!r}")
    names = data.get("joint_names")
    waypoints = data.get("waypoint_q_rad")
    if names not in (ARM_JOINT_NAMES, WAIST_ARM_JOINT_NAMES):
        raise ValueError(f"Expected canonical 14DoF or 16DoF joint names in {path}")
    if not isinstance(waypoints, list) or not waypoints:
        raise ValueError(f"Plan has no waypoints: {path}")
    if any(len(row) != len(names) or not all(math.isfinite(float(v)) for v in row) for row in waypoints):
        raise ValueError(f"Invalid waypoint rows in {path}")
    return data


def _snapshot_target_pose_b(plan: dict) -> list[float]:
    """Find the target-box pose stored by the snapshot behind a composed plan."""
    queue = [plan]
    seen = set()
    while queue:
        item = queue.pop()
        snapshot_dir = item.get("snapshot_dir")
        if snapshot_dir:
            runtime_path = Path(snapshot_dir) / "runtime.json"
            if runtime_path not in seen and runtime_path.exists():
                seen.add(runtime_path)
                runtime = json.loads(runtime_path.read_text())
                state = runtime.get("pose_editor_state", {})
                pose = state.get("target_box_body_pose_b")
                if isinstance(pose, list) and len(pose) == 7:
                    return [float(value) for value in pose]
        for segment_path in reversed(item.get("segment_plans", [])):
            segment_path = Path(segment_path)
            if segment_path in seen or not segment_path.exists():
                continue
            seen.add(segment_path)
            queue.append(json.loads(segment_path.read_text()))
    raise ValueError("Approach plan has no snapshot target_box_body_pose_b")


def _quaternion_error_deg(first, second) -> float:
    dot = abs(sum(float(a) * float(b) for a, b in zip(first, second, strict=True)))
    return math.degrees(2.0 * math.acos(min(1.0, max(-1.0, dot))))


def _phase_summary(samples: list[dict], phase: str) -> dict:
    rows = [row for row in samples if row["phase"] == phase]
    if not rows:
        return {"sample_count": 0}
    result = {
        "sample_count": len(rows),
        "max_arm_tracking_error_rad": max(row["arm_tracking_error_max_rad"] for row in rows),
    }
    waist_errors = [row.get("waist_tracking_error_max_rad") for row in rows]
    if all(value is not None for value in waist_errors):
        result["max_waist_tracking_error_rad"] = max(waist_errors)
    return result


args = _parse_args()
os.environ["KUAVO_ROBOT_MODEL"] = "s200062"
os.environ["KUAVO_GRIPPER"] = "s200062_integrated"
os.environ["KUAVO_RACK_BOX_POSES"] = str(args.rack_box_poses.expanduser().resolve())
os.environ.pop("KUAVO_IGNORE_RACK_BOX_POSES", None)

from isaaclab.app import AppLauncher

launcher = AppLauncher(args)
simulation_app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
from kuavo_isaaclab_scene.robots.end_effector import (
    CENTER_FRAME_NAME,
    get_end_effector_frames,
)
from kuavo_isaaclab_scene.robots.initial_states import apply_initial_state, load_initial_state
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.workcell.rack_box_layout import same_shelf_instance_names


ARM_JOINTS = tuple(f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8))
WAIST_ARM_JOINTS = tuple(WAIST_ARM_JOINT_NAMES)
MOTOR_JOINTS = (
    "l_f_bar_1_joint",
    "l_b_bar_1_joint",
    "r_f_bar_1_joint",
    "r_b_bar_1_joint",
)
EEF_BODIES = ("zarm_l7_end_effector", "zarm_r7_end_effector")


def _configure_env() -> ManagerBasedRLEnv:
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.seed = args.seed
    cfg.scene.xr_left_eye_camera = None
    cfg.scene.xr_right_eye_camera = None
    cfg.scene.robustness_camera = None
    cfg.scene.left_wrist_camera = None
    cfg.scene.right_wrist_camera = None
    cfg.scene.waist_camera = None
    cfg.scene.joint_editor_camera.width = 848
    cfg.scene.joint_editor_camera.height = 480
    cfg.scene.joint_editor_camera.data_types = ["rgb"]
    set_domain_randomization(cfg, False)
    return ManagerBasedRLEnv(cfg=cfg)


def _park_same_shelf_boxes(env) -> list[str]:
    from kuavo_isaaclab_scene.envs.manager_env import RACK_BOX_SPAWN_PLAN

    parked = []
    for index, instance_name in enumerate(same_shelf_instance_names(RACK_BOX_SPAWN_PLAN, "MediumBox_0")):
        asset = env.scene[RACK_BOX_SPAWN_PLAN[instance_name].scene_key]
        pose = asset.data.root_pose_w.clone()
        pose[0, :3] = env.scene.env_origins[0] + torch.tensor(
            (5.0, 5.0 + 0.5 * index, 0.02), device=env.device, dtype=pose.dtype
        )
        pose[0, 3:] = torch.tensor((1.0, 0.0, 0.0, 0.0), device=env.device, dtype=pose.dtype)
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device, dtype=pose.dtype))
        parked.append(instance_name)
    return parked


def _body_pose_b(robot, asset, body_id: int):
    root_pos = robot.data.root_pos_w
    root_quat = robot.data.root_quat_w
    pos = asset.data.body_link_pos_w[:, body_id]
    quat = asset.data.body_link_quat_w[:, body_id]
    return subtract_frame_transforms(root_pos, root_quat, pos, quat)


def _eef_positions_b(robot, eef_ids):
    if len(eef_ids) != 2:
        raise ValueError("Expected both original EEF bodies for calibrated TCP lookup")
    return get_end_effector_frames(robot).center_pose_b[0, :, :3]


def _main() -> None:
    print("[GRASP_PULL] main_start", flush=True)
    approach = _load_plan(args.approach_plan)
    retreat = _load_plan(args.retreat_plan)
    snapshot_target_pose_b = _snapshot_target_pose_b(approach)
    plan_joint_names = tuple(approach["joint_names"])
    if plan_joint_names not in (ARM_JOINTS, WAIST_ARM_JOINTS):
        raise ValueError("Approach plan has an unsupported joint order.")
    if tuple(retreat["joint_names"]) != plan_joint_names:
        raise ValueError("Approach and retreat plans must use the same canonical joint order.")
    waist_count = 2 if plan_joint_names == WAIST_ARM_JOINTS else 0
    approach_last = torch.tensor(approach["waypoint_q_rad"][-1])
    retreat_first = torch.tensor(retreat["waypoint_q_rad"][0])
    junction_error = float(torch.max(torch.abs(approach_last - retreat_first)).item())
    if junction_error > 1.0e-4:
        raise ValueError(f"Approach/retreat joint discontinuity is {junction_error:.6g} rad")

    env = _configure_env()
    print("[GRASP_PULL] env_created", flush=True)
    samples: list[dict] = []
    frame_count = 0
    global_step = 0
    frame_dir = args.video_out.expanduser().resolve().parent / "frames"
    try:
        env.reset(seed=args.seed)
        print("[GRASP_PULL] env_reset", flush=True)
        if args.initial_state == "meta_default":
            print("[GRASP_PULL] initial_state_applied=meta_default_reset", flush=True)
        else:
            state = load_initial_state(
                args.initial_state,
                robot_model=resolve_robot_model().name,
                gripper=resolve_gripper_settings().name,
            )
            apply_initial_state(env, None, state, args.initial_state)
            print(f"[GRASP_PULL] initial_state_applied={args.initial_state}", flush=True)
        if args.torso_height_m is not None:
            from kuavo_isaaclab_scene.teleop.teleop_body import TeleopBodyMapper

            mapper = TeleopBodyMapper(
                PROJECT_DIR
                / "src/kuavo_isaaclab_scene/assets/kuavo_s200062/urdf/biped_s200062.urdf"
            )
            if not mapper.set_height(args.torso_height_m):
                raise ValueError(f"Unreachable torso height: {args.torso_height_m:.3f} m")
            torso_state = {
                "robot_model": resolve_robot_model().name,
                "gripper": resolve_gripper_settings().name,
                "coordinate_frame": "env-origin",
                "assets": {
                    "robot": {
                        "joint_positions": dict(
                            zip(
                                ("knee_joint", "leg_joint", "waist_pitch_joint"),
                                map(float, mapper.joints[:3]),
                                strict=True,
                            )
                        )
                    }
                },
            }
            apply_initial_state(env, None, torso_state, f"torso_{args.torso_height_m:.3f}m")
            print(f"[GRASP_PULL] torso_height_m={args.torso_height_m:.3f}", flush=True)
        parked = _park_same_shelf_boxes(env)
        print(f"[GRASP_PULL] parked={parked}", flush=True)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.step_dt)
        print("[GRASP_PULL] scene_forwarded", flush=True)

        robot = env.scene["robot"]
        box = env.scene["medium_box_0"]
        camera = env.scene["joint_editor_camera"]
        arm_ids, arm_names = robot.find_joints(plan_joint_names, preserve_order=True)
        motor_ids, motor_names = robot.find_joints(MOTOR_JOINTS, preserve_order=True)
        eef_ids, eef_names = robot.find_bodies(EEF_BODIES, preserve_order=True)
        box_body_ids, box_body_names = box.find_bodies("Body")
        if tuple(arm_names) != plan_joint_names or tuple(motor_names) != MOTOR_JOINTS:
            raise RuntimeError(f"Joint lookup mismatch: arms={arm_names}, motors={motor_names}")
        if tuple(eef_names) != EEF_BODIES or len(box_body_ids) != 1:
            raise RuntimeError(f"Body lookup mismatch: eef={eef_names}, box={box_body_names}")
        box_body_id = box_body_ids[0]
        print("[GRASP_PULL] joints_and_bodies_resolved", flush=True)

        root_pos = robot.data.root_pos_w[0]
        root_quat = robot.data.root_quat_w[0]
        dtype = root_pos.dtype
        eye = root_pos + quat_apply(root_quat, torch.tensor((2.0, 1.8, 1.65), device=env.device, dtype=dtype))
        look = root_pos + quat_apply(root_quat, torch.tensor((0.20, 0.0, 1.15), device=env.device, dtype=dtype))
        camera.set_world_poses_from_view(eye.unsqueeze(0), look.unsqueeze(0))
        print("[GRASP_PULL] camera_positioned", flush=True)

        arm_target = robot.data.joint_pos[:, arm_ids].clone()
        arm_reference = arm_target.clone()
        tracking_compensation = torch.zeros_like(arm_target)
        motor_target = robot.data.joint_pos[:, motor_ids].clone()
        open_motor_target = motor_target.clone()
        closed_motor_target = torch.zeros_like(motor_target)
        if frame_dir.exists():
            if not args.overwrite_video:
                raise FileExistsError(f"Frame directory exists: {frame_dir}")
            import shutil

            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)
        print(f"[GRASP_PULL] frame_directory={frame_dir}", flush=True)

        def step_once(
            phase: str,
            *,
            capture: bool = True,
            advance_physics: bool = True,
        ) -> None:
            nonlocal global_step, frame_count
            robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(motor_target, joint_ids=motor_ids)
            env.scene.write_data_to_sim()
            render = capture and global_step % args.capture_stride == 0
            if advance_physics:
                env.sim.step(render=render)
            else:
                env.sim.forward()
                if render:
                    env.sim.render()
            env.scene.update(env.physics_dt)
            if render:
                frame = camera.data.output["rgb"][0, ..., :3].detach().cpu()
                if frame.dtype != torch.uint8:
                    frame = frame.clamp(0, 255).to(torch.uint8)
                from PIL import Image

                Image.fromarray(frame.contiguous().numpy()).save(
                    frame_dir / f"{frame_count:05d}.jpg", quality=90, subsampling=0
                )
                frame_count += 1
            actual = robot.data.joint_pos[0, arm_ids]
            tracking_error = torch.abs(actual - arm_reference[0])
            box_pos_b, _ = _body_pose_b(robot, box, box_body_id)
            eef_pos_b = _eef_positions_b(robot, eef_ids)
            samples.append(
                {
                    "step": global_step,
                    "phase": phase,
                    "arm_tracking_error_max_rad": float(
                        torch.max(tracking_error[waist_count:]).item()
                    ),
                    "waist_tracking_error_max_rad": (
                        float(torch.max(tracking_error[:waist_count]).item())
                        if waist_count
                        else None
                    ),
                    "box_body_position_b_m": box_pos_b[0].detach().cpu().tolist(),
                    "eef_positions_b_m": eef_pos_b.detach().cpu().tolist(),
                    "gripper_motor_position_rad": robot.data.joint_pos[0, motor_ids].detach().cpu().tolist(),
                }
            )
            global_step += 1

        for _ in range(args.settle_steps):
            step_once("settle")
        settled_box_pos_b, settled_box_quat_b = _body_pose_b(robot, box, box_body_id)
        snapshot_position_error_m = float(
            torch.linalg.vector_norm(
                settled_box_pos_b[0]
                - torch.tensor(
                    snapshot_target_pose_b[:3],
                    device=env.device,
                    dtype=settled_box_pos_b.dtype,
                )
            ).item()
        )
        snapshot_orientation_error_deg = _quaternion_error_deg(
            settled_box_quat_b[0].detach().cpu().tolist(),
            snapshot_target_pose_b[3:],
        )
        if (
            snapshot_position_error_m > args.snapshot_position_tolerance_m
            or snapshot_orientation_error_deg > args.snapshot_orientation_tolerance_deg
        ):
            raise RuntimeError(
                "settled target does not match the collision snapshot: "
                f"position_error_m={snapshot_position_error_m:.6f}, "
                f"orientation_error_deg={snapshot_orientation_error_deg:.3f}"
            )

        if args.terminal_hold_only:
            terminal = torch.tensor(
                approach["waypoint_q_rad"][-1],
                device=env.device,
                dtype=arm_target.dtype,
            ).unsqueeze(0)
            arm_reference[:] = terminal
            arm_target[:] = terminal
            tracking_compensation.zero_()
            robot.write_joint_state_to_sim(
                terminal,
                torch.zeros_like(terminal),
                joint_ids=arm_ids,
            )
            env.scene.write_data_to_sim()
            env.sim.forward()
            env.scene.update(env.step_dt)
            terminal_start_box_pos_b, _ = _body_pose_b(robot, box, box_body_id)
            for _ in range(args.final_hold_steps):
                step_once("terminal_hold")
            terminal_box_pos_b, terminal_box_quat_b = _body_pose_b(
                robot, box, box_body_id
            )
            terminal_box_motion = float(
                torch.linalg.vector_norm(
                    terminal_box_pos_b - terminal_start_box_pos_b
                ).item()
            )
            contact_free = bool(
                terminal_box_motion <= args.approach_box_motion_max_m
            )
            report = {
                "passed": contact_free,
                "terminal_hold_only": True,
                "terminal_contact_free": contact_free,
                "acceptance": {
                    "terminal_box_motion_m_max": args.approach_box_motion_max_m,
                },
                "approach_plan": str(args.approach_plan.expanduser().resolve()),
                "tcp_frame": CENTER_FRAME_NAME,
                "frame_directory": str(frame_dir),
                "frame_count": frame_count,
                "snapshot_position_error_m": snapshot_position_error_m,
                "snapshot_orientation_error_deg": snapshot_orientation_error_deg,
                "terminal_box_motion_m": terminal_box_motion,
                "terminal_box_pose_b": torch.cat(
                    (terminal_box_pos_b[0], terminal_box_quat_b[0])
                ).cpu().tolist(),
                "phase_summaries": {
                    "terminal_hold": _phase_summary(samples, "terminal_hold")
                },
                "samples": samples,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            prefix = "[GRASP_PULL]" if contact_free else "[GRASP_PULL_ERROR]"
            print(
                prefix,
                json.dumps({key: value for key, value in report.items() if key != "samples"}),
                flush=True,
            )
            return

        def track_waypoint(waypoint, phase: str, minimum_steps: int) -> None:
            arm_reference[:] = torch.tensor(
                waypoint, device=env.device, dtype=arm_reference.dtype
            )
            if args.direct_arm_replay or (
                args.direct_approach_replay and phase == "approach"
            ):
                arm_target[:] = arm_reference
                tracking_compensation.zero_()
                robot.write_joint_state_to_sim(
                    arm_reference,
                    torch.zeros_like(arm_reference),
                    joint_ids=arm_ids,
                )
                for _ in range(minimum_steps):
                    step_once(
                        phase,
                        advance_physics=not (
                            phase == "approach"
                            and args.kinematic_direct_approach_render
                        ),
                    )
                return
            for tracking_step in range(args.waypoint_max_steps):
                actual = robot.data.joint_pos[:, arm_ids]
                reference_error = arm_reference - actual
                tracking_compensation.add_(
                    args.tracking_integral_gain * reference_error
                ).clamp_(
                    -args.tracking_compensation_limit_rad,
                    args.tracking_compensation_limit_rad,
                )
                arm_target[:] = arm_reference + tracking_compensation
                step_once(phase)
                actual = robot.data.joint_pos[0, arm_ids]
                error = float(torch.max(torch.abs(actual - arm_reference[0])).item())
                if tracking_step + 1 >= minimum_steps and error <= args.waypoint_tracking_tolerance_rad:
                    return
            print(
                f"[GRASP_PULL_ERROR] {phase} waypoint tracking failed: "
                f"max_error_rad={error:.6f}",
                flush=True,
            )
            raise RuntimeError(
                f"{phase} waypoint tracking failed: max_error_rad={error:.6f}"
            )

        for waypoint in approach["waypoint_q_rad"]:
            track_waypoint(waypoint, "approach", args.approach_steps_per_waypoint)
        grasp_box_pos_b, grasp_box_quat_b = _body_pose_b(robot, box, box_body_id)
        approach_box_motion = float(
            torch.linalg.vector_norm(grasp_box_pos_b - settled_box_pos_b).item()
        )
        approach_contact_free = bool(
            approach_box_motion <= args.approach_box_motion_max_m
        )
        if args.approach_only or not approach_contact_free:
            report = {
                "passed": approach_contact_free,
                "approach_only": True,
                "approach_contact_free": approach_contact_free,
                "acceptance": {
                    "approach_box_motion_m_max": args.approach_box_motion_max_m,
                },
                "approach_plan": str(args.approach_plan.expanduser().resolve()),
                "tcp_frame": CENTER_FRAME_NAME,
                "video_path": str(args.video_out.expanduser().resolve()),
                "frame_directory": str(frame_dir),
                "initial_state": args.initial_state,
                "torso_height_m": args.torso_height_m,
                "parked_same_shelf_boxes": parked,
                "junction_max_abs_rad": junction_error,
                "physics_dt_s": env.physics_dt,
                "frame_count": frame_count,
                "settled_box_pose_b": torch.cat(
                    (settled_box_pos_b[0], settled_box_quat_b[0])
                ).cpu().tolist(),
                "snapshot_target_box_pose_b": snapshot_target_pose_b,
                "snapshot_position_error_m": snapshot_position_error_m,
                "snapshot_orientation_error_deg": snapshot_orientation_error_deg,
                "waypoint_tracking_tolerance_rad": args.waypoint_tracking_tolerance_rad,
                "arm_replay_mode": (
                    "direct_all"
                    if args.direct_arm_replay
                    else "kinematic_direct_approach_then_position_servo"
                    if args.kinematic_direct_approach_render
                    else "direct_approach_then_position_servo"
                    if args.direct_approach_replay
                    else "position_servo"
                ),
                "grasp_box_pose_b": torch.cat(
                    (grasp_box_pos_b[0], grasp_box_quat_b[0])
                ).cpu().tolist(),
                "approach_box_motion_m": approach_box_motion,
                "phase_summaries": {
                    phase: _phase_summary(samples, phase)
                    for phase in ("settle", "approach")
                },
                "samples": samples,
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            prefix = "[GRASP_PULL]" if approach_contact_free else "[GRASP_PULL_ERROR]"
            print(
                prefix,
                json.dumps({key: value for key, value in report.items() if key != "samples"}),
                flush=True,
            )
            return

        for close_step in range(args.close_steps):
            alpha = (close_step + 1) / args.close_steps
            motor_target[:] = open_motor_target + alpha * (closed_motor_target - open_motor_target)
            step_once("close")
        for _ in range(args.closed_hold_steps):
            step_once("closed_hold")
        closed_box_pos_b, closed_box_quat_b = _body_pose_b(robot, box, box_body_id)
        closed_eef_pos_b = _eef_positions_b(robot, eef_ids)
        closed_motor_position = robot.data.joint_pos[0, motor_ids].detach().cpu().tolist()

        for waypoint in retreat["waypoint_q_rad"]:
            track_waypoint(waypoint, "retreat", args.retreat_steps_per_waypoint)
        for _ in range(args.final_hold_steps):
            step_once("final_hold")

        final_box_pos_b, final_box_quat_b = _body_pose_b(robot, box, box_body_id)
        final_eef_pos_b = _eef_positions_b(robot, eef_ids)
        closed_mid = closed_eef_pos_b.mean(dim=0)
        final_mid = final_eef_pos_b.mean(dim=0)
        closed_relative = closed_box_pos_b[0] - closed_mid
        final_relative = final_box_pos_b[0] - final_mid
        retention_drift = float(torch.linalg.vector_norm(final_relative - closed_relative).item())
        pull_robotward = float((closed_box_pos_b[0, 0] - final_box_pos_b[0, 0]).item())
        total_translation = float(torch.linalg.vector_norm(final_box_pos_b - settled_box_pos_b).item())
        final_z_drop = float((closed_box_pos_b[0, 2] - final_box_pos_b[0, 2]).item())
        retreat_rows = [row for row in samples if row["phase"] == "retreat"]
        final_hold_rows = [row for row in samples if row["phase"] == "final_hold"]
        retreat_box_motion = float(
            torch.linalg.vector_norm(
                torch.tensor(retreat_rows[-1]["box_body_position_b_m"])
                - torch.tensor(retreat_rows[0]["box_body_position_b_m"])
            ).item()
        )
        final_hold_box_motion = float(
            torch.linalg.vector_norm(
                torch.tensor(final_hold_rows[-1]["box_body_position_b_m"])
                - torch.tensor(final_hold_rows[0]["box_body_position_b_m"])
            ).item()
        )
        hand_motor_obstruction_rad = [
            sum(abs(value) for value in closed_motor_position[:2]),
            sum(abs(value) for value in closed_motor_position[2:]),
        ]
        bilateral_obstruction = bool(
            min(hand_motor_obstruction_rad) >= args.motor_obstruction_min_rad
        )
        object_followed = bool(
            retreat_box_motion >= args.retreat_box_motion_min_m
            and final_hold_box_motion <= 0.01
        )
        stable_bimanual = bool(
            object_followed
            and bilateral_obstruction
            and retention_drift <= args.retention_drift_max_m
        )
        report = {
            "passed": bool(stable_bimanual and approach_contact_free),
            "approach_contact_free": approach_contact_free,
            "object_followed_to_retreat": object_followed,
            "stable_bimanual_retention": stable_bimanual,
            "acceptance": {
                "retreat_box_motion_m_min": args.retreat_box_motion_min_m,
                "final_hold_box_motion_m_max": 0.01,
                "approach_box_motion_m_max": args.approach_box_motion_max_m,
                "stable_bimanual_retention_drift_m_max": args.retention_drift_max_m,
                "motor_obstruction_rad_min": args.motor_obstruction_min_rad,
            },
            "approach_plan": str(args.approach_plan.expanduser().resolve()),
            "plan_joint_names": list(plan_joint_names),
            "tcp_frame": CENTER_FRAME_NAME,
            "retreat_plan": str(args.retreat_plan.expanduser().resolve()),
            "video_path": str(args.video_out.expanduser().resolve()),
            "frame_directory": str(frame_dir),
            "initial_state": args.initial_state,
            "torso_height_m": args.torso_height_m,
            "parked_same_shelf_boxes": parked,
            "junction_max_abs_rad": junction_error,
            "physics_dt_s": env.physics_dt,
            "frame_count": frame_count,
            "settled_box_pose_b": torch.cat((settled_box_pos_b[0], settled_box_quat_b[0])).cpu().tolist(),
            "snapshot_target_box_pose_b": snapshot_target_pose_b,
            "snapshot_position_error_m": snapshot_position_error_m,
            "snapshot_orientation_error_deg": snapshot_orientation_error_deg,
            "waypoint_tracking_tolerance_rad": args.waypoint_tracking_tolerance_rad,
            "arm_replay_mode": (
                "direct_all"
                if args.direct_arm_replay
                else "kinematic_direct_approach_then_position_servo"
                if args.kinematic_direct_approach_render
                else "direct_approach_then_position_servo"
                if args.direct_approach_replay
                else "position_servo"
            ),
            "grasp_box_pose_b": torch.cat((grasp_box_pos_b[0], grasp_box_quat_b[0])).cpu().tolist(),
            "closed_box_pose_b": torch.cat((closed_box_pos_b[0], closed_box_quat_b[0])).cpu().tolist(),
            "final_box_pose_b": torch.cat((final_box_pos_b[0], final_box_quat_b[0])).cpu().tolist(),
            "robotward_pull_m": pull_robotward,
            "approach_box_motion_m": approach_box_motion,
            "retreat_box_motion_m": retreat_box_motion,
            "final_hold_box_motion_m": final_hold_box_motion,
            "total_box_translation_from_settle_m": total_translation,
            "final_z_drop_m": final_z_drop,
            "hand_box_retention_drift_m": retention_drift,
            "bilateral_motor_obstruction": bilateral_obstruction,
            "hand_motor_obstruction_rad": hand_motor_obstruction_rad,
            "closed_gripper_motor_position_rad": closed_motor_position,
            "phase_summaries": {
                phase: _phase_summary(samples, phase)
                for phase in ("settle", "approach", "close", "closed_hold", "retreat", "final_hold")
            },
            "final_gripper_motor_position_rad": robot.data.joint_pos[0, motor_ids].detach().cpu().tolist(),
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print("[GRASP_PULL]", json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
        if not report["passed"]:
            print("[GRASP_PULL] Physical acceptance failed; keeping video/report for diagnosis.", flush=True)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    _main()
