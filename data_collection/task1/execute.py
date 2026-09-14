#!/usr/bin/env python3
"""Physically replay a Task1 cuMotion approach, close, and pull retreat.

Unlike the pose editor, this runner never pins the target box.  The arm path is
sent through the articulation position servos, the integrated gripper motors
close under physics, and the target box is free to move on the rack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback


PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from data_collection.task1.contract import ARM_JOINT_NAMES, WAIST_ARM_JOINT_NAMES
from data_collection.task1.execution_contract import (
    box_extraction_metrics,
    closed_motor_targets,
    evenly_spaced_capture_indices,
    executor_target_keys,
    motor_obstruction_gates,
    paired_box_acceptance,
    paired_retention_metrics,
    physical_acceptance,
    resolved_kinematic_capture_frame_count,
    retention_reference,
)
from data_collection.task1.video import encode_jpeg_sequence


def _parse_args():
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approach-plan", type=Path, required=True)
    parser.add_argument("--retreat-plan", type=Path, required=True)
    parser.add_argument("--scenario-path", type=Path)
    parser.add_argument("--paired-boxes", nargs=2)
    parser.add_argument(
        "--clear-same-shelf-boxes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
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
    parser.add_argument("--pair-separation-drift-max-m", type=float, default=0.01)
    parser.add_argument("--motor-obstruction-min-rad", type=float, default=0.005)
    parser.add_argument("--box-size-m", type=float, nargs=3, default=(0.32, 0.22, 0.185))
    parser.add_argument("--rack-front-x-b-m", type=float, default=0.415)
    parser.add_argument("--partial-extraction-front-progress-min-m", type=float, default=0.05)
    parser.add_argument("--partial-extraction-front-inside-max-m", type=float, default=0.05)
    parser.add_argument("--final-hold-box-motion-max-m", type=float, default=0.01)
    parser.add_argument(
        "--active-gripper",
        choices=("both", "left", "right"),
        default="both",
        help="Close and evaluate both hands, or only the selected single hand.",
    )
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
    parser.add_argument(
        "--kinematic-approach-render-frames",
        type=int,
        default=None,
        help=(
            "Capture exactly this many evenly spaced approach waypoints while still "
            "replaying every collision-checked waypoint. Kinematic approach only; "
            "defaults to 73 frames (normal-speed verified replay)."
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
    if args.kinematic_approach_render_frames is not None:
        if not args.kinematic_direct_approach_render:
            parser.error(
                "--kinematic-approach-render-frames requires "
                "--kinematic-direct-approach-render."
            )
        if args.kinematic_approach_render_frames < 2:
            parser.error("--kinematic-approach-render-frames must be at least 2.")
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
        "pair_separation_drift_max_m",
        "motor_obstruction_min_rad",
        "partial_extraction_front_progress_min_m",
        "partial_extraction_front_inside_max_m",
        "final_hold_box_motion_max_m",
    ):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be finite and positive.")
    if not math.isfinite(args.rack_front_x_b_m):
        parser.error("--rack-front-x-b-m must be finite.")
    if any(not math.isfinite(value) or value <= 0.0 for value in args.box_size_m):
        parser.error("--box-size-m values must be finite and positive.")
    args.enable_cameras = True
    if args.paired_boxes is not None:
        executor_target_keys(args.paired_boxes, args.clear_same_shelf_boxes)
        if args.scenario_path is None:
            parser.error("--scenario-path is required with --paired-boxes.")
        if args.active_gripper == "both":
            parser.error("--paired-boxes requires one active gripper.")
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _snapshot_target_poses_b(plan: dict, target_keys: tuple[str, ...]) -> dict[str, list[float]]:
    """Find exact target poses for a legacy single box or a captured pair."""
    queue = [plan]
    seen = set()
    while queue:
        item = queue.pop()
        snapshot_dir = item.get("snapshot_dir")
        if snapshot_dir:
            runtime_path = Path(snapshot_dir) / "runtime.json"
            if runtime_path not in seen and runtime_path.exists():
                seen.add(runtime_path)
                state = json.loads(runtime_path.read_text()).get("pose_editor_state", {})
                paired = state.get("paired_box_body_poses_b")
                if isinstance(paired, dict) and set(paired) == set(target_keys):
                    return {
                        key: [float(value) for value in paired[key]] for key in target_keys
                    }
                pose = state.get("target_box_body_pose_b")
                if len(target_keys) == 1 and isinstance(pose, list) and len(pose) == 7:
                    return {target_keys[0]: [float(value) for value in pose]}
        for segment_path in reversed(item.get("segment_plans", [])):
            segment_path = Path(segment_path)
            if segment_path in seen or not segment_path.exists():
                continue
            seen.add(segment_path)
            queue.append(json.loads(segment_path.read_text()))
    raise ValueError("Approach plan has no matching snapshot target box poses")


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
    target_keys = executor_target_keys(args.paired_boxes, args.clear_same_shelf_boxes)
    paired_mode = len(target_keys) == 2
    if paired_mode:
        from data_collection.task1.scenario import load_scenario_config

        scenario = load_scenario_config(args.scenario_path)
        if tuple(scenario["scene"]["paired_boxes"]) != target_keys:
            raise ValueError("executor paired boxes do not match scenario")
        if scenario["execution"]["active_gripper"] != args.active_gripper:
            raise ValueError("executor active gripper does not match scenario")
    else:
        scenario = None
    snapshot_target_poses_b = _snapshot_target_poses_b(approach, target_keys)
    snapshot_target_pose_b = snapshot_target_poses_b[target_keys[0]]
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_DIR, text=True
    ).strip()
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
    phase_frame_counts: dict[str, int] = {}
    kinematic_approach_render_frames_effective = None
    global_step = 0
    frame_dir = args.video_out.expanduser().resolve().parent / "frames"
    report_written = False
    write_report = None
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
        parked = _park_same_shelf_boxes(env) if args.clear_same_shelf_boxes else []
        print(f"[GRASP_PULL] parked={parked}", flush=True)
        env.scene.write_data_to_sim()
        env.sim.forward()
        env.scene.update(env.step_dt)
        print("[GRASP_PULL] scene_forwarded", flush=True)

        robot = env.scene["robot"]
        boxes = {key: env.scene[key] for key in target_keys}
        box = boxes[target_keys[0]]
        camera = env.scene["joint_editor_camera"]
        arm_ids, arm_names = robot.find_joints(plan_joint_names, preserve_order=True)
        motor_ids, motor_names = robot.find_joints(MOTOR_JOINTS, preserve_order=True)
        eef_ids, eef_names = robot.find_bodies(EEF_BODIES, preserve_order=True)
        target_body_ids = {}
        target_body_names = {}
        for key, target_box in boxes.items():
            body_ids, body_names = target_box.find_bodies("Body")
            if len(body_ids) != 1:
                raise RuntimeError(f"Body lookup mismatch for {key}: {body_names}")
            target_body_ids[key] = body_ids[0]
            target_body_names[key] = body_names
        if tuple(arm_names) != plan_joint_names or tuple(motor_names) != MOTOR_JOINTS:
            raise RuntimeError(f"Joint lookup mismatch: arms={arm_names}, motors={motor_names}")
        if tuple(eef_names) != EEF_BODIES:
            raise RuntimeError(f"Body lookup mismatch: eef={eef_names}")
        box_body_id = target_body_ids[target_keys[0]]
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
        closed_motor_target = torch.tensor(
            closed_motor_targets(
                open_motor_target[0].detach().cpu().tolist(),
                motor_names,
                args.active_gripper,
            ),
            device=env.device,
            dtype=motor_target.dtype,
        ).unsqueeze(0)
        if frame_dir.exists():
            if not args.overwrite_video:
                raise FileExistsError(f"Frame directory exists: {frame_dir}")
            import shutil

            shutil.rmtree(frame_dir)
        frame_dir.mkdir(parents=True)
        print(f"[GRASP_PULL] frame_directory={frame_dir}", flush=True)

        def write_report(report: dict) -> None:
            nonlocal report_written
            report["phase_frame_counts"] = dict(phase_frame_counts)
            report["kinematic_approach_render_frames_requested"] = (
                args.kinematic_approach_render_frames
            )
            report["kinematic_approach_render_frames_effective"] = (
                kinematic_approach_render_frames_effective
            )
            report["video_path"] = str(args.video_out.expanduser().resolve())
            report["video_encoding"] = encode_jpeg_sequence(
                frame_dir,
                args.video_out,
                fps=1.0 / (env.physics_dt * args.capture_stride),
                overwrite=args.overwrite_video,
            )
            report["source_commit"] = source_commit
            report["input_sha256"] = {
                "approach_plan": _sha256_file(args.approach_plan.expanduser().resolve()),
                "retreat_plan": _sha256_file(args.retreat_plan.expanduser().resolve()),
                "scenario": (
                    _sha256_file(args.scenario_path.expanduser().resolve())
                    if args.scenario_path is not None
                    else None
                ),
            }
            report["video_sha256"] = _sha256_file(args.video_out.expanduser().resolve())
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            report_written = True

        def step_once(
            phase: str,
            *,
            capture: bool = True,
            advance_physics: bool = True,
            force_capture: bool = False,
        ) -> None:
            nonlocal global_step, frame_count
            robot.set_joint_position_target(arm_target, joint_ids=arm_ids)
            robot.set_joint_position_target(motor_target, joint_ids=motor_ids)
            env.scene.write_data_to_sim()
            render = capture and (
                force_capture or global_step % args.capture_stride == 0
            )
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
                phase_frame_counts[phase] = phase_frame_counts.get(phase, 0) + 1
            actual = robot.data.joint_pos[0, arm_ids]
            tracking_error = torch.abs(actual - arm_reference[0])
            box_positions_b = {
                key: _body_pose_b(robot, boxes[key], target_body_ids[key])[0][0]
                .detach()
                .cpu()
                .tolist()
                for key in target_keys
            }
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
                    "box_body_position_b_m": box_positions_b[target_keys[0]],
                    "box_body_positions_b_m": box_positions_b,
                    "eef_positions_b_m": eef_pos_b.detach().cpu().tolist(),
                    "gripper_motor_position_rad": robot.data.joint_pos[0, motor_ids].detach().cpu().tolist(),
                }
            )
            global_step += 1

        for _ in range(args.settle_steps):
            step_once("settle")
        print("[GRASP_PULL] settle_complete", flush=True)
        settled_box_poses = {}
        snapshot_position_errors_m = {}
        snapshot_orientation_errors_deg = {}
        for key in target_keys:
            position, quaternion = _body_pose_b(robot, boxes[key], target_body_ids[key])
            settled_box_poses[key] = (position, quaternion)
            target_pose = snapshot_target_poses_b[key]
            snapshot_position_errors_m[key] = float(
                torch.linalg.vector_norm(
                    position[0]
                    - torch.tensor(target_pose[:3], device=env.device, dtype=position.dtype)
                ).item()
            )
            snapshot_orientation_errors_deg[key] = _quaternion_error_deg(
                quaternion[0].detach().cpu().tolist(), target_pose[3:]
            )
        settled_box_pos_b, settled_box_quat_b = settled_box_poses[target_keys[0]]
        snapshot_position_error_m = max(snapshot_position_errors_m.values())
        snapshot_orientation_error_deg = max(snapshot_orientation_errors_deg.values())
        print(
            "[GRASP_PULL] snapshot_gate "
            f"position_error_m={snapshot_position_error_m:.6f} "
            f"orientation_error_deg={snapshot_orientation_error_deg:.3f}",
            flush=True,
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
            write_report(report)
            prefix = "[GRASP_PULL]" if contact_free else "[GRASP_PULL_ERROR]"
            print(
                prefix,
                json.dumps({key: value for key, value in report.items() if key != "samples"}),
                flush=True,
            )
            return

        def track_waypoint(
            waypoint,
            phase: str,
            minimum_steps: int,
            *,
            direct_capture_waypoint: bool | None = None,
        ) -> None:
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
                for direct_step in range(minimum_steps):
                    scheduled_capture = (
                        True
                        if direct_capture_waypoint is None
                        else direct_capture_waypoint and direct_step == 0
                    )
                    step_once(
                        phase,
                        capture=scheduled_capture,
                        advance_physics=not (
                            phase == "approach"
                            and args.kinematic_direct_approach_render
                        ),
                        force_capture=(
                            direct_capture_waypoint is True and direct_step == 0
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

        approach_capture_indices = None
        if args.kinematic_direct_approach_render:
            kinematic_approach_render_frames_effective = (
                resolved_kinematic_capture_frame_count(
                    len(approach["waypoint_q_rad"]),
                    args.kinematic_approach_render_frames,
                )
            )
            approach_capture_indices = set(
                evenly_spaced_capture_indices(
                    len(approach["waypoint_q_rad"]),
                    kinematic_approach_render_frames_effective,
                )
            )
            print(
                "[GRASP_PULL] kinematic_approach_render_schedule "
                f"waypoints={len(approach['waypoint_q_rad'])} "
                f"frames={len(approach_capture_indices)}",
                flush=True,
            )
        for approach_index, waypoint in enumerate(approach["waypoint_q_rad"]):
            track_waypoint(
                waypoint,
                "approach",
                args.approach_steps_per_waypoint,
                direct_capture_waypoint=(
                    None
                    if approach_capture_indices is None
                    else approach_index in approach_capture_indices
                ),
            )
        grasp_box_poses = {
            key: _body_pose_b(robot, boxes[key], target_body_ids[key]) for key in target_keys
        }
        approach_box_motion_by_key = {
            key: float(
                torch.linalg.vector_norm(
                    grasp_box_poses[key][0] - settled_box_poses[key][0]
                ).item()
            )
            for key in target_keys
        }
        grasp_box_pos_b, grasp_box_quat_b = grasp_box_poses[target_keys[0]]
        approach_box_motion = max(approach_box_motion_by_key.values())
        approach_contact_free = bool(
            all(
                motion <= args.approach_box_motion_max_m
                for motion in approach_box_motion_by_key.values()
            )
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
                "snapshot_target_box_poses_b": snapshot_target_poses_b,
                "snapshot_position_errors_m": snapshot_position_errors_m,
                "snapshot_orientation_errors_deg": snapshot_orientation_errors_deg,
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
                "approach_box_motion_by_key_m": approach_box_motion_by_key,
                "phase_summaries": {
                    phase: _phase_summary(samples, phase)
                    for phase in ("settle", "approach")
                },
                "samples": samples,
            }
            write_report(report)
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
        closed_box_poses = {
            key: _body_pose_b(robot, boxes[key], target_body_ids[key]) for key in target_keys
        }
        closed_box_pos_b, closed_box_quat_b = closed_box_poses[target_keys[0]]
        closed_eef_pos_b = _eef_positions_b(robot, eef_ids)
        closed_motor_position = robot.data.joint_pos[0, motor_ids].detach().cpu().tolist()

        for waypoint in retreat["waypoint_q_rad"]:
            track_waypoint(waypoint, "retreat", args.retreat_steps_per_waypoint)
        for _ in range(args.final_hold_steps):
            step_once("final_hold")

        final_box_poses = {
            key: _body_pose_b(robot, boxes[key], target_body_ids[key]) for key in target_keys
        }
        final_box_pos_b, final_box_quat_b = final_box_poses[target_keys[0]]
        final_eef_pos_b = _eef_positions_b(robot, eef_ids)
        closed_reference = torch.tensor(
            retention_reference(
                closed_eef_pos_b.detach().cpu().tolist(), args.active_gripper
            ),
            device=env.device,
            dtype=closed_box_pos_b.dtype,
        )
        final_reference = torch.tensor(
            retention_reference(
                final_eef_pos_b.detach().cpu().tolist(), args.active_gripper
            ),
            device=env.device,
            dtype=final_box_pos_b.dtype,
        )
        closed_relative = closed_box_pos_b[0] - closed_reference
        final_relative = final_box_pos_b[0] - final_reference
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
        active_motor_obstruction, bilateral_obstruction = motor_obstruction_gates(
            hand_motor_obstruction_rad,
            args.active_gripper,
            args.motor_obstruction_min_rad,
        )
        if paired_mode:
            closed_positions = {
                key: closed_box_poses[key][0][0].detach().cpu().tolist()
                for key in target_keys
            }
            pair_metrics = paired_retention_metrics(
                samples, target_keys, args.active_gripper, closed_positions
            )
            tracking_error_max = max(
                row["arm_tracking_error_max_rad"]
                for row in samples
                if row["phase"] in ("approach", "retreat")
            )
            tracking_passed = bool(
                tracking_error_max <= args.waypoint_tracking_tolerance_rad
            )
            passed, acceptance_mode = paired_box_acceptance(
                pair_metrics,
                approach_box_motion_m=approach_box_motion_by_key,
                approach_box_motion_max_m=args.approach_box_motion_max_m,
                progress_min_m=args.partial_extraction_front_progress_min_m,
                pair_separation_drift_max_m=args.pair_separation_drift_max_m,
                hand_pair_center_drift_max_m=args.retention_drift_max_m,
                final_hold_box_motion_max_m=args.final_hold_box_motion_max_m,
                active_motor_obstruction=active_motor_obstruction,
                tracking_passed=tracking_passed,
            )
            box_extractions = {}
            closed_pose_by_key = {}
            final_pose_by_key = {}
            for key in target_keys:
                closed_pose_by_key[key] = torch.cat(
                    (closed_box_poses[key][0][0], closed_box_poses[key][1][0])
                ).cpu().tolist()
                final_pose_by_key[key] = torch.cat(
                    (final_box_poses[key][0][0], final_box_poses[key][1][0])
                ).cpu().tolist()
                box_extractions[key] = box_extraction_metrics(
                    closed_pose_by_key[key],
                    final_pose_by_key[key],
                    box_size_m=args.box_size_m,
                    rack_front_x_b_m=args.rack_front_x_b_m,
                    front_progress_min_m=args.partial_extraction_front_progress_min_m,
                    front_inside_max_m=args.partial_extraction_front_inside_max_m,
                    final_hold_box_motion_m=pair_metrics["final_hold_box_motion_m"][key],
                    final_hold_box_motion_max_m=args.final_hold_box_motion_max_m,
                )
            report = {
                "passed": passed,
                "acceptance_mode": acceptance_mode,
                "active_gripper": args.active_gripper,
                "paired_boxes": list(target_keys),
                "approach_contact_free": approach_contact_free,
                "tracking_passed": tracking_passed,
                "tracking_error_max_rad": tracking_error_max,
                "active_motor_obstruction": active_motor_obstruction,
                "bilateral_motor_obstruction": bilateral_obstruction,
                "hand_motor_obstruction_rad": hand_motor_obstruction_rad,
                **pair_metrics,
                "box_extraction_metrics": box_extractions,
                "acceptance": {
                    "box_robotward_progress_m_min": args.partial_extraction_front_progress_min_m,
                    "pair_separation_drift_m_max": args.pair_separation_drift_max_m,
                    "hand_pair_center_drift_m_max": args.retention_drift_max_m,
                    "final_hold_box_motion_m_max": args.final_hold_box_motion_max_m,
                    "approach_box_motion_m_max": args.approach_box_motion_max_m,
                    "motor_obstruction_rad_min": args.motor_obstruction_min_rad,
                    "waypoint_tracking_error_rad_max": args.waypoint_tracking_tolerance_rad,
                },
                "scenario_path": str(args.scenario_path.expanduser().resolve()),
                "approach_plan": str(args.approach_plan.expanduser().resolve()),
                "retreat_plan": str(args.retreat_plan.expanduser().resolve()),
                "plan_joint_names": list(plan_joint_names),
                "tcp_frame": CENTER_FRAME_NAME,
                "frame_directory": str(frame_dir),
                "initial_state": args.initial_state,
                "torso_height_m": args.torso_height_m,
                "parked_same_shelf_boxes": parked,
                "junction_max_abs_rad": junction_error,
                "physics_dt_s": env.physics_dt,
                "frame_count": frame_count,
                "snapshot_target_box_poses_b": snapshot_target_poses_b,
                "snapshot_position_errors_m": snapshot_position_errors_m,
                "snapshot_orientation_errors_deg": snapshot_orientation_errors_deg,
                "approach_box_motion_by_key_m": approach_box_motion_by_key,
                "closed_box_poses_b": closed_pose_by_key,
                "final_box_poses_b": final_pose_by_key,
                "closed_gripper_motor_position_rad": closed_motor_position,
                "final_gripper_motor_position_rad": robot.data.joint_pos[0, motor_ids]
                .detach()
                .cpu()
                .tolist(),
                "phase_summaries": {
                    phase: _phase_summary(samples, phase)
                    for phase in (
                        "settle",
                        "approach",
                        "close",
                        "closed_hold",
                        "retreat",
                        "final_hold",
                    )
                },
                "samples": samples,
            }
            write_report(report)
            print(
                "[GRASP_PULL]",
                json.dumps({key: value for key, value in report.items() if key != "samples"}),
                flush=True,
            )
            if not passed:
                print(
                    "[GRASP_PULL] Paired-box physical acceptance failed; keeping video/report for diagnosis.",
                    flush=True,
                )
            return
        object_followed = bool(
            retreat_box_motion >= args.retreat_box_motion_min_m
            and final_hold_box_motion <= args.final_hold_box_motion_max_m
        )
        stable_retention = bool(
            object_followed
            and active_motor_obstruction
            and retention_drift <= args.retention_drift_max_m
        )
        stable_bimanual = bool(stable_retention and args.active_gripper == "both")
        closed_box_pose = torch.cat((closed_box_pos_b[0], closed_box_quat_b[0])).cpu().tolist()
        final_box_pose = torch.cat((final_box_pos_b[0], final_box_quat_b[0])).cpu().tolist()
        extraction = box_extraction_metrics(
            closed_box_pose,
            final_box_pose,
            box_size_m=args.box_size_m,
            rack_front_x_b_m=args.rack_front_x_b_m,
            front_progress_min_m=args.partial_extraction_front_progress_min_m,
            front_inside_max_m=args.partial_extraction_front_inside_max_m,
            final_hold_box_motion_m=final_hold_box_motion,
            final_hold_box_motion_max_m=args.final_hold_box_motion_max_m,
        )
        passed, acceptance_mode = physical_acceptance(
            args.active_gripper,
            approach_contact_free=approach_contact_free,
            partial_extraction_success=extraction["partial_extraction_success"],
            stable_retention=stable_retention,
        )
        report = {
            "passed": passed,
            "acceptance_mode": acceptance_mode,
            "active_gripper": args.active_gripper,
            "approach_contact_free": approach_contact_free,
            "object_followed_to_retreat": object_followed,
            "stable_retention": stable_retention,
            "stable_bimanual_retention": stable_bimanual,
            **extraction,
            "acceptance": {
                "retreat_box_motion_m_min": args.retreat_box_motion_min_m,
                "final_hold_box_motion_m_max": args.final_hold_box_motion_max_m,
                "approach_box_motion_m_max": args.approach_box_motion_max_m,
                "stable_bimanual_retention_drift_m_max": args.retention_drift_max_m,
                "motor_obstruction_rad_min": args.motor_obstruction_min_rad,
                "box_size_m": list(args.box_size_m),
                "rack_front_x_b_m": args.rack_front_x_b_m,
                "partial_extraction_front_progress_m_min": args.partial_extraction_front_progress_min_m,
                "partial_extraction_front_inside_rack_m_max": args.partial_extraction_front_inside_max_m,
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
            "closed_box_pose_b": closed_box_pose,
            "final_box_pose_b": final_box_pose,
            "robotward_pull_m": pull_robotward,
            "approach_box_motion_m": approach_box_motion,
            "retreat_box_motion_m": retreat_box_motion,
            "final_hold_box_motion_m": final_hold_box_motion,
            "total_box_translation_from_settle_m": total_translation,
            "final_z_drop_m": final_z_drop,
            "hand_box_retention_drift_m": retention_drift,
            "active_motor_obstruction": active_motor_obstruction,
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
        write_report(report)
        print("[GRASP_PULL]", json.dumps({key: value for key, value in report.items() if key != "samples"}), flush=True)
        if not report["passed"]:
            print("[GRASP_PULL] Physical acceptance failed; keeping video/report for diagnosis.", flush=True)
    except BaseException as error:
        traceback.print_exc()
        if write_report is not None and not report_written and frame_count:
            try:
                write_report(
                    {
                        "passed": False,
                        "acceptance_mode": (
                            "paired_single_hand_partial_extraction"
                            if paired_mode
                            else "runtime_failure"
                        ),
                        "runtime_failure": type(error).__name__,
                        "runtime_failure_message": str(error),
                        "active_gripper": args.active_gripper,
                        "paired_boxes": list(target_keys) if paired_mode else None,
                        "frame_directory": str(frame_dir),
                        "frame_count": frame_count,
                        "samples": samples,
                    }
                )
            except BaseException:
                traceback.print_exc()
        raise
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    _main()
