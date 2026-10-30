"""Run one fixed-box bimanual IK pregrasp smoke on the Isaac workcell.

This is deliberately a stage-1 runner: it moves the two end-effectors to
geometry-derived pregrasp targets, does not close the grippers, and does not
claim a grasp or extraction success.  The report is an execution artifact for
the first ``MediumBox_0`` validation and is not a production collector yet.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import traceback


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_DIR = PROJECT_DIR / "data_collection" / "configs"
DEFAULT_POSES = PROJECT_DIR / "configs" / "rack_box_poses.json"


def _parse_args(argv=None):
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", type=Path, default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--pregrasp-distance-m", type=float, default=0.10)
    parser.add_argument("--grasp-depth-m", type=float, default=0.015)
    parser.add_argument("--orientation-weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args(argv)
    if not args.headless:
        parser.error("task1_pregrasp_smoke requires --headless")
    if args.device != "cuda:0":
        parser.error("use --device cuda:0 with CUDA_VISIBLE_DEVICES selecting the physical GPU")
    if args.steps <= 0 or args.settle_steps < 0:
        parser.error("steps must be positive and settle-steps must be nonnegative")
    if not math.isfinite(args.pregrasp_distance_m) or args.pregrasp_distance_m <= 0:
        parser.error("pregrasp-distance-m must be finite and positive")
    if not math.isfinite(args.grasp_depth_m) or args.grasp_depth_m <= 0:
        parser.error("grasp-depth-m must be finite and positive")
    if not math.isfinite(args.orientation_weight) or not 0.0 <= args.orientation_weight <= 1.0:
        parser.error("orientation-weight must be finite and in [0, 1]")
    args.enable_cameras = True
    return args


def _load_configs(config_dir: Path) -> dict[str, dict]:
    from kuavo_isaaclab_scene.planning.config import load_yaml

    configs = {
        name: load_yaml(config_dir / f"{name}.yaml")
        for name in ("robot", "task1", "collection")
    }
    task = configs["task1"]
    if task.get("target_box_type") != "medium" or task.get("target_box_id") != "MediumBox_0":
        raise ValueError("This smoke is pinned to task1 target medium/MediumBox_0.")
    collection = configs["collection"]
    for name in ("head", "left_wrist", "right_wrist"):
        camera = collection["cameras"][name]
        if (camera.get("width"), camera.get("height")) != (848, 480):
            raise ValueError(f"{name} must be configured as 848x480 for this smoke.")
    wrist = task.get("wrist_pitch", {})
    if wrist.get("enabled", False):
        full_target = float(wrist.get("full_target_rad", 0.65))
        transit_fraction = float(wrist.get("transit_fraction", 0.33))
        posture_weight = float(wrist.get("posture_weight", 0.60))
        direct_q7_gain = float(wrist.get("direct_q7_gain", 0.0))
        transit_orientation_weight = float(wrist.get("transit_orientation_weight", 0.0))
        prepare_steps = int(wrist.get("prepare_steps", 30))
        rotate_steps = int(wrist.get("rotate_steps", 90))
        if not math.isfinite(full_target) or not 0.0 < full_target < math.radians(40.0):
            raise ValueError("wrist_pitch.full_target_rad must be finite and inside the +40 degree joint stop.")
        if not math.isfinite(transit_fraction) or not 0.0 <= transit_fraction <= 1.0:
            raise ValueError("wrist_pitch.transit_fraction must be in [0, 1].")
        if not math.isfinite(posture_weight) or posture_weight < 0.0:
            raise ValueError("wrist_pitch.posture_weight must be finite and nonnegative.")
        if not math.isfinite(direct_q7_gain) or direct_q7_gain < 0.0:
            raise ValueError("wrist_pitch.direct_q7_gain must be finite and nonnegative.")
        if not math.isfinite(transit_orientation_weight) or not 0.0 <= transit_orientation_weight <= 1.0:
            raise ValueError("wrist_pitch.transit_orientation_weight must be finite and in [0, 1].")
        if prepare_steps < 0 or rotate_steps <= 0:
            raise ValueError("wrist_pitch.prepare_steps must be nonnegative and rotate_steps must be positive.")
    return configs


def _configure_environment(args, configs):
    robot_cfg = configs["robot"]
    os.environ["KUAVO_ROBOT_MODEL"] = str(robot_cfg["robot_model"])
    os.environ["KUAVO_GRIPPER"] = str(robot_cfg["hand_model"])
    os.environ["KUAVO_RACK_BOX_POSES"] = str(DEFAULT_POSES)
    os.environ.pop("KUAVO_IGNORE_RACK_BOX_POSES", None)

    from isaaclab.envs import ManagerBasedRLEnv
    from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization

    cfg = KuavoQuestTeleopEnvCfg()
    cfg.scene.robustness_camera.width = 848
    cfg.scene.robustness_camera.height = 480
    cfg.scene.robustness_camera.data_types = ["rgb"]
    cfg.scene.left_wrist_camera.width = 848
    cfg.scene.left_wrist_camera.height = 480
    cfg.scene.left_wrist_camera.data_types = ["rgb"]
    cfg.scene.right_wrist_camera.width = 848
    cfg.scene.right_wrist_camera.height = 480
    cfg.scene.right_wrist_camera.data_types = ["rgb"]
    # Absolute pose actions let the smoke report the requested 6D target
    # directly.  Orientation is held at the measured pre-command EE pose.
    cfg.actions.left_arm.controller.use_relative_mode = False
    cfg.actions.right_arm.controller.use_relative_mode = False
    set_domain_randomization(cfg, False)
    cfg.seed = args.seed
    return ManagerBasedRLEnv(cfg=cfg)


def _quat_rotate(quat, vector):
    import torch
    from isaaclab.utils.math import quat_apply

    return quat_apply(quat, vector)


def _quat_axis_angle(axis, angle):
    """Build wxyz quaternions for rotations around already-world-frame axes."""
    import torch

    half = angle.unsqueeze(-1) * 0.5
    return torch.cat((torch.cos(half), axis * torch.sin(half)), dim=-1)


def _wrist_pitch_orientations(robot, ee_ids, parent_ids, joint_ids, target_q7):
    """Return world TCP orientations that request an absolute q7 pitch.

    The q7 joint axis is +Y in the zarm_*6 parent frame.  Rotating the current
    TCP around that measured world axis preserves the current mirrored left/right
    hand convention while asking IK to move only the shared wrist-pitch posture.
    """
    import torch
    from isaaclab.utils.math import quat_mul

    current_tcp = robot.data.body_link_quat_w[0, ee_ids]
    parent_quat = robot.data.body_link_quat_w[0, parent_ids]
    current_q7 = robot.data.joint_pos[0, joint_ids]
    # ``Articulation`` exposes the tensor device through its data buffers; use
    # that public path instead of relying on an implementation-specific
    # ``robot.device`` attribute.
    axis_local = torch.zeros((len(parent_ids), 3), device=current_tcp.device, dtype=current_tcp.dtype)
    axis_local[:, 1] = 1.0
    axis_world = _quat_rotate(parent_quat, axis_local)
    axis_world = axis_world / axis_world.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
    delta = _quat_axis_angle(axis_world, target_q7 - current_q7)
    return torch.nn.functional.normalize(quat_mul(delta, current_tcp), dim=-1)


def _set_wrist_posture(terms, robot, target_q7, weight, direct_q7_gain):
    """Set both arm null-space targets while preserving the other current joints."""
    for index, term in enumerate(terms):
        target = robot.data.joint_pos[:, term._joint_ids].clone()
        if target.shape[-1] != 7:
            raise RuntimeError(f"Expected a 7-DoF arm term, got {tuple(target.shape)}")
        target[:, -1] = target_q7[index]
        term.set_posture_target(
            target,
            weight=weight,
            direct_indices=(6,),
            direct_gain=direct_q7_gain,
        )


def _clear_wrist_posture(terms):
    for term in terms:
        term.clear_posture_target()


def _pregrasp_targets(env, *, distance_m: float, grasp_depth_m: float):
    """Return left/right world targets and diagnostic geometry tensors."""
    import torch
    from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry

    box = env.scene["medium_box_0"]
    geometry = box_geometry(env.cfg.scene.medium_box_0, ("flap_right", "flap_left"))
    flap_ids, flap_names = box.find_bodies(("flap_right", "flap_left"), preserve_order=True)
    body_ids, body_names = box.find_bodies("Body")
    if len(flap_ids) != 2 or len(body_ids) != 1:
        raise RuntimeError(
            f"MediumBox_0 body lookup failed: flaps={flap_names}, body={body_names}"
        )
    flap_pos = box.data.body_link_pos_w[0, flap_ids]
    flap_quat = box.data.body_link_quat_w[0, flap_ids]
    body_pos = box.data.body_link_pos_w[0, body_ids[0]]

    targets = []
    diagnostics = []
    for index, flap_name in enumerate(("flap_right", "flap_left")):
        flap = geometry.flaps[flap_name]
        local_grasp = torch.tensor(flap.center, device=env.device, dtype=flap_pos.dtype)
        local_grasp[2] += flap.half_size[2] - grasp_depth_m
        grasp = flap_pos[index] + _quat_rotate(flap_quat[index], local_grasp)
        # The flap's thin local X axis is its grasp-facing normal.  The
        # flap-to-body centre vector is not a surface normal for an opened
        # hinge: it includes the hinge elevation and would incorrectly lift
        # the hand above the box.  Use the current flap pose so a free hinge
        # is handled from its measured state, then orient the sign away from
        # the Body link defensively.
        local_normal = torch.tensor(
            (1.0, 0.0, 0.0) if flap_name == "flap_right" else (-1.0, 0.0, 0.0),
            device=env.device,
            dtype=flap_pos.dtype,
        )
        outward = _quat_rotate(flap_quat[index], local_normal)
        outward = outward / outward.norm().clamp_min(1e-6)
        if torch.dot(outward, flap_pos[index] - body_pos) < 0:
            outward = -outward
        pregrasp = grasp + outward * distance_m
        targets.append(pregrasp)
        diagnostics.append({
            "flap": flap_name,
            "flap_body_position_w": flap_pos[index].detach().cpu().tolist(),
            "grasp_position_w": grasp.detach().cpu().tolist(),
            "outward_normal_w": outward.detach().cpu().tolist(),
            "pregrasp_position_w": pregrasp.detach().cpu().tolist(),
            "geometry_center": list(flap.center),
            "geometry_half_size": list(flap.half_size),
        })
    # Task config maps the left robot hand to flap_right and the right hand to
    # flap_left.  The diagnostics stay in that same order for reproducibility.
    return torch.stack(targets), diagnostics


def _action_with_targets(env, left_target, right_target, left_quat, right_quat):
    import torch

    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    offset = 0
    for name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        if name == "left_arm":
            action[0, offset : offset + dimension] = torch.cat((left_target, left_quat))
        elif name == "right_arm":
            action[0, offset : offset + dimension] = torch.cat((right_target, right_quat))
        offset += dimension
    return action


def _world_targets_to_root_frame(robot, positions_w, orientations_w):
    """Convert world-frame target poses to the root-frame pose expected by IK."""
    from isaaclab.utils.math import subtract_frame_transforms

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    root_pos_w = root_pos_w.expand(positions_w.shape[0], -1)
    root_quat_w = root_quat_w.expand(positions_w.shape[0], -1)
    return subtract_frame_transforms(root_pos_w, root_quat_w, positions_w, orientations_w)


def _pose_command(env, robot, positions_w, orientations_w):
    """Build the absolute pose action after converting world targets to root frame."""
    root_targets, root_orientations = _world_targets_to_root_frame(
        robot, positions_w, orientations_w
    )
    return _action_with_targets(
        env,
        root_targets[0],
        root_targets[1],
        root_orientations[0],
        root_orientations[1],
    )


def _save_rgb(env, name: str, output_dir: Path):
    from PIL import Image

    camera = env.scene[name]
    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    if rgb.dtype.kind == "f":
        if float(rgb.max()) <= 1.01:
            rgb = rgb * 255.0
        rgb = rgb.clip(0, 255).astype("uint8")
    else:
        rgb = rgb.astype("uint8", copy=False)
    path = output_dir / f"{name}.png"
    Image.fromarray(rgb).save(path)
    return {"path": str(path), "shape": list(rgb.shape)}


def run(args, configs):
    import torch
    from kuavo_isaaclab_scene.robots.initial_states import apply_initial_state, load_initial_state
    from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
    from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = _configure_environment(args, configs)
    for arm_name in ("left_arm", "right_arm"):
        env.action_manager.get_term(arm_name).orientation_weight = args.orientation_weight
    task_cfg = configs["task1"]
    robot_cfg = configs["robot"]
    report: dict = {
        "status": "started",
        "task": "task1_box_pick",
        "stage": "pregrasp",
        "target_box_type": task_cfg["target_box_type"],
        "target_box_id": task_cfg["target_box_id"],
        "target_rack_shelf": task_cfg.get("target_rack_shelf"),
        "planner_backend": robot_cfg.get("planner_backend"),
        "requested_pregrasp_distance_m": args.pregrasp_distance_m,
        "requested_grasp_depth_m": args.grasp_depth_m,
        "orientation_weight": args.orientation_weight,
        "steps": args.steps,
        "settle_steps": args.settle_steps,
        "execution_success": "NOT_RUN",
        "grasp_success": "NOT_RUN",
        "extraction_success": "NOT_RUN",
    }
    try:
        env.reset(seed=args.seed)
        state = load_initial_state(
            task_cfg["initial_state"],
            robot_model=resolve_robot_model().name,
            gripper=resolve_gripper_settings().name,
        )
        # This is the named stationary reset preset, not an in-episode pose
        # overwrite.  It is applied before the settle window, then only drive
        # targets are used for the pregrasp motion.
        apply_initial_state(env, None, state, task_cfg["initial_state"])
        env.scene.write_data_to_sim()
        zero = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
        for _ in range(args.settle_steps):
            env.step(zero)

        robot = env.scene["robot"]
        ee_ids, _ = robot.find_bodies(
            (robot_cfg["left_tcp_link"], robot_cfg["right_tcp_link"]),
            preserve_order=True,
        )
        if len(ee_ids) != 2:
            raise RuntimeError(f"TCP body lookup failed: {robot_cfg['left_tcp_link']}, {robot_cfg['right_tcp_link']}")
        parent_ids, parent_names = robot.find_bodies(
            ("zarm_l6_link", "zarm_r6_link"), preserve_order=True
        )
        wrist_joint_ids, wrist_joint_names = robot.find_joints(
            ("zarm_l7_joint", "zarm_r7_joint"), preserve_order=True
        )
        if (
            tuple(parent_names) != ("zarm_l6_link", "zarm_r6_link")
            or tuple(wrist_joint_names) != ("zarm_l7_joint", "zarm_r7_joint")
        ):
            raise RuntimeError(
                f"Unexpected wrist order: parents={parent_names}, joints={wrist_joint_names}"
            )
        if len(parent_ids) != 2 or len(wrist_joint_ids) != 2:
            raise RuntimeError("Could not resolve both zarm_*6 parent links and zarm_*7 wrist joints.")
        arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
        wrist_cfg = task_cfg.get("wrist_pitch", {})
        wrist_enabled = bool(wrist_cfg.get("enabled", False))
        if wrist_enabled:
            full_q7 = float(wrist_cfg.get("full_target_rad", 0.65))
            transit_fraction = float(wrist_cfg.get("transit_fraction", 0.33))
            posture_weight = float(wrist_cfg.get("posture_weight", 0.60))
            direct_q7_gain = float(wrist_cfg.get("direct_q7_gain", 0.0))
            transit_orientation_weight = float(
                wrist_cfg.get("transit_orientation_weight", 0.0)
            )
            prepare_steps = int(wrist_cfg.get("prepare_steps", 30))
            rotate_steps = int(wrist_cfg.get("rotate_steps", 90))
        else:
            full_q7 = transit_fraction = posture_weight = direct_q7_gain = 0.0
            transit_orientation_weight = 0.0
            prepare_steps = rotate_steps = 0
        wrist_limits = robot.data.joint_pos_limits[0, wrist_joint_ids]
        full_below_limit = bool((full_q7 < wrist_limits[:, 0]).any().item())
        full_above_limit = bool((full_q7 > wrist_limits[:, 1]).any().item())
        if wrist_enabled and (full_below_limit or full_above_limit):
            raise RuntimeError(
                f"wrist_pitch.full_target_rad={full_q7} exceeds q7 limits "
                f"{wrist_limits.detach().cpu().tolist()}"
            )
        transit_q7 = torch.full(
            (2,), full_q7 * transit_fraction, device=env.device, dtype=robot.data.joint_pos.dtype
        )
        full_q7_tensor = torch.full(
            (2,), full_q7, device=env.device, dtype=robot.data.joint_pos.dtype
        )
        box = env.scene["medium_box_0"]
        box_start = box.data.root_pose_w[0].clone()
        initial_q7 = robot.data.joint_pos[0, wrist_joint_ids].clone()
        from kuavo_isaaclab_scene.robots.end_effector import get_end_effector_frames

        report["tcp_frame"] = "endeffector_center"
        report["start"] = {
            "ee_pose_w": get_end_effector_frames(robot).center_pose_w[0].detach().cpu().tolist(),
            "box_root_pose_w": box_start.detach().cpu().tolist(),
            "wrist_pitch_q7_rad": initial_q7.detach().cpu().tolist(),
        }
        report["wrist_pitch"] = {
            "enabled": wrist_enabled,
            "transit_fraction": transit_fraction,
            "transit_target_rad": transit_q7.detach().cpu().tolist(),
            "full_target_rad": full_q7_tensor.detach().cpu().tolist(),
            "joint_limits_rad": wrist_limits.detach().cpu().tolist(),
            "posture_weight": posture_weight,
            "direct_q7_gain": direct_q7_gain,
            "transit_orientation_weight": transit_orientation_weight,
            "prepare_steps": prepare_steps,
            "rotate_steps": rotate_steps,
        }

        phase_records = []

        def run_phase(name, positions_w, target_q7, steps, phase_orientation_weight):
            if steps <= 0:
                return
            for term in arm_terms:
                term.orientation_weight = phase_orientation_weight
            if wrist_enabled:
                orientations_w = _wrist_pitch_orientations(
                    robot, ee_ids, parent_ids, wrist_joint_ids, target_q7
                )
                _set_wrist_posture(
                    arm_terms,
                    robot,
                    target_q7,
                    posture_weight,
                    direct_q7_gain,
                )
            else:
                _clear_wrist_posture(arm_terms)
                orientations_w = robot.data.body_link_quat_w[0, ee_ids].clone()
            command = _pose_command(env, robot, positions_w, orientations_w)
            for _ in range(steps):
                env.step(command)
            measured_q7 = robot.data.joint_pos[0, wrist_joint_ids].clone()
            phase_records.append({
                "name": name,
                "steps": steps,
                "target_q7_rad": target_q7.detach().cpu().tolist(),
                "measured_q7_rad": measured_q7.detach().cpu().tolist(),
                "q7_error_rad": (measured_q7 - target_q7).detach().cpu().tolist(),
                "position_error_m": (
                    positions_w - get_end_effector_frames(robot).center_pose_w[0, :, :3]
                ).norm(dim=-1).detach().cpu().tolist(),
            })

        if wrist_enabled:
            # First make both arms share the narrow-gap pitch at the current
            # ready pose.  This avoids entering the rack with one wrist at the
            # captured full angle and the other at a different angle.
            run_phase(
                "wrist_prepare_transit_pitch",
                get_end_effector_frames(robot).center_pose_w[0, :, :3].clone(),
                transit_q7,
                prepare_steps,
                args.orientation_weight,
            )

        targets, geometry = _pregrasp_targets(
            env,
            distance_m=args.pregrasp_distance_m,
            grasp_depth_m=args.grasp_depth_m,
        )
        report["target_geometry"] = geometry
        # Enter the rack gap with the partial pitch, then rotate in place at
        # pregrasp.  The later grasp runner will keep this full orientation for
        # the final flap-normal approach and claw close.
        run_phase(
            "transit_partial_pitch",
            targets,
            transit_q7,
            args.steps,
            transit_orientation_weight,
        )
        if wrist_enabled:
            run_phase(
                "pregrasp_full_pitch_staging",
                targets,
                full_q7_tensor,
                rotate_steps,
                args.orientation_weight,
            )

        # Do not leak the transit-only orientation setting into later callers
        # or the final diagnostics.  The command has already been processed;
        # this restores the configured default for the action terms.
        for term in arm_terms:
            term.orientation_weight = args.orientation_weight

        # Force one final sensor update before writing evidence.
        env.sim.render()
        final_pos = get_end_effector_frames(robot).center_pose_w[0, :, :3].clone()
        errors = (targets - final_pos).norm(dim=-1)
        box_end = box.data.root_pose_w[0].clone()
        box_motion = (box_end[:3] - box_start[:3]).norm()
        try:
            robot_contact_force_max_n = float(
                env.scene["robot_contact"].data.net_forces_w.norm(dim=-1).max().item()
            )
        except KeyError:
            # The teleop scene intentionally does not register the RL-only
            # robot_contact sensor.  Keep the field explicit instead of
            # pretending that an unavailable contact measurement was zero.
            robot_contact_force_max_n = None
        report["end"] = {
            "ee_pose_w": get_end_effector_frames(robot).center_pose_w[0].detach().cpu().tolist(),
            "position_error_m": errors.detach().cpu().tolist(),
            "max_position_error_m": float(errors.max().item()),
            "box_root_pose_w": box_end.detach().cpu().tolist(),
            "box_translation_motion_m": float(box_motion.item()),
            "robot_contact_force_max_n": robot_contact_force_max_n,
            "wrist_pitch_q7_rad": robot.data.joint_pos[0, wrist_joint_ids].detach().cpu().tolist(),
            "orientation_error_rad": [
                float(term.target_orientation_error()[0].item()) for term in arm_terms
            ],
        }
        report["phases"] = phase_records
        report["camera_outputs"] = {
            "robustness_camera": _save_rgb(env, "robustness_camera", output_dir),
            "left_wrist_camera": _save_rgb(env, "left_wrist_camera", output_dir),
            "right_wrist_camera": _save_rgb(env, "right_wrist_camera", output_dir),
        }
        report["status"] = "completed"
        report["execution_success"] = bool(float(errors.max().item()) <= 0.03)
        report["note"] = (
            "Pregrasp position tracking only. Grippers stayed open; no grasp, pull, "
            "extraction, or dataset episode was claimed."
        )
        return report
    finally:
        env.close()


def main(argv=None):
    args = _parse_args(argv)
    configs = _load_configs(args.config_dir.expanduser().resolve())
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    from isaaclab.app import AppLauncher

    app = None
    report = {"status": "failed", "error": "not started"}
    try:
        app = AppLauncher(args).app
        report = run(args, configs)
    except BaseException as exc:
        report.update(error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "pregrasp_smoke.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        if app is not None:
            app.close(wait_for_replicator=False)
    print(json.dumps({"status": report["status"], "report": str(output_dir / "pregrasp_smoke.json")}))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
