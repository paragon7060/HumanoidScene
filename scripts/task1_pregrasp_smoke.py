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
        outward = flap_pos[index] - body_pos
        outward = outward / outward.norm().clamp_min(1e-6)
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
    import numpy as np
    import torch
    from kuavo_isaaclab_scene.robots.initial_states import apply_initial_state, load_initial_state
    from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
    from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = _configure_environment(args, configs)
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
        left_quat = robot.data.body_link_quat_w[0, ee_ids[0]].clone()
        right_quat = robot.data.body_link_quat_w[0, ee_ids[1]].clone()
        box = env.scene["medium_box_0"]
        box_start = box.data.root_pose_w[0].clone()
        targets, geometry = _pregrasp_targets(
            env,
            distance_m=args.pregrasp_distance_m,
            grasp_depth_m=args.grasp_depth_m,
        )
        report["target_geometry"] = geometry
        report["start"] = {
            "ee_pose_w": robot.data.body_link_pose_w[0, ee_ids].detach().cpu().tolist(),
            "box_root_pose_w": box_start.detach().cpu().tolist(),
        }
        command = _action_with_targets(env, targets[0], targets[1], left_quat, right_quat)
        for _ in range(args.steps):
            env.step(command)

        # Force one final sensor update before writing evidence.
        env.sim.render()
        final_pos = robot.data.body_link_pos_w[0, ee_ids].clone()
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
            "ee_pose_w": robot.data.body_link_pose_w[0, ee_ids].detach().cpu().tolist(),
            "position_error_m": errors.detach().cpu().tolist(),
            "max_position_error_m": float(errors.max().item()),
            "box_root_pose_w": box_end.detach().cpu().tolist(),
            "box_translation_motion_m": float(box_motion.item()),
            "robot_contact_force_max_n": robot_contact_force_max_n,
        }
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
