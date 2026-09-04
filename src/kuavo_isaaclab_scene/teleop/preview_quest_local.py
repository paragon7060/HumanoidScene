#!/usr/bin/env python3
"""Preview the exact Quest teleop scene locally without OpenXR or a headset."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher
from ..robots.gripper_config import add_gripper_cli_args, export_gripper_cli, resolve_gripper_settings
from ..robots.robot_model import add_robot_model_cli_args, export_robot_model_cli


parser = argparse.ArgumentParser(description="Preview Kuavo Quest cameras locally without Meta Quest.")
parser.add_argument("--steps", type=int, default=0, help="Simulation steps; 0 runs until the window closes.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--head-sweep",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Automatically sweep Kuavo head yaw/pitch to verify the head camera follows the joints.",
)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--head-camera-height", type=int, default=360)
parser.add_argument("--wrist-camera-width", type=int, default=240)
parser.add_argument("--wrist-camera-height", type=int, default=180)
parser.add_argument(
    "--domain-randomization",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable the same robustness randomization available during teleoperation.",
)
parser.add_argument("--rack-boxes", type=str, default=None, metavar="SPEC")
parser.add_argument("--rack-box-layout", type=Path, default=None, metavar="JSON")
parser.add_argument("--rack-box-poses", type=Path, default=None, metavar="JSON")
parser.add_argument("--ignore-captured-box-poses", action="store_true")
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)
try:
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))

if args_cli.steps < 0:
    parser.error("--steps must be 0 or greater.")
if min(
    args_cli.head_camera_width,
    args_cli.head_camera_height,
    args_cli.wrist_camera_width,
    args_cli.wrist_camera_height,
) <= 0:
    parser.error("Camera width/height values must be positive.")
if args_cli.headless:
    parser.error("Local Quest preview requires a GUI; do not pass --headless.")
if args_cli.rack_boxes is not None and args_cli.rack_box_layout is not None:
    parser.error("Use only one of --rack-boxes and --rack-box-layout.")
if args_cli.rack_boxes is not None:
    os.environ["KUAVO_RACK_BOXES"] = args_cli.rack_boxes
    os.environ.pop("KUAVO_RACK_BOX_LAYOUT", None)
elif args_cli.rack_box_layout is not None:
    os.environ["KUAVO_RACK_BOX_LAYOUT"] = str(args_cli.rack_box_layout.expanduser().resolve())
    os.environ.pop("KUAVO_RACK_BOXES", None)
if args_cli.ignore_captured_box_poses:
    os.environ["KUAVO_IGNORE_RACK_BOX_POSES"] = "1"
    os.environ.pop("KUAVO_RACK_BOX_POSES", None)
elif args_cli.rack_box_poses is not None:
    os.environ["KUAVO_RACK_BOX_POSES"] = str(args_cli.rack_box_poses.expanduser().resolve())

# RTX cameras are required, but XR is intentionally left disabled. This makes
# the preview independent of CloudXR Runtime, OpenXR and Meta Quest hardware.
args_cli.enable_cameras = True
args_cli.xr = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from isaaclab.envs import ManagerBasedRLEnv

from ..display.camera_viewports import open_camera_viewports
from ..envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization


def main() -> None:
    cfg = KuavoQuestTeleopEnvCfg()
    # Desktop preview uses the main viewport and physical robot cameras only.
    cfg.scene.xr_left_eye_camera = None
    cfg.scene.xr_right_eye_camera = None
    cfg.seed = args_cli.seed
    cfg.scene.robustness_camera.width = args_cli.head_camera_width
    cfg.scene.robustness_camera.height = args_cli.head_camera_height
    cfg.scene.left_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.left_wrist_camera.height = args_cli.wrist_camera_height
    cfg.scene.right_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.right_wrist_camera.height = args_cli.wrist_camera_height
    set_domain_randomization(cfg, args_cli.domain_randomization)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=args_cli.seed)
    open_camera_viewports(
        env.scene,
        ["robustness_camera", "left_wrist_camera", "right_wrist_camera"],
        headless=False,
        width=320,
        height=220,
        columns=3,
    )

    print("[INFO] Local Quest visual preview is ready; no OpenXR device was created.")
    print(f"[INFO] Gripper preset: {GRIPPER_SETTINGS.name}.")
    print("[INFO] Main window: workcell scene; small windows: head, left wrist, right wrist cameras.")
    print("[LIMIT] This does not test CloudXR streaming, HMD tracking, hand tracking, or the XR-only overlay.")
    if args_cli.head_sweep:
        print("[INFO] Automatic Kuavo head yaw/pitch sweep is enabled.")

    step = 0
    try:
        while simulation_app.is_running() and (args_cli.steps == 0 or step < args_cli.steps):
            action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
            if args_cli.head_sweep:
                sim_time = step * float(env.step_dt)
                action[0, 12] = 0.55 * math.sin(0.45 * sim_time)
                action[0, 13] = 0.22 * math.sin(0.63 * sim_time)
            env.step(action)
            step += 1
    except KeyboardInterrupt:
        print("\n[INFO] Local preview interrupted by user.")
    finally:
        env.close()
        print(f"[RESULT] Local preview closed after {step} simulation step(s).")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
