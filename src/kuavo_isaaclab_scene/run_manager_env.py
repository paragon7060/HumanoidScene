#!/usr/bin/env python3
"""Launch and smoke-test the Kuavo ManagerBased robustness environment."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from .box_flap_friction import (
    export_flap_friction_environment,
    resolve_flap_friction_settings,
)
from .gripper_config import add_gripper_cli_args, export_gripper_cli, resolve_gripper_settings
from .robot_model import add_robot_model_cli_args, export_robot_model_cli
from .rack_box_layout import (
    format_rack_box_layout,
    resolve_rack_box_layout,
    resolve_rack_box_pose_path,
)
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Run the Kuavo robustness ManagerBasedRLEnv.")
parser.add_argument("--steps", type=int, default=240, help="Number of manager steps to run.")
parser.add_argument("--num-envs", type=int, default=2, help="Parallel randomized environments.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--random-actions",
    action="store_true",
    help="Apply small random upper-body actions instead of holding the nominal pose.",
)
rack_layout_group = parser.add_mutually_exclusive_group()
rack_layout_group.add_argument(
    "--rack-boxes",
    type=str,
    default=None,
    metavar="SPEC",
    help="Rack contents, e.g. '1:small*2,medium;2:large;3:xlarge*2'.",
)
rack_layout_group.add_argument(
    "--rack-box-layout",
    type=Path,
    default=None,
    metavar="JSON",
    help="JSON file defining shelf 1/2/3 box lists or type/count maps.",
)
captured_pose_group = parser.add_mutually_exclusive_group()
captured_pose_group.add_argument(
    "--rack-box-poses",
    type=Path,
    default=None,
    metavar="JSON",
    help="Exact Rack-anchor-relative box poses captured from Isaac Sim.",
)
captured_pose_group.add_argument(
    "--ignore-captured-box-poses",
    action="store_true",
    help="Ignore the default rack_box_poses.json file for this run.",
)
parser.add_argument("--flap-static-friction", type=float, default=None, metavar="VALUE")
parser.add_argument("--flap-dynamic-friction", type=float, default=None, metavar="VALUE")
parser.add_argument(
    "--randomize-flap-friction",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Randomize box flap joint friction at every environment reset (default: enabled).",
)
parser.add_argument(
    "--flap-static-friction-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
)
parser.add_argument(
    "--flap-dynamic-friction-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
)
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)
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
    os.environ.pop("KUAVO_IGNORE_RACK_BOX_POSES", None)
try:
    RACK_BOX_LAYOUT = resolve_rack_box_layout(args_cli.rack_boxes, args_cli.rack_box_layout)
    CAPTURED_RACK_BOX_POSE_PATH = resolve_rack_box_pose_path(
        args_cli.rack_box_poses,
        ignore=args_cli.ignore_captured_box_poses,
    )
    FLAP_FRICTION = resolve_flap_friction_settings(
        static=args_cli.flap_static_friction,
        dynamic=args_cli.flap_dynamic_friction,
        randomize=args_cli.randomize_flap_friction,
        static_range=args_cli.flap_static_friction_range,
        dynamic_range=args_cli.flap_dynamic_friction_range,
        randomize_default=True,
    )
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))
export_flap_friction_environment(FLAP_FRICTION)
if CAPTURED_RACK_BOX_POSE_PATH is not None:
    os.environ["KUAVO_RACK_BOX_POSES"] = str(CAPTURED_RACK_BOX_POSE_PATH)
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from isaaclab.envs import ManagerBasedRLEnv

from .manager_env import KuavoRobustWorkcellEnvCfg
from . import manager_mdp as workcell_mdp
from .camera_viewports import open_camera_viewports


def main() -> None:
    cfg = KuavoRobustWorkcellEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed
    env = ManagerBasedRLEnv(cfg=cfg)
    observations, _ = env.reset(seed=args_cli.seed)

    open_camera_viewports(
        env.scene,
        ["robustness_camera", "waist_camera", "left_wrist_camera", "right_wrist_camera"],
        headless=args_cli.headless,
    )

    print(f"[INFO] ManagerBasedRLEnv initialized with {env.num_envs} environments.")
    print(f"[INFO] Action dimension: {env.action_manager.total_action_dim}.")
    print(
        f"[INFO] Gripper preset: {GRIPPER_SETTINGS.name} "
        f"(active sides: {GRIPPER_SETTINGS.active_sides or 'none'})."
    )
    print(f"[INFO] Observation groups: {list(observations.keys())}.")
    print(f"[INFO] Event modes: {env.event_manager.available_modes}.")
    print(f"[INFO] Rack-box layout: {format_rack_box_layout(RACK_BOX_LAYOUT)}")
    if CAPTURED_RACK_BOX_POSE_PATH is not None:
        print(f"[INFO] Captured rack-box poses: {CAPTURED_RACK_BOX_POSE_PATH}")
    print(
        "[INFO] Flap joint friction: "
        f"base=({FLAP_FRICTION.static}, {FLAP_FRICTION.dynamic}), "
        f"randomize={FLAP_FRICTION.randomize}, "
        f"ranges=({FLAP_FRICTION.static_range}, {FLAP_FRICTION.dynamic_range})."
    )

    spill_terminations = 0
    obstacle_terminations = 0
    other_terminations = 0
    reward_sum = torch.zeros(env.num_envs, device=env.device)
    human_start = env.scene["moving_human"].data.root_pos_w.clone()
    mobile_start = env.scene["moving_robot"].data.root_pos_w.clone()

    for _ in range(args_cli.steps):
        if args_cli.random_actions:
            action = 0.08 * torch.randn_like(env.action_manager.action)
        else:
            action = torch.zeros_like(env.action_manager.action)
        observations, reward, terminated, truncated, extras = env.step(action)
        reward_sum += reward

        if torch.any(terminated):
            spill_terminations += int(
                env.termination_manager.get_term("cargo_spill").sum().item()
            )
            obstacle_terminations += int(
                env.termination_manager.get_term("human_or_robot_contact").sum().item()
            )
            other_terminations += int(torch.sum(terminated).item())
        del truncated, extras

    retained = workcell_mdp.cargo_retained_fraction(env)
    prefill = getattr(env, "_workcell_prefill_count", torch.zeros(env.num_envs, device=env.device))
    human_motion = torch.linalg.norm(
        env.scene["moving_human"].data.root_pos_w - human_start,
        dim=1,
    )
    mobile_motion = torch.linalg.norm(
        env.scene["moving_robot"].data.root_pos_w - mobile_start,
        dim=1,
    )
    print(
        f"[RESULT] steps={args_cli.steps}, mean_reward={reward_sum.mean().item():.3f}, "
        f"prefill_range=[{int(prefill.min())}, {int(prefill.max())}]"
    )
    print(
        f"[RESULT] termination_events={other_terminations}, "
        f"spill_log={spill_terminations}, obstacle_log={obstacle_terminations}, "
        f"retained_obs_mean={retained.float().mean().item():.3f}"
    )
    print(
        f"[RESULT] mover_displacement_mean: human={human_motion.mean().item():.3f} m, "
        f"amr={mobile_motion.mean().item():.3f} m, "
        f"difficulty={float(getattr(env, '_robustness_difficulty', 0.0)):.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
