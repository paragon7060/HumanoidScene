#!/usr/bin/env python3
"""Verify measured gripper feedback through the real manager and policy bridge.

Run using the Isaac Lab conda Python, e.g.:
  python scripts/verify_gripper_io.py --headless --robot-model s200062 \
    --gripper s200062_integrated --output artifacts/eval/s200062_gripper_io.json
No checkpoint is needed. USD joint ownership and camera mounts are unchanged.
"""

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from isaaclab.app import AppLauncher
from kuavo_isaaclab_scene.robots.gripper_config import (
    add_gripper_cli_args, export_gripper_cli, resolve_gripper_settings,
)
from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args, export_robot_model_cli

parser = argparse.ArgumentParser(description=__doc__)
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps-per-pose", type=int, default=30)
parser.add_argument("--tolerance", type=float, default=0.1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps_per_pose < 1 or not 0.0 < args.tolerance < 1.0:
    parser.error("Use positive steps-per-pose and tolerance between 0 and 1.")
if args.output.exists():
    parser.error(f"Refusing to overwrite {args.output}; choose a new output path.")
export_robot_model_cli(args)
export_gripper_cli(args)
settings = resolve_gripper_settings()
if settings.active_sides != ("left", "right"):
    parser.error("This diagnostic requires two configured hands.")
args.enable_cameras = True
launcher = AppLauncher(args)
app = launcher.app

import torch
from isaaclab.envs import ManagerBasedRLEnv
from kuavo_isaaclab_scene.envs.manager_env import KuavoRobustWorkcellEnvCfg
from kuavo_isaaclab_scene.evaluation.groot_lerobot_bridge import KuavoLeRobotBridge
from kuavo_isaaclab_scene.robots.gripper_runtime import build_gripper_action_cfg


def main():
    cfg = KuavoRobustWorkcellEnvCfg()
    cfg.scene.num_envs = 1
    cfg.curriculum = None
    cfg.decimation = 12
    cfg.sim.render_interval = 12
    cfg.episode_length_s = max(60.0, 8 * args.steps_per_pose * cfg.sim.dt * cfg.decimation)
    for name, term in vars(cfg.events).items():
        if name != "reset_all" and hasattr(term, "mode"):
            setattr(cfg.events, name, None)
    cfg.actions.left_gripper = build_gripper_action_cfg(settings, "left", continuous=True)
    cfg.actions.right_gripper = build_gripper_action_cfg(settings, "right", continuous=True)
    env = ManagerBasedRLEnv(cfg=cfg)
    rows = []
    try:
        bridge = KuavoLeRobotBridge(
            env, gripper_settings=settings, policy_profile="kuavo-arm-claw",
            state_mode="joint_position", action_mode="joint_position", action_clip=None,
        )
        generic = KuavoLeRobotBridge(env, gripper_settings=settings)
        for cycle in range(2):
            env.reset()
            target = bridge.state()
            for left, right in ((0.0, 1.0), (1.0, 0.0), (0.5, 0.5)):
                target[:, 7], target[:, 15] = left, right
                for _ in range(args.steps_per_pose):
                    _, _, terminated, truncated, _ = env.step(bridge.action(target).action)
                    if bool(terminated.any() or truncated.any()):
                        raise RuntimeError("Unexpected episode reset during hand measurement.")
                measured = bridge.state()
                generic_state = generic.state()
                assert measured.shape == (1, 16)
                assert generic_state.shape[-1] == len(generic.state_names)
                assert len(generic.action_names) == env.action_manager.total_action_dim == 17
                raw_hands = []
                offset = 15
                for view in generic.grippers.values():
                    state = view.joint_state(relative=True)
                    assert torch.equal(generic_state[:, offset:offset + state.shape[-1]], state)
                    offset += state.shape[-1]
                    raw_hands.append(view.joint_state()[0].tolist())
                actual = measured[0, [7, 15]]
                error = (actual - torch.tensor([left, right], device=env.device)).abs().max().item()
                row = {"cycle": cycle, "target_claw": [left, right],
                       "measured_claw": actual.tolist(), "max_claw_error": error,
                       "measured_hand_joints": raw_hands}
                rows.append(row)
                print(f"[GRIPPER_IO] {row}", flush=True)
        report = {
            "robot_model": args.robot_model, "gripper_preset": settings.name,
            "default_state_dim": len(generic.state_names),
            "default_state_names": list(generic.state_names),
            "policy_state_dim": 16, "policy_action_dim": bridge.action_dim,
            "manager_action_names": list(bridge.manager_action_names),
            "gripper_views": {side: view.metadata() for side, view in bridge.grippers.items()},
            "tolerance": args.tolerance, "rows": rows,
            "passed": all(row["max_claw_error"] <= args.tolerance for row in rows),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if not report["passed"]:
            raise RuntimeError(f"Claw tracking exceeded {args.tolerance}; inspect {args.output}.")
        print(f"[GRIPPER_IO] PASS: {args.output}", flush=True)
    finally:
        env.close()


exit_code = 0
try:
    with torch.inference_mode():
        main()
except Exception:
    traceback.print_exc()
    sys.stderr.flush()
    exit_code = 1
finally:
    # Kit fast shutdown can otherwise mask a pending Python exception as exit 0.
    import omni.kit.app

    omni.kit.app.get_app().post_quit(exit_code)
    app.close()
raise SystemExit(exit_code)
