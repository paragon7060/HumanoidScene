"""Opt-in headless single-cell check of the deployable v2 runtime.

This is not a training runner and never starts OpenXR. It advances zero actions
only long enough to inspect reset, target selection, and actor-state assembly.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import fcntl
from pathlib import Path


LOCK_PATH = Path("/tmp/kuavo_multi_box_v2_runtime_inspect.lock")


def main() -> None:
    from isaaclab.app import AppLauncher
    from ....robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from ....robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from ....workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli

    parser = argparse.ArgumentParser(description="Headless multi-box v2 runtime inspection",
                                     allow_abbrev=False)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--box-count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(headless=True, robot_model="s63", gripper="leju-twofinger")
    args = parser.parse_args()
    if args.steps < 1 or not 1 <= args.box_count <= 12:
        parser.error("--steps must be positive and --box-count must be 1-12.")
    if not args.headless:
        parser.error("This inspection command must run headless.")
    if args.xr:
        parser.error("This inspection command does not start XR; omit --xr.")
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)

    with LOCK_PATH.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Multi-box v2 runtime inspection is already running.") from exc

        app = AppLauncher(args).app
        env = None
        try:
            import torch
            from isaaclab.envs import ManagerBasedRLEnv
            from ..spec import MultiBoxSpec
            from ..teleop_env_cfg import MultiBoxTeleopEnvCfg
            from ..state.isaac_deployable import IsaacDeployableStateAdapter

            spec = replace(MultiBoxSpec(), full_spawn_count_range=(args.box_count, args.box_count))
            cfg = MultiBoxTeleopEnvCfg(multi_box=spec)
            cfg.sim.device = args.device or "cuda:0"
            cfg.seed = args.seed
            env = ManagerBasedRLEnv(cfg)
            env.reset(seed=args.seed)
            runtime = IsaacDeployableStateAdapter(env)
            runtime.reset()
            action = torch.zeros(1, env.action_manager.total_action_dim, device=env.device)
            for index in range(args.steps):
                env.step(action)
                result = runtime.step(previous_action=action, dt=env.step_dt)
                control = result.state.control
                print(
                    f"[V2 RUNTIME] step={index + 1} selected={result.selected_box.item()} "
                    f"target={control.target_box.item()} skill={control.current_skill.item()} "
                    f"placed={int(result.state.placement.placed.sum().item())} "
                    f"waiting_without_box={int(result.waiting_without_box.item())}",
                    flush=True,
                )
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


if __name__ == "__main__":
    main()
