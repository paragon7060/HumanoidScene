"""Headless smoke check for the v2 grasp manager assembly."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path


LOCK_PATH = Path("/tmp/kuavo_multi_box_v2_grasp_assembly.lock")


def main() -> None:
    from isaaclab.app import AppLauncher
    from ....robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from ....robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from ....workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli

    parser = argparse.ArgumentParser(
        description="Headless multi-box v2 grasp assembly inspection",
        allow_abbrev=False,
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        headless=True, robot_model="s63", gripper="leju-twofinger")
    args = parser.parse_args()
    if args.steps < 1 or args.num_envs < 1:
        parser.error("--steps and --num-envs must be positive.")
    if not args.headless or args.xr:
        parser.error("This smoke check is headless and does not start XR.")
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)

    with LOCK_PATH.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("The v2 grasp assembly check is already running.") from exc

        app = AppLauncher(args).app
        env = None
        try:
            import torch
            from isaaclab.envs import ManagerBasedRLEnv
            from ..training_env_cfg import MultiBoxGraspAssemblyEnvCfg

            cfg = MultiBoxGraspAssemblyEnvCfg(
                num_envs=args.num_envs, env_spacing=8.0)
            cfg.sim.device = args.device or "cuda:0"
            cfg.seed = args.seed
            env = ManagerBasedRLEnv(cfg)
            observation, _ = env.reset(seed=args.seed)
            action = torch.zeros_like(env.action_manager.action)
            reward = torch.zeros(args.num_envs, device=env.device)
            for _ in range(args.steps):
                observation, reward, _, _, _ = env.step(action)
            grasp = env._multi_box_privileged_grasp_step
            safety = env._multi_box_grasp_safety_step
            policy = observation["policy"]
            critic = observation["critic"]
            print(
                f"[V2 GRASP ASSEMBLY] envs={args.num_envs} "
                f"actions={env.action_manager.total_action_dim} "
                f"policy_shape={tuple(policy.shape)} "
                f"critic_shape={tuple(critic.shape)} "
                f"active_boxes={env._multi_box_active.sum(-1).tolist()} "
                f"grasp_targets={grasp.target_logical_id.tolist()} "
                f"contact_shape={tuple(grasp.contacts.force_n.shape)} "
                f"reward={reward.tolist()} "
                f"unsafe={safety.unsafe.tolist()} "
                f"obstacle_force_n={safety.obstacle_force_n.tolist()} "
                f"reward_terms={env.reward_manager.active_terms} "
                f"termination_terms={env.termination_manager.active_terms} "
                f"prepared_state={cfg.prepared_state_name}",
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
