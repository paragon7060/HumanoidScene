"""Train the isolated multi-box v2 grasp skill with asymmetric PPO."""

from __future__ import annotations

import argparse
from datetime import datetime
import importlib.metadata
import json
from pathlib import Path
from uuid import uuid4


def _compatible_checkpoint(checkpoint: Path, manifest: dict) -> None:
    source_path = checkpoint.parent / "manifest.json"
    if not source_path.is_file():
        raise ValueError(f"Checkpoint needs its manifest.json beside it: {source_path}")
    source = json.loads(source_path.read_text())
    for key in (
        "task_family", "schema_version", "skill", "robot_model", "gripper",
        "actions", "observations", "critic_mapping",
    ):
        if source.get(key) != manifest.get(key):
            raise ValueError(f"Checkpoint {key} differs from the v2 grasp environment.")


def main() -> None:
    from isaaclab.app import AppLauncher
    from ....core.paths import default_artifacts_dir
    from ....robots.base_drive import add_base_drive_cli_args, export_base_drive_cli
    from ....robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from ....robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from ....workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli

    parser = argparse.ArgumentParser(
        description="Multi-box v2 staged grasp PPO", allow_abbrev=False)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--env-spacing", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run one four-step PPO iteration with at most four environments.",
    )
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    add_base_drive_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        headless=True, robot_model="s63", gripper="leju-twofinger")
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.steps_per_env, args.save_interval) < 1:
        parser.error("Environment, iteration, rollout, and save counts must be positive.")
    if args.env_spacing < 5.0:
        parser.error("--env-spacing must be at least 5 metres.")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"Missing checkpoint: {args.checkpoint}")
    try:
        version = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        parser.error("Install the RL extra: python -m pip install -e '.[rl]'")
    if version != "3.1.2":
        parser.error(f"Expected rsl-rl-lib==3.1.2, found {version}.")
    if args.smoke_test:
        args.num_envs = min(args.num_envs, 4)
        args.max_iterations = 1
        args.steps_per_env = 4
        args.save_interval = 1

    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)
    export_base_drive_cli(args)
    app = AppLauncher(args).app
    env = None
    runner = None
    directory = None
    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.utils.io import dump_yaml
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        from ...agents.multi_box_v2_ppo import MultiBoxV2GraspPPOCfg
        from ...runners.checkpoints import AtomicCheckpointMixin
        from ...runners.common import install_stop_handlers
        from ....robots.gripper_config import resolve_gripper_settings
        from ....robots.robot_model import resolve_robot_model
        from ..training_env_cfg import MultiBoxGraspAssemblyEnvCfg

        class DriveSafeRunner(AtomicCheckpointMixin, OnPolicyRunner):
            pass

        install_stop_handlers()
        cfg = MultiBoxGraspAssemblyEnvCfg(
            num_envs=args.num_envs, env_spacing=args.env_spacing)
        cfg.seed = args.seed
        cfg.sim.device = args.device or "cuda:0"
        agent = MultiBoxV2GraspPPOCfg(
            seed=args.seed,
            device=cfg.sim.device,
            num_steps_per_env=args.steps_per_env,
            max_iterations=args.max_iterations,
            save_interval=args.save_interval,
        )
        if args.num_envs * args.steps_per_env < agent.algorithm.num_mini_batches:
            raise ValueError("PPO has more mini-batches than rollout samples.")

        parent = (args.log_dir.expanduser().resolve() if args.log_dir
                  else default_artifacts_dir() / "rl" / "multi_box_v2" / "grasp")
        parent.mkdir(parents=True, exist_ok=True)
        directory = parent / f"train_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
        directory.mkdir(exist_ok=False)
        cfg.log_dir = str(directory)

        base_env = ManagerBasedRLEnv(cfg)
        env = RslRlVecEnvWrapper(base_env, clip_actions=agent.clip_actions)
        model = resolve_robot_model()
        hand = resolve_gripper_settings()
        observation_dims = {
            name: list(value) for name, value in base_env.observation_manager.group_obs_dim.items()
        }
        manifest = {
            "version": 2,
            "task_family": "multi_box_v2",
            "schema_version": int(cfg.multi_box.schema_version),
            "skill": "grasp",
            "robot_model": model.name,
            "gripper": hand.name,
            "actions": {
                name: base_env.action_manager.get_term(name).action_dim
                for name in base_env.action_manager.active_terms
            },
            "observations": observation_dims,
            "critic_mapping": agent.obs_groups,
            "num_envs": args.num_envs,
            "seed": args.seed,
            "git_hygiene": {
                "run_artifacts_ignored": True,
                "personal_paths_in_manifest": False,
            },
        }
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False))
        dump_yaml(str(directory / "env.yaml"), cfg)
        dump_yaml(str(directory / "agent.yaml"), agent)
        if args.checkpoint:
            _compatible_checkpoint(args.checkpoint, manifest)

        runner = DriveSafeRunner(
            env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint))
        print(
            f"[V2 PPO] envs={args.num_envs} actor={observation_dims['policy']} "
            f"privileged={observation_dims['critic']} run={directory.name}",
            flush=True,
        )
        runner.learn(
            num_learning_iterations=agent.max_iterations,
            init_at_random_ep_len=False,
        )
        final_path = directory / f"model_{runner.current_learning_iteration}.pt"
        if not final_path.exists():
            runner.save(final_path)
        (directory / "status.json").write_text(json.dumps({
            "status": "complete",
            "last_iteration": runner.current_learning_iteration,
            "smoke_test": bool(args.smoke_test),
        }, indent=2))
    except BaseException as error:
        if directory is not None:
            (directory / "status.json").write_text(json.dumps({
                "status": "failed", "error": repr(error)}, indent=2))
        raise
    finally:
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


if __name__ == "__main__":
    main()
