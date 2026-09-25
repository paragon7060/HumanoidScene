"""Train multi-box v2 grasp with deployable-actor/privileged-critic SAC."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import traceback
from uuid import uuid4


def _gpu_lock_path() -> Path:
    """Allow separate physical GPUs while preventing duplicate runs on one GPU."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "unscoped")
    identifier = re.sub(r"[^a-zA-Z0-9_-]", "_", visible)
    return Path(f"/tmp/kuavo_multi_box_v2_grasp_sac_gpu_{identifier}.lock")


def apply_run_profile(args) -> None:
    """Apply bounded profiles after parsing so a pilot cannot become a long run."""
    if args.smoke_test and args.pilot:
        raise ValueError("--smoke-test and --pilot are mutually exclusive")
    if args.smoke_test:
        args.num_envs = min(args.num_envs, 4)
        args.max_iterations = 1
        # Includes the 0.5 s reset acceptance window and still leaves enough
        # task transitions for one optimizer wiring update.
        args.rollout_steps = 64
        args.batch_size = args.num_envs * 4
        args.replay_capacity = max(64, args.batch_size)
        args.learning_starts = 0
        args.warmup_vector_steps = 0
        args.updates_per_step = 1
        args.save_interval = 1
    elif args.pilot:
        # About 41k transitions at the defaults: enough to exercise warmup,
        # optimization, terminal statistics and checkpoints without silently
        # starting the long 2000-iteration experiment.
        args.num_envs = min(args.num_envs, 64)
        args.max_iterations = min(args.max_iterations, 20)
        args.rollout_steps = min(args.rollout_steps, 32)
        args.batch_size = min(args.batch_size, 512)
        args.replay_capacity = max(args.batch_size, min(args.replay_capacity, 50_000))
        args.learning_starts = min(args.learning_starts, 4_096)
        args.warmup_vector_steps = min(args.warmup_vector_steps, 64)
        args.updates_per_step = 1
        args.save_interval = min(args.save_interval, 5)


def _compatible_checkpoint(checkpoint: Path, manifest: dict) -> None:
    source_path = checkpoint.parent / "manifest.json"
    if not source_path.is_file():
        raise ValueError(f"Checkpoint needs its manifest.json beside it: {source_path}")
    source = json.loads(source_path.read_text())
    for key in (
        "task_family", "schema_version", "skill", "algorithm", "robot_model",
        "gripper", "actions", "observations", "observation_contract", "critic_mapping",
        "reward_profile", "exploration", "demonstrations", "self_collision",
    ):
        if source.get(key) != manifest.get(key):
            raise ValueError(f"Checkpoint {key} differs from this v2 SAC environment")


def main() -> None:
    from isaaclab.app import AppLauncher
    from ....core.paths import default_artifacts_dir
    from ....robots.base_drive import add_base_drive_cli_args, export_base_drive_cli
    from ....robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from ....robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from ....workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli

    parser = argparse.ArgumentParser(
        description="Multi-box v2 staged grasp asymmetric SAC", allow_abbrev=False)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--env-spacing", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=250_000)
    parser.add_argument("--replay-device", choices=("cpu", "cuda:0"), default="cpu")
    parser.add_argument("--learning-starts", type=int, default=100_000)
    parser.add_argument("--warmup-vector-steps", type=int, default=450)
    parser.add_argument("--warmup-action-hold-steps", type=int, default=8)
    parser.add_argument("--warmup-continuous-scale", type=float, default=0.35)
    parser.add_argument("--min-alpha", type=float, default=0.005)
    parser.add_argument("--demo-dataset", type=Path,
                        help="Successful Quest v2 HDF5 episodes; convert 403-D observations offline.")
    parser.add_argument("--demo-batch-fraction", type=float, default=0.2,
                        help="Fraction of each SAC minibatch sampled from the persistent demo replay.")
    parser.add_argument("--updates-per-step", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument(
        "--external-checkpoint-retention", action="store_true",
        help="Leave checkpoint deletion to the checksum-verifying Drive backup worker.",
    )
    parser.add_argument(
        "--self-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the privileged URDF/FCL self-collision penalty and termination.",
    )
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run four vector steps and one SAC update with at most four environments.")
    parser.add_argument(
        "--pilot", action="store_true",
        help="Bounded learning pilot: <=64 envs, 20 iterations, 50k replay, checkpoints every 5 iterations.")
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    add_base_drive_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    parser.set_defaults(
        headless=True, robot_model="s63", gripper="leju-twofinger",
        rack_rollers=True)
    args = parser.parse_args()
    positive = (
        args.num_envs, args.max_iterations, args.rollout_steps, args.batch_size,
        args.replay_capacity, args.updates_per_step, args.hidden,
        args.save_interval, args.keep_checkpoints, args.warmup_action_hold_steps,
    )
    if min(positive) < 1 or min(args.learning_starts, args.warmup_vector_steps) < 0:
        parser.error("Counts must be positive and warmup counts nonnegative")
    if not 0 <= args.min_alpha <= 0.1:
        parser.error("--min-alpha must be between zero and the initial alpha 0.1")
    if not 0 < args.warmup_continuous_scale <= 1:
        parser.error("--warmup-continuous-scale must be in (0, 1]")
    if not 0 <= args.demo_batch_fraction < 1:
        parser.error("--demo-batch-fraction must be in [0, 1)")
    if args.demo_dataset and args.demo_batch_fraction == 0:
        parser.error("--demo-batch-fraction must be positive with --demo-dataset")
    if args.env_spacing < 5.0:
        parser.error("--env-spacing must be at least 5 metres")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"Missing checkpoint: {args.checkpoint}")
    if args.demo_dataset:
        args.demo_dataset = args.demo_dataset.expanduser().resolve()
        if not args.demo_dataset.is_file():
            parser.error(f"Missing demonstration dataset: {args.demo_dataset}")
    try:
        apply_run_profile(args)
    except ValueError as error:
        parser.error(str(error))

    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)
    export_base_drive_cli(args)

    with _gpu_lock_path().open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A v2 grasp SAC process is already running") from exc

        app = AppLauncher(args).app
        env = None
        directory = None
        try:
            import torch
            from isaaclab.envs import ManagerBasedRLEnv
            from isaaclab.utils.io import dump_yaml

            from ...runners.common import install_stop_handlers
            from ...runners.storage import load_checkpoint
            from ...runners.train_asymmetric_sac import train
            from ....robots.gripper_config import resolve_gripper_settings
            from ....robots.robot_model import resolve_robot_model
            from ...envs.terminal_observation import TerminalObservationMixin
            from ..rewards import MultiBoxRewardWeights
            from ..demo_replay import load_v2_grasp_demonstrations
            from ..state.isaac_privileged_grasp import (
                GRASP_APPROACH_REWARD_SCALE_M,
                GRASP_CAPTURE_REWARD_SCALE_M,
            )
            from ..training_env_cfg import MultiBoxGraspAssemblyEnvCfg

            class TransitionEnv(TerminalObservationMixin, ManagerBasedRLEnv):
                pass

            install_stop_handlers()
            torch.manual_seed(args.seed)
            cfg = MultiBoxGraspAssemblyEnvCfg(
                num_envs=args.num_envs, env_spacing=args.env_spacing)
            cfg.multi_box = replace(
                cfg.multi_box, self_collision_enabled=bool(args.self_collision))
            cfg.multi_box.validate()
            cfg.seed = args.seed
            cfg.sim.device = args.device or "cuda:0"
            demonstration_batch = demonstration_meta = None
            if args.demo_dataset:
                demonstration_batch, demonstration_meta = load_v2_grasp_demonstrations(
                    args.demo_dataset,
                    self_collision_enabled=bool(args.self_collision),
                )
            parent = (
                args.log_dir.expanduser().resolve() if args.log_dir
                else default_artifacts_dir() / "rl" / "multi_box_v2" / "grasp_sac")
            parent.mkdir(parents=True, exist_ok=True)
            directory = parent / (
                f"sac_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}")
            directory.mkdir(exist_ok=False)
            cfg.log_dir = str(directory)

            env = TransitionEnv(cfg)
            observation_dims = {
                name: list(value)
                for name, value in env.observation_manager.group_obs_dim.items()
            }
            action_dims = {
                name: env.action_manager.get_term(name).action_dim
                for name in env.action_manager.active_terms
            }
            if demonstration_batch is not None:
                if demonstration_meta["action_terms"] != [
                    [name, width] for name, width in action_dims.items()
                ] or demonstration_batch["actor_obs"].shape[1] != observation_dims["policy"][0] \
                        or demonstration_batch["critic_obs"].shape[1] != sum(
                            dimension[0] for dimension in observation_dims.values()):
                    raise ValueError("Converted demonstration action/observation contract differs from environment")
            manifest = {
                "version": 2,
                "task_family": "multi_box_v2",
                "schema_version": int(cfg.multi_box.schema_version),
                "skill": "grasp",
                "algorithm": "asymmetric_sac",
                "robot_model": resolve_robot_model().name,
                "gripper": resolve_gripper_settings().name,
                "actions": action_dims,
                "observations": observation_dims,
                "observation_contract": "neutral_flap_center_tcp_frame_v1",
                "critic_mapping": {
                    "actor": ["policy"],
                    "critic": ["policy", "critic"],
                },
                "num_envs": args.num_envs,
                "seed": args.seed,
                "discount": MultiBoxRewardWeights().discount,
                "reward_profile": {
                    "weights": asdict(MultiBoxRewardWeights()),
                    "approach_scale_m": GRASP_APPROACH_REWARD_SCALE_M,
                    "capture_scale_m": GRASP_CAPTURE_REWARD_SCALE_M,
                    "geometry_profile": "opposing_flap_weaker_hand_reach_and_pinch_gated_lift_v2",
                },
                "exploration": {
                    "min_alpha": args.min_alpha,
                    "warmup_action_hold_steps": args.warmup_action_hold_steps,
                    "warmup_continuous_scale": args.warmup_continuous_scale,
                },
                "demonstrations": (
                    {key: value for key, value in demonstration_meta.items() if key != "path"}
                    | {"batch_fraction": args.demo_batch_fraction}
                    if demonstration_meta else None
                ),
                "demo_source_path": demonstration_meta["path"] if demonstration_meta else None,
                "run_profile": (
                    "smoke" if args.smoke_test else "pilot" if args.pilot else "train"
                ),
                "terminal_contract": {
                    "success": "exact_grasp_success",
                    "invalid_reset": "partial_respawn_excluded_from_replay",
                    "safety_thresholds": {
                        "rack_contact_force_n": float(cfg.multi_box.rack_contact_force),
                        "obstacle_contact_force_n": float(cfg.task.obstacle_contact_force),
                        "workspace_radius_m": float(cfg.multi_box.workspace_radius),
                        "max_box_lift_height_m": float(cfg.multi_box.max_box_lift_height),
                        "max_box_linear_speed_mps": float(cfg.multi_box.max_box_linear_speed),
                        "max_box_angular_speed_radps": float(cfg.multi_box.max_box_angular_speed),
                    },
                    "unsafe": [
                        "robot_rack_collision",
                        *(["self_collision"] if args.self_collision else []),
                        "obstacle_collision", "workspace_limit", "box_drop",
                        "box_lift_limit", "box_speed_limit",
                    ],
                    "timeouts_bootstrap": True,
                },
                "self_collision": {
                    "enabled": bool(args.self_collision),
                    "backend": "reviewed_urdf_fcl",
                    "clearance_m": float(cfg.multi_box.self_collision_clearance),
                    "actor_observation": False,
                },
            }
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False))
            dump_yaml(str(directory / "env.yaml"), cfg)
            dump_yaml(str(directory / "agent.yaml"), {
                key: value for key, value in vars(args).items()
                if key not in {"log_dir", "checkpoint", "demo_dataset"}
            })

            state = None
            if args.checkpoint:
                _compatible_checkpoint(args.checkpoint, manifest)
                state = load_checkpoint(args.checkpoint, device=cfg.sim.device)
                if state.get("algorithm") != "asymmetric_sac":
                    raise ValueError("Checkpoint is not multi-box v2 asymmetric SAC")
            print(
                f"[V2 SAC] envs={args.num_envs} policy={observation_dims['policy']} "
                f"privileged={observation_dims['critic']} run={directory.name}",
                flush=True,
            )
            train(env, args, directory, state, demonstration_batch)
            (directory / "status.json").write_text(json.dumps({
                "status": "complete",
                "algorithm": "asymmetric_sac",
                "smoke_test": bool(args.smoke_test),
                "pilot": bool(args.pilot),
            }, indent=2))
        except BaseException as error:
            if directory is not None:
                (directory / "status.json").write_text(json.dumps({
                    "status": "failed",
                    "algorithm": "asymmetric_sac",
                    "error": repr(error),
                    "traceback": traceback.format_exc(),
                }, indent=2))
            raise
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


if __name__ == "__main__":
    main()
