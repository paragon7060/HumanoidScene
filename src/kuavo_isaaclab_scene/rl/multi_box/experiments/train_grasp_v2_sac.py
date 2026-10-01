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
        if hasattr(args, "critic_warmup_updates"):
            args.critic_warmup_updates = 0
        args.save_interval = 1
        if hasattr(args, "demo_pretrain_steps"):
            args.demo_pretrain_steps = min(args.demo_pretrain_steps, 4)
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
        if hasattr(args, "critic_warmup_updates"):
            args.critic_warmup_updates = min(args.critic_warmup_updates, 64)
        args.save_interval = min(args.save_interval, 5)
        if hasattr(args, "demo_pretrain_steps"):
            args.demo_pretrain_steps = min(args.demo_pretrain_steps, 100)


def _compatible_checkpoint(checkpoint: Path, manifest: dict, *, data_only=False) -> None:
    source_path = checkpoint.parent / "manifest.json"
    if not source_path.is_file():
        raise ValueError(f"Checkpoint needs its manifest.json beside it: {source_path}")
    source = json.loads(source_path.read_text())
    for key in (
        "task_family", "schema_version", "skill", "algorithm", "robot_model",
        "gripper", "actions", "action_contract", "observations", "observation_contract", "critic_mapping", "contact_contract",
        "reward_profile", "exploration", "demonstrations", "self_collision",
        "action_projection", "discount", "terminal_contract", "sac_stability",
    ):
        if data_only and key in {"exploration", "demonstrations", "sac_stability"}:
            continue
        saved, requested = source.get(key), manifest.get(key)
        if key in {"exploration", "demonstrations"} and isinstance(saved, dict) and isinstance(requested, dict):
            defaults = ({"online_ik_min_episode_fraction": 0.0, "online_ik_decay_updates": 0,
                         "ik_orientation_mode": "full"}
                        if key == "exploration" else {
                            "teacher_batch_fraction": 0.0, "teacher_min_batch_fraction": 0.0,
                            "teacher_bc_strength": 10.0, "teacher_decay_updates": 128_000,
                            "teacher_usage": "independent_actor_labels_only; no_hypothetical_Q_transitions"})
            saved, requested = defaults | saved, defaults | requested
        if key == "exploration" and isinstance(saved, dict) and isinstance(requested, dict):
            # Episode assignment and the critic-only warmup threshold control
            # collection/update timing, not learned parameters or physical contracts.
            saved = {name: value for name, value in saved.items()
                     if name not in {"online_ik_episode_fraction", "critic_warmup_updates"}}
            requested = {name: value for name, value in requested.items()
                         if name not in {"online_ik_episode_fraction", "critic_warmup_updates"}}
        if saved != requested:
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
    parser.add_argument("--min-alpha", type=float, default=0.00001)
    parser.add_argument("--initial-alpha", type=float, default=0.001)
    parser.add_argument("--max-alpha", type=float, default=0.001)
    parser.add_argument("--critic-layer-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actor-q-normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--initial-policy-std", type=float, default=0.15)
    parser.add_argument("--max-policy-std", type=float, default=0.3)
    parser.add_argument("--guided-warmup-mode", choices=("bc", "ik"), default="bc")
    parser.add_argument("--ik-grasp-goal", choices=("center", "demo", "center-to-demo"), default="demo")
    parser.add_argument("--ik-orientation-mode", choices=("full", "closing-axis"), default="full",
                        help="Full wrist pose or symmetric jaw-axis alignment, leaving roll free.")
    parser.add_argument("--ik-lift-distance-m", type=float, default=0.025)
    parser.add_argument("--ik-base-clearance-m", type=float, default=0.65)
    parser.add_argument("--ik-torso-forward-m", type=float, default=0.0)
    parser.add_argument("--teacher-pretrain-steps", type=int, default=5000,
                        help="Fit the SAC actor to new IK-collected actions once warmup ends.")
    parser.add_argument("--online-teacher-labels", action=argparse.BooleanOptionalAction, default=False,
                        help="Label SAC-visited states for actor imitation without overriding its actions.")
    parser.add_argument("--online-ik-episode-fraction", type=float, default=0.0,
                        help="Expert episode fraction at SAC start; decays at episode boundaries.")
    from .imitation_schedule import add_teacher_schedule_arguments, validate_teacher_schedule
    add_teacher_schedule_arguments(parser)
    parser.add_argument("--actor-lr", type=float, default=0.00003)
    parser.add_argument("--critic-warmup-updates", type=int, default=500,
                        help="Learn Q before allowing it to change the pretrained actor.")
    parser.add_argument("--success-replay-capacity", type=int, default=10000)
    parser.add_argument("--success-batch-fraction", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--entropy-backup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-feature-mode", choices=("flat", "grasp_target"), default="grasp_target")
    parser.add_argument("--freeze-actor-normalizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--goal-replay-capacity", type=int, default=100_000)
    parser.add_argument("--goal-batch-fraction", type=float, default=0.25)
    parser.add_argument("--demo-dataset", type=Path,
                        help="Successful Quest v2 HDF5 episodes; convert 403-D observations offline.")
    parser.add_argument("--demo-batch-fraction", type=float, default=0.2,
                        help="Initial actor imitation weight and demo minibatch fraction; critic uses online replay only.")
    parser.add_argument("--demo-bc-strength", type=float, default=10.0,
                        help="Independent loss scale; sampling still starts at 20% and decays.")
    parser.add_argument("--demo-decay-fraction", type=float, default=0.3,
                        help="Fraction of planned SAC updates over which demo imitation decays to zero.")
    parser.add_argument("--demo-pretrain-steps", type=int, default=1000,
                        help="Actor-only behavior-cloning updates before online rollout.")
    parser.add_argument("--demo-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--demo-guided-warmup", action=argparse.BooleanOptionalAction, default=True,
                        help="Use the pretrained actor with correlated noise during warmup.")
    parser.add_argument("--demo-warmup-noise-scale", type=float, default=0.12)
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
    parser.add_argument("--teacher-label-checkpoint", type=Path,
                        help="Import actor-only controller labels from a compatible older v2 checkpoint.")
    parser.add_argument("--experience-checkpoint", type=Path,
                        help="Reuse only genuine success-tail transitions; never source model/Q/optimizers.")
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
    try:
        validate_teacher_schedule(args)
    except ValueError as error:
        parser.error(str(error))
    positive = (
        args.num_envs, args.max_iterations, args.rollout_steps, args.batch_size,
        args.replay_capacity, args.updates_per_step, args.hidden,
        args.save_interval, args.keep_checkpoints, args.warmup_action_hold_steps,
    )
    if min(positive) < 1 or min(args.learning_starts, args.warmup_vector_steps) < 0:
        parser.error("Counts must be positive and warmup counts nonnegative")
    if not 0 <= args.min_alpha <= args.initial_alpha <= 0.1:
        parser.error("Require 0 <= min-alpha <= initial-alpha <= 0.1")
    if not args.initial_alpha <= args.max_alpha <= 0.1:
        parser.error("Require initial-alpha <= max-alpha <= 0.1")
    if not 0 < args.initial_policy_std <= 1 or not 0 < args.reward_scale < 1000:
        parser.error("Invalid policy standard deviation or reward scale")
    if not args.initial_policy_std <= args.max_policy_std <= 1:
        parser.error("Require initial-policy-std <= max-policy-std <= 1")
    if args.guided_warmup_mode == "ik" and not args.demo_dataset:
        parser.error("IK warmup needs successful demos for wrist orientations")
    if args.online_teacher_labels and (args.guided_warmup_mode != "ik"
                                      or not args.demo_guided_warmup):
        parser.error("Online teacher labels require enabled IK guidance")
    if not 0 <= args.online_ik_episode_fraction < 1 or (args.online_ik_episode_fraction and (
            args.guided_warmup_mode != "ik" or not args.demo_guided_warmup or not args.demo_dataset)):
        parser.error("Online IK episodes require a fraction in [0,1) and enabled IK guidance")
    if args.teacher_pretrain_steps < 0:
        parser.error("--teacher-pretrain-steps must be nonnegative")
    if not 0.008 <= args.ik_lift_distance_m <= 0.15:
        parser.error("IK wrist lift distance must be in [0.008, 0.15] m")
    if not 0.4 <= args.ik_base_clearance_m <= 0.8 or not 0 <= args.ik_torso_forward_m <= 0.15:
        parser.error("Invalid IK base clearance or bounded upright torso assistance")
    if not 0 < args.actor_lr <= 0.001 or args.critic_warmup_updates < 0 \
            or args.success_replay_capacity < 1 \
            or not 0 <= args.success_batch_fraction < 1 \
            or args.success_batch_fraction + args.goal_batch_fraction >= 1:
        parser.error("Invalid critic warmup, actor learning rate or success replay")
    if args.goal_replay_capacity < 1 or not 0 <= args.goal_batch_fraction < 1:
        parser.error("Invalid goal replay configuration")
    if not 0 < args.warmup_continuous_scale <= 1:
        parser.error("--warmup-continuous-scale must be in (0, 1]")
    if not 0 <= args.demo_batch_fraction < 1:
        parser.error("--demo-batch-fraction must be in [0, 1)")
    if not 0 <= args.demo_bc_strength <= 1000:
        parser.error("Invalid demonstration loss strength")
    if not 0 < args.demo_decay_fraction <= 1:
        parser.error("--demo-decay-fraction must be in (0, 1]")
    if args.demo_pretrain_steps < 0 or args.demo_pretrain_batch_size < 1:
        parser.error("Demo pretraining steps must be nonnegative and batch size positive")
    if not 0 <= args.demo_warmup_noise_scale <= 1:
        parser.error("--demo-warmup-noise-scale must be in [0, 1]")
    if args.demo_dataset and args.demo_batch_fraction == 0:
        parser.error("--demo-batch-fraction must be positive with --demo-dataset")
    if args.env_spacing < 5.0:
        parser.error("--env-spacing must be at least 5 metres")
    if args.checkpoint:
        args.checkpoint = args.checkpoint.expanduser().resolve()
        if not args.checkpoint.is_file():
            parser.error(f"Missing checkpoint: {args.checkpoint}")
    if args.teacher_label_checkpoint:
        args.teacher_label_checkpoint = args.teacher_label_checkpoint.expanduser().resolve()
        if not args.teacher_label_checkpoint.is_file():
            parser.error(f"Missing teacher label checkpoint: {args.teacher_label_checkpoint}")
    if args.experience_checkpoint:
        args.experience_checkpoint = args.experience_checkpoint.expanduser().resolve()
        if args.checkpoint:
            parser.error("--experience-checkpoint starts fresh; cannot combine with --checkpoint")
        if not args.experience_checkpoint.is_file():
            parser.error(f"Missing experience checkpoint: {args.experience_checkpoint}")
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
            from ..geometry.grasp import GRASP_ASSIGNMENT_SCALE_M
            from ..metrics.potentials import (
                FRONT_STAGE_CLEARANCE_M, FRONT_STAGE_LANE_TOLERANCE_M,
                FRONT_STAGE_REWARD_SCALE_M,
            )
            from ..training_env_cfg import MultiBoxGraspAssemblyEnvCfg
            from .guided_exploration import GraspActionProjector

            class TransitionEnv(TerminalObservationMixin, ManagerBasedRLEnv):
                pass

            install_stop_handlers(defer=True)
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
            env.enable_numerical_dynamics_recovery()
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
                "action_contract": "s63_upright_torso_xz_fixed_pitch_v1",
                "action_projection": GraspActionProjector.name,
                "observations": observation_dims,
                "observation_contract": "neutral_flap_center_controller_state_actual_base_twist_v2",
                "contact_contract": "max_filtered_rack_and_workcell_pairs_without_boxes_or_floor_v2",
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
                    "assignment_scale_m": GRASP_ASSIGNMENT_SCALE_M,
                    "capture_scale_m": GRASP_CAPTURE_REWARD_SCALE_M,
                    "front_stage_clearance_m": FRONT_STAGE_CLEARANCE_M,
                    "front_stage_lane_tolerance_m": FRONT_STAGE_LANE_TOLERANCE_M,
                    "front_stage_scale_m": FRONT_STAGE_REWARD_SCALE_M,
                    "geometry_profile": "rack_front_lane_then_opposing_flap_reach_v3",
                },
                "exploration": {
                    "min_alpha": args.min_alpha,
                    "initial_alpha": args.initial_alpha,
                    "initial_policy_std": args.initial_policy_std,
                    "max_policy_std": args.max_policy_std,
                    "guided_warmup_mode": args.guided_warmup_mode,
                    "teacher_pretrain_steps": args.teacher_pretrain_steps,
                    "ik_grasp_goal": args.ik_grasp_goal,
                    "ik_orientation_mode": args.ik_orientation_mode,
                    "ik_lift_distance_m": args.ik_lift_distance_m,
                    "ik_base_clearance_m": args.ik_base_clearance_m,
                    "ik_torso_forward_m": args.ik_torso_forward_m,
                    "online_teacher_labels": args.online_teacher_labels,
                    "online_ik_episode_fraction": args.online_ik_episode_fraction,
                    "online_ik_min_episode_fraction": args.online_ik_min_episode_fraction,
                    "online_ik_decay_updates": args.online_ik_decay_updates,
                    "actor_lr": args.actor_lr,
                    "critic_warmup_updates": args.critic_warmup_updates,
                    "success_replay_capacity": args.success_replay_capacity,
                    "success_batch_fraction": args.success_batch_fraction,
                    "reward_scale": args.reward_scale,
                    "entropy_backup": args.entropy_backup,
                    "actor_feature_mode": args.actor_feature_mode,
                    "freeze_actor_normalizer": args.freeze_actor_normalizer,
                    "goal_replay_capacity": args.goal_replay_capacity,
                    "goal_batch_fraction": args.goal_batch_fraction,
                    "warmup_action_hold_steps": args.warmup_action_hold_steps,
                    "warmup_continuous_scale": args.warmup_continuous_scale,
                    "demo_guided_warmup": args.demo_guided_warmup,
                    "demo_warmup_noise_scale": args.demo_warmup_noise_scale,
                },
                "demonstrations": (
                    {key: value for key, value in demonstration_meta.items() if key != "path"}
                    | {"initial_batch_fraction": args.demo_batch_fraction,
                       "decay_fraction": args.demo_decay_fraction,
                       "pretrain_steps": args.demo_pretrain_steps,
                       "pretrain_batch_size": args.demo_pretrain_batch_size,
                       "bc_strength": args.demo_bc_strength,
                       "teacher_batch_fraction": args.teacher_batch_fraction,
                       "teacher_min_batch_fraction": args.teacher_min_batch_fraction,
                       "teacher_bc_strength": args.teacher_bc_strength,
                       "teacher_decay_updates": args.teacher_decay_updates,
                       "teacher_usage": "independent_actor_labels_only; no_hypothetical_Q_transitions",
                       "usage": "actor_behavior_cloning_only; recorded_rewards_ignored"}
                    if demonstration_meta else None
                ),
                "collection_data_contract": {
                    "teacher_critical_fraction": 0.5,
                    "teacher_critical_rows": "both_flaps_within_0p25m_or_close_label",
                    "teacher_servo_velocity": "measured_joint_velocity",
                    "protected_success_history_steps": 64,
                    "history_crosses_resets": False,
                },
                "entropy_contract": "squash_aware_active_dims_v2",
                "sac_stability": {
                    "max_alpha": args.max_alpha,
                    "critic_layer_norm": args.critic_layer_norm,
                    "actor_q_normalize": args.actor_q_normalize,
                },
                "experience_initialization": {
                    "experience_source": str(args.experience_checkpoint) if args.experience_checkpoint else None,
                    "experience_import": "executed_success_tails_only; model_Q_optimizers_ignored",
                },
                "numerical_failure_contract": {
                    "reset_fk_refresh": True,
                    "pre_grasp_robot_pose_guard": True,
                    "recovery": "partial_respawn_before_next_physics_write",
                    "outcome": "failure_excluded_from_replay_and_imitation",
                    "diagnostics": ["gravity_nonfinite", "feedforward_nonfinite", "joint_state_nonfinite",
                                    "mass_nonfinite", "coriolis_nonfinite", "root_state_nonfinite",
                                    "robot_pose_invalid"],
                },
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
            manifest["teacher_imitation_retention"] = {
                "persistent_critical_capacity": 100_000,
                "critical_batch_fraction": 0.5,
                "checkpoint_snapshot_max_rows": 100_000,
                "teacher_label_source": str(args.teacher_label_checkpoint) if args.teacher_label_checkpoint else None,
                "critic_uses_imported_teacher_labels": False,
            }
            if args.teacher_label_checkpoint:
                _compatible_checkpoint(args.teacher_label_checkpoint, manifest, data_only=True)
            if args.experience_checkpoint:
                _compatible_checkpoint(args.experience_checkpoint, manifest, data_only=True)
            (directory / "manifest.json").write_text(
                json.dumps(manifest, indent=2, allow_nan=False))
            dump_yaml(str(directory / "env.yaml"), cfg)
            dump_yaml(str(directory / "agent.yaml"), {
                key: value for key, value in vars(args).items()
                if key not in {"log_dir", "checkpoint", "demo_dataset", "teacher_label_checkpoint", "experience_checkpoint"}
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
            agent = train(env, args, directory, state, demonstration_batch)
            (directory / "status.json").write_text(json.dumps({
                "status": "stopped" if agent.stopped_early else "complete",
                "iteration": agent.last_iteration,
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
