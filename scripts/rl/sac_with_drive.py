#!/usr/bin/env python3
"""GPU-isolated SAC with the existing Drive supervisor and verified retention."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4

from drive_backup import ROOT
from gpu_budget import GpuBudget
from train_with_drive import archive, supervise, add_action_space_argument


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument(
        "--experiment",
        choices=("flap-pick", "mobile-flap-pick", "multi-box-v2-grasp"),
        default="flap-pick",
    )
    parser.add_argument("--robot-model", choices=("s200062", "s63", "s56"), default="s200062")
    parser.add_argument("--gripper", help="Model-compatible gripper preset; omitted uses the model default")
    add_action_space_argument(parser)
    parser.add_argument("--source-root", type=Path, default=ROOT,
                        help="Optional frozen source checkout; Drive auth stays in the original checkout")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--updates-per-step", type=int, default=16)
    parser.add_argument("--learning-starts", type=int, default=100000)
    parser.add_argument("--warmup-vector-steps", type=int, default=450)
    parser.add_argument("--warmup-action-hold-steps", type=int, default=4)
    parser.add_argument("--warmup-continuous-scale", type=float, default=0.35)
    parser.add_argument("--min-alpha", type=float, default=0.00001)
    parser.add_argument("--initial-alpha", type=float, default=0.001)
    parser.add_argument("--initial-policy-std", type=float, default=0.15)
    parser.add_argument("--max-policy-std", type=float, default=0.3)
    parser.add_argument("--guided-warmup-mode", choices=("bc", "ik"), default="bc")
    parser.add_argument("--ik-grasp-goal", choices=("center", "demo", "center-to-demo"), default="center")
    parser.add_argument("--ik-lift-distance-m", type=float, default=0.025)
    parser.add_argument("--ik-base-clearance-m", type=float, default=0.65)
    parser.add_argument("--ik-torso-forward-m", type=float, default=0.0)
    parser.add_argument("--teacher-pretrain-steps", type=int, default=5000)
    parser.add_argument("--online-teacher-labels", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-lr", type=float, default=0.00003)
    parser.add_argument("--critic-warmup-updates", type=int, default=500)
    parser.add_argument("--success-replay-capacity", type=int, default=10000)
    parser.add_argument("--success-batch-fraction", type=float, default=0.05)
    parser.add_argument("--reward-scale", type=float, default=10.0)
    parser.add_argument("--entropy-backup", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--actor-feature-mode", choices=("flat", "grasp_target"), default="grasp_target")
    parser.add_argument("--freeze-actor-normalizer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--goal-replay-capacity", type=int, default=100_000)
    parser.add_argument("--goal-batch-fraction", type=float, default=0.25)
    parser.add_argument("--demo-dataset", type=Path,
                        help="Multi-box v2 only: successful Quest demonstrations for replay")
    parser.add_argument("--demo-batch-fraction", type=float, default=0.2)
    parser.add_argument("--demo-bc-strength", type=float, default=10.0)
    parser.add_argument("--demo-decay-fraction", type=float, default=0.3)
    parser.add_argument("--demo-pretrain-steps", type=int, default=1000)
    parser.add_argument("--demo-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--demo-guided-warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demo-warmup-noise-scale", type=float, default=0.12)
    parser.add_argument("--replay-capacity", type=int, default=1000000)
    parser.add_argument("--replay-device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--save-interval", type=int, default=2)
    parser.add_argument(
        "--self-collision", action=argparse.BooleanOptionalAction, default=True,
        help="Multi-box v2 only: enable the reviewed URDF/FCL self-collision check.",
    )
    parser.add_argument("--remote-root", default=os.environ.get(
        "RL_DRIVE_REMOTE_ROOT", "gdrive:HumanoidScene-RL"))
    parser.add_argument("--gpu-limit-mib", type=int, default=34816)
    parser.add_argument("--gpu-reserve-mib", type=int, default=8192)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    if (min(args.num_envs, args.max_iterations, args.rollout_steps, args.batch_size, args.updates_per_step,
            args.replay_capacity, args.save_interval, args.gpu_limit_mib, args.max_seconds,
            args.warmup_action_hold_steps) < 1
            or min(args.gpu, args.gpu_reserve_mib, args.learning_starts, args.warmup_vector_steps) < 0):
        parser.error("Invalid counts or resource budget")
    if not 0 <= args.min_alpha <= 0.1:
        parser.error("--min-alpha must be between zero and the initial alpha 0.1")
    if not 0 < args.warmup_continuous_scale <= 1:
        parser.error("--warmup-continuous-scale must be in (0, 1]")
    if not 0 <= args.demo_batch_fraction < 1:
        parser.error("--demo-batch-fraction must be in [0, 1)")
    if not 0 < args.demo_decay_fraction <= 1:
        parser.error("--demo-decay-fraction must be in (0, 1]")
    if args.demo_pretrain_steps < 0 or args.demo_pretrain_batch_size < 1 \
            or not 0 <= args.demo_warmup_noise_scale <= 1:
        parser.error("Invalid demonstration warmup configuration")
    if args.demo_dataset and args.experiment != "multi-box-v2-grasp":
        parser.error("--demo-dataset is only supported for multi-box v2 grasp")
    if args.demo_dataset and not args.demo_dataset.is_file():
        parser.error("Missing demonstration dataset")
    if args.online_teacher_labels and (args.experiment != "multi-box-v2-grasp"
                                      or not args.demo_dataset
                                      or args.guided_warmup_mode != "ik"
                                      or not args.demo_guided_warmup):
        parser.error("Online teacher labels require v2 demos and enabled IK guidance")
    if args.replay_capacity < args.num_envs:
        parser.error("Replay capacity must hold a full vector step")
    if not 0.008 <= args.ik_lift_distance_m <= 0.15:
        parser.error("IK wrist lift distance must be in [0.008, 0.15] m")
    if not 0.4 <= args.ik_base_clearance_m <= 0.8 or not 0 <= args.ik_torso_forward_m <= 0.15:
        parser.error("Invalid IK base clearance or bounded upright torso assistance")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("Missing checkpoint")
    source = args.source_root.resolve()
    launcher_name = {
        "flap-pick": "flap_pick.sh",
        "mobile-flap-pick": "mobile_flap_pick.sh",
        "multi-box-v2-grasp": "multi_box.sh",
    }[args.experiment]
    launcher = source / "scripts/rl" / launcher_name
    if not launcher.is_file():
        parser.error(f"Missing experiment launcher: {launcher}")
    parent = (args.experiment_dir or ROOT / "artifacts/rl/drive_runs" /
              f"sac_gpu{args.gpu}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}").resolve()
    parent.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="8",
                       ISAACLAB_PYTHON=str(Path.home() / "miniconda3/envs/env_isaaclab_232/bin/python"),
                       PYTHONPATH=str(source / "src"))
    mode = "grasp-v2-sac" if args.experiment == "multi-box-v2-grasp" else "sac"
    command = ["bash", str(launcher), mode, "--external-checkpoint-retention"]
    command.extend(("--robot-model", args.robot_model))
    if args.gripper:
        command.extend(("--gripper", args.gripper))
    if args.action_space and args.experiment == "multi-box-v2-grasp":
        parser.error("Multi-box v2 grasp requires its fixed all-joints action contract")
    if args.action_space:
        command.extend(("--action-space", args.action_space))
    if args.experiment == "multi-box-v2-grasp":
        command.append("--self-collision" if args.self_collision else "--no-self-collision")
    for name in ("num_envs", "max_iterations", "rollout_steps", "batch_size", "updates_per_step",
                 "learning_starts", "warmup_vector_steps", "replay_capacity", "replay_device",
                 "save_interval"):
        command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
    if args.experiment == "multi-box-v2-grasp":
        if not 0 <= args.min_alpha <= args.initial_alpha <= 0.1:
            parser.error("V2 SAC requires 0 <= min-alpha <= initial-alpha <= 0.1")
        if not 0 < args.initial_policy_std <= 1 or not 0 < args.reward_scale < 1000 \
                or args.goal_replay_capacity < 1 or not 0 <= args.goal_batch_fraction < 1 \
                or not args.initial_policy_std <= args.max_policy_std <= 1 \
                or args.teacher_pretrain_steps < 0 or not 0 <= args.demo_bc_strength <= 1000:
            parser.error("Invalid V2 SAC recovery configuration")
        if not 0 < args.actor_lr <= 0.001 or args.critic_warmup_updates < 0 \
                or args.success_replay_capacity < 1 \
                or not 0 <= args.success_batch_fraction < 1 \
                or args.success_batch_fraction + args.goal_batch_fraction >= 1:
            parser.error("Invalid V2 critic warmup or success replay")
        for name in ("warmup_action_hold_steps", "warmup_continuous_scale", "min_alpha",
                     "initial_alpha", "initial_policy_std", "max_policy_std", "guided_warmup_mode",
                     "teacher_pretrain_steps",
                     "ik_grasp_goal", "ik_lift_distance_m",
                     "ik_base_clearance_m", "ik_torso_forward_m",
                     "actor_lr", "critic_warmup_updates", "success_replay_capacity", "success_batch_fraction",
                     "reward_scale", "actor_feature_mode",
                     "goal_replay_capacity", "goal_batch_fraction",
                     "demo_batch_fraction", "demo_bc_strength", "demo_decay_fraction", "demo_pretrain_steps",
                     "demo_pretrain_batch_size", "demo_warmup_noise_scale"):
            command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
        command.append("--entropy-backup" if args.entropy_backup else "--no-entropy-backup")
        command.append("--freeze-actor-normalizer" if args.freeze_actor_normalizer
                       else "--no-freeze-actor-normalizer")
        command.append("--demo-guided-warmup" if args.demo_guided_warmup
                       else "--no-demo-guided-warmup")
        command.append("--online-teacher-labels" if args.online_teacher_labels
                       else "--no-online-teacher-labels")
        if args.demo_dataset:
            command.extend(("--demo-dataset", str(args.demo_dataset.resolve())))
    command.extend(("--log-dir", str(parent)))
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint.resolve())))
    (parent / "launch.json").write_text(json.dumps(dict(command=command, remote_root=args.remote_root,
        args=vars(args)), default=str, indent=2))
    budget = GpuBudget(parent, args.gpu, args.gpu_limit_mib, args.gpu_reserve_mib, args.max_seconds, "sac_")
    raise SystemExit(supervise(command, parent, environment,
        lambda run, finished: archive(run, args.remote_root, finished),
        interval=300, run_prefix="sac_", resource_check=budget, require_run_status=True))


if __name__ == "__main__":
    main()
