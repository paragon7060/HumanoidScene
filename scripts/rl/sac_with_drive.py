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
    parser.add_argument("--min-alpha", type=float, default=0.01)
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
    if args.replay_capacity < args.num_envs:
        parser.error("Replay capacity must hold a full vector step")
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
        for name in ("warmup_action_hold_steps", "min_alpha"):
            command.extend(("--" + name.replace("_", "-"), str(getattr(args, name))))
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
