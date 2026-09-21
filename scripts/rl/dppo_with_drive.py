#!/usr/bin/env python3
"""One GPU-isolated DPPO experiment with verified Drive retention and final logs."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from uuid import uuid4

from drive_backup import ROOT
from train_with_drive import archive, supervise, add_action_space_argument


from gpu_budget import GpuBudget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_action_space_argument(parser)
    parser.add_argument("--experiment-dir", type=Path,
                        help="New unique parent; defaults to timestamp+UUID under artifacts/rl/drive_runs")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--smoke-test", action="store_true", help="Random initialization, wiring test only")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=6)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--critic-warmup", type=int, default=0)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--remote-root", default=os.environ.get(
        "RL_DRIVE_REMOTE_ROOT", "gdrive:HumanoidScene-RL"))
    parser.add_argument("--gpu-limit-mib", type=int, default=34816)
    parser.add_argument("--gpu-reserve-mib", type=int, default=8192)
    parser.add_argument("--max-seconds", type=int, default=3600)
    args = parser.parse_args()
    if (min(args.num_envs, args.max_iterations, args.rollout_steps, args.batch_size, args.epochs,
            args.save_interval, args.gpu_limit_mib, args.max_seconds) < 1
            or min(args.gpu, args.gpu_reserve_mib, args.critic_warmup) < 0):
        parser.error("Invalid counts or resource budget")
    if args.smoke_test and args.max_iterations > 10:
        parser.error("Random smoke tests are limited to 10 iterations")
    if args.checkpoint and not args.checkpoint.is_file():
        parser.error("Missing checkpoint")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parent = (args.experiment_dir or ROOT / "artifacts/rl/drive_runs" /
              f"dppo_gpu{args.gpu}_{stamp}_{uuid4().hex[:8]}").resolve()
    parent.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    environment.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMNI_KIT_ACCEPT_EULA="YES", OMP_NUM_THREADS="8",
                       ISAACLAB_PYTHON=str(Path.home() / "miniconda3/envs/env_isaaclab_232/bin/python"),
                       PYTHONPATH=str(ROOT / "src"))
    command = ["bash", str(ROOT / "scripts/rl/flap_pick.sh"), "dppo", "--external-checkpoint-retention",
               "--num-envs", str(args.num_envs), "--max-iterations", str(args.max_iterations),
               "--rollout-steps", str(args.rollout_steps), "--batch-size", str(args.batch_size),
               "--epochs", str(args.epochs), "--critic-warmup", str(args.critic_warmup),
               "--save-interval", str(args.save_interval), "--log-dir", str(parent)]
    command += ["--smoke-test"] if args.smoke_test else ["--checkpoint", str(args.checkpoint.resolve())]
    if args.action_space:
        command.extend(("--action-space", args.action_space))
    (parent / "launch.json").write_text(json.dumps(dict(command=command, remote_root=args.remote_root,
        args=vars(args)), default=str, indent=2))
    budget = GpuBudget(parent, args.gpu, args.gpu_limit_mib, args.gpu_reserve_mib, args.max_seconds)
    exit_code = supervise(command, parent, environment,
                         lambda run, finished: archive(run, args.remote_root, finished),
                         interval=300, run_prefix="dppo_", resource_check=budget, require_run_status=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
