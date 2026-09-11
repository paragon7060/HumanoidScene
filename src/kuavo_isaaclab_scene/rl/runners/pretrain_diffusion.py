"""Offline diffusion BC; uses the existing torch environment, without Isaac startup."""

import argparse
import json
import os
from pathlib import Path
import torch
from ..algorithms.diffusion import DiffusionPolicy, DiffusionConfig
from ..algorithms.common import optimize
from ..data.episodes import EpisodeDataset
from .storage import save_checkpoint, log_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--denoising-steps", type=int, default=20)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--min-std", type=float, default=.1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.save_interval, args.keep_checkpoints) < 1:
        parser.error("Counts must be positive")
    if args.device != "cpu" and (args.device != "cuda:0" or not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit()):
        parser.error("Use CUDA_VISIBLE_DEVICES=<one physical index> with --device cuda:0")
    if not 0 < args.lr < 1:
        parser.error("Learning rate must be in (0, 1)")
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    dataset = EpisodeDataset(args.dataset, args.horizon)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "manifest.json").write_text(json.dumps(dataset.manifest, indent=2))
        (args.output / "pretrain.json").write_text(json.dumps(vars(args), default=str, indent=2))
        policy = DiffusionPolicy(dataset.obs_dim, dataset.action_dim,
                                 DiffusionConfig(args.horizon, args.denoising_steps, args.hidden, args.min_std)).to(args.device)
        dataset.fit_normalizer(policy.normalizer)
        optimizer = torch.optim.Adam(policy.network.parameters(), lr=args.lr)
        for step in range(1, args.steps + 1):
            examples = [dataset[i] for i in torch.randint(len(dataset), (args.batch_size,)).tolist()]
            obs = torch.stack([item[0] for item in examples]).to(args.device)
            actions = torch.stack([item[1] for item in examples]).to(args.device)
            loss = policy.loss(obs, actions)
            optimize(optimizer, loss, policy.network.parameters())
            if step == 1 or step % 100 == 0 or step == args.steps:
                log_metrics(args.output, step, {"noise_mse": loss.item(), "dataset_chunks": len(dataset)})
            if step % args.save_interval == 0 or step == args.steps:
                state = policy.checkpoint()
                state["optimizer"] = optimizer.state_dict()
                save_checkpoint(args.output, state, step, args.keep_checkpoints)
    finally:
        dataset.close()


if __name__ == "__main__":
    main()
