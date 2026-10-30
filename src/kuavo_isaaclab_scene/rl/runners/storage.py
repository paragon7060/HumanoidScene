"""Small atomic checkpoints, contract validation and append-only scalar logs."""

import json
from pathlib import Path
import torch


CONTRACT_KEYS = ("contract_hash", "actions", "action_config", "observations")


def check_contract(source, target):
    for key in CONTRACT_KEYS:
        if key not in source or key not in target or json.loads(json.dumps(source[key])) != json.loads(json.dumps(target[key])):
            raise ValueError(f"Dataset/checkpoint {key} differs from the current environment")
    if source["task"]["name"] != target["task"]["name"]:
        raise ValueError("Dataset/checkpoint belongs to another task")


def save_checkpoint(directory, state, iteration, keep=2):
    if keep < 1:
        raise ValueError("Checkpoint retention must be positive")
    directory = Path(directory)
    destination = directory / f"checkpoint_{iteration:08d}.pt"
    temporary = destination.with_suffix(".tmp")
    state = dict(state, iteration=iteration, format_version=1)
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    for obsolete in sorted(directory.glob("checkpoint_*.pt"))[:-keep]:
        obsolete.unlink()
    return destination


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Expected native version-1 SAC/diffusion/DPPO checkpoint")
    return state


def log_metrics(directory, iteration, metrics):
    row = dict(iteration=iteration, **metrics)
    line = json.dumps(row, allow_nan=False)
    with (Path(directory) / "metrics.jsonl").open("a") as stream:
        stream.write(line + "\n")
    print(f"[RL] {line}", flush=True)


class EpisodeMetrics:
    def __init__(self):
        self.count = self.successes = self.unsafe = 0

    def record(self, env):
        latest = getattr(env, "_rl_last_outcomes", {})
        if latest.get("step") == getattr(env, "common_step_counter", None):
            for episode in latest.get("episodes", []):
                self.count += 1
                self.successes += int(episode["success"])
                self.unsafe += int(episode["unsafe"])

    def report(self):
        return dict(episode_success_rate=self.successes / self.count if self.count else None,
                    episode_unsafe_rate=self.unsafe / self.count if self.count else None)
