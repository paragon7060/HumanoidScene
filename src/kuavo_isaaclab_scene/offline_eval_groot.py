#!/usr/bin/env python3
"""Evaluate a GR00T checkpoint against recorded LeRobot Dataset observations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .groot_lerobot_bridge import LeRobotGrootRunner
from .offline_policy_eval import evaluate_offline_frames, write_action_csv


def _episode_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    episodes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not episodes or any(episode < 0 for episode in episodes):
        raise argparse.ArgumentTypeError(
            "episodes must be a comma-separated list of non-negative integers"
        )
    return episodes


def _default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(__file__).resolve().parents[2] / "artifacts" / "offline_eval" / stamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run teacher-forced GR00T inference over a LeRobot Dataset. Action errors are "
            "pipeline regression diagnostics, not closed-loop task-success metrics."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id stored in metadata.")
    parser.add_argument("--episodes", type=_episode_list, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--task", default=None, help="Override every recorded task string.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--actions-per-inference", type=int, default=None)
    parser.add_argument("--video-backend", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive")
    if args.actions_per_inference is not None and args.actions_per_inference <= 0:
        parser.error("--actions-per-inference must be positive")
    return args


def main() -> None:
    args = parse_args()
    try:
        from lerobot.datasets import LeRobotDataset
    except ImportError as exc:
        raise RuntimeError("Install a current LeRobot Dataset v3 environment before offline eval.") from exc

    dataset_root = args.dataset_root.expanduser().resolve()
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=dataset_root,
        episodes=args.episodes,
        video_backend=args.video_backend,
    )
    if len(dataset) == 0:
        raise ValueError(f"dataset has no frames: {dataset_root}")

    first_action = np.asarray(dataset[0]["action"])
    if first_action.ndim != 1:
        raise ValueError(f"expected a one-step action vector, received {first_action.shape}")
    runner = LeRobotGrootRunner.from_pretrained(
        args.checkpoint,
        device=args.device,
        actions_per_inference=args.actions_per_inference,
        local_files_only=args.local_files_only,
        expected_action_dim=int(first_action.shape[0]),
    )
    evaluation = evaluate_offline_frames(
        (dataset[index] for index in range(len(dataset))),
        runner,
        task_override=args.task,
        max_frames=args.max_frames,
    )

    output_dir = (args.output_dir or _default_output_dir()).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "kuavo_offline_policy_eval",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint,
        "dataset_root": str(dataset_root),
        "repo_id": args.repo_id,
        "episodes": args.episodes,
        "task_override": args.task,
        "diagnostic_warning": (
            "Teacher-forced action error validates the inference pipeline; it is not a "
            "closed-loop task-success score."
        ),
        "metrics": evaluation.metrics,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "actions.npz",
        predictions=evaluation.predictions,
        targets=evaluation.targets,
        errors=evaluation.predictions - evaluation.targets,
        episode_indices=evaluation.episode_indices,
        inference_latencies_ms=evaluation.inference_latencies_ms,
    )
    write_action_csv(output_dir / "actions.csv", evaluation)
    print(
        "[RESULT] offline regression "
        f"frames={evaluation.metrics['num_frames']} episodes={evaluation.metrics['episodes_seen']} "
        f"rmse={evaluation.metrics['rmse']:.6f} "
        f"mean_inference_ms={evaluation.metrics['mean_inference_ms']:.2f} "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
