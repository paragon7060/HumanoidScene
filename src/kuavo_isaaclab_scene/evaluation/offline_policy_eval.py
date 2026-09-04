"""Teacher-forced offline regression utilities for LeRobot policy runners."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from .eval_metrics import action_comparison_metrics
from .groot_lerobot_bridge import ACTION_KEY, InferenceSample, PolicyRunner


@dataclass(frozen=True)
class OfflineEvaluation:
    """Arrays and summary produced by an offline dataset pass."""

    predictions: np.ndarray
    targets: np.ndarray
    episode_indices: np.ndarray
    inference_latencies_ms: np.ndarray
    metrics: dict[str, object]


def _scalar_int(value: Any, *, name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must be scalar")
        return int(value.item())
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be scalar")
    return int(array.reshape(-1)[0])


def _batch_feature(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.unsqueeze(0)
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).unsqueeze(0)
    if isinstance(value, (float, int, bool, np.number)):
        return torch.as_tensor(value).reshape(1, 1)
    return value


def policy_observation_from_frame(
    frame: Mapping[str, Any],
    *,
    expected_input_keys: Iterable[str],
    task_override: str | None = None,
) -> dict[str, Any]:
    """Build a batch-of-one observation from a LeRobot Dataset frame."""
    observation: dict[str, Any] = {}
    missing = []
    for key in expected_input_keys:
        if key not in frame:
            missing.append(key)
            continue
        observation[key] = _batch_feature(frame[key])
    if missing:
        raise KeyError(f"dataset frame is missing policy input keys: {tuple(missing)}")

    task = task_override if task_override is not None else frame.get("task")
    if task is None:
        raise KeyError("dataset frame has no task; pass an explicit task override")
    if isinstance(task, (list, tuple)):
        if len(task) != 1:
            raise ValueError("task sequence must contain exactly one item for offline evaluation")
        task = task[0]
    observation["task"] = [str(task)]
    return observation


def _action_vector(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    action = np.asarray(value, dtype=np.float64)
    if action.ndim == 2 and action.shape[0] == 1:
        action = action[0]
    if action.ndim != 1:
        raise ValueError(f"dataset action must have shape (action_dim,), received {action.shape}")
    if not np.isfinite(action).all():
        raise ValueError("dataset action contains NaN or infinity")
    return action


def evaluate_offline_frames(
    frames: Iterable[Mapping[str, Any]],
    runner: PolicyRunner,
    *,
    task_override: str | None = None,
    action_key: str = ACTION_KEY,
    episode_key: str = "episode_index",
    max_frames: int | None = None,
) -> OfflineEvaluation:
    """Run a policy against recorded observations without stepping a simulator."""
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive or None")

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    episode_indices: list[int] = []
    inference_latencies: list[float] = []
    current_episode: int | None = None

    for frame_index, frame in enumerate(frames):
        if max_frames is not None and frame_index >= max_frames:
            break
        if action_key not in frame:
            raise KeyError(f"dataset frame is missing action key {action_key!r}")
        episode_index = _scalar_int(frame.get(episode_key, 0), name=episode_key)
        if current_episode != episode_index:
            runner.reset()
            current_episode = episode_index

        observation = policy_observation_from_frame(
            frame,
            expected_input_keys=runner.expected_input_keys,
            task_override=task_override,
        )
        sample: InferenceSample = runner.select_action(observation)
        predicted = _action_vector(sample.action)
        target = _action_vector(frame[action_key])
        if predicted.shape != target.shape:
            raise ValueError(
                f"predicted action shape {predicted.shape} does not match dataset action "
                f"shape {target.shape} at frame {frame_index}"
            )
        predictions.append(predicted)
        targets.append(target)
        episode_indices.append(episode_index)
        if sample.inferred_new_chunk:
            inference_latencies.append(float(sample.inference_ms))

    if not predictions:
        raise ValueError("offline evaluation received no frames")

    predicted_array = np.stack(predictions)
    target_array = np.stack(targets)
    metrics = action_comparison_metrics(
        predicted_array,
        target_array,
        inference_latencies_ms=inference_latencies,
    )
    metrics["episodes_seen"] = len(set(episode_indices))
    return OfflineEvaluation(
        predictions=predicted_array,
        targets=target_array,
        episode_indices=np.asarray(episode_indices, dtype=np.int64),
        inference_latencies_ms=np.asarray(inference_latencies, dtype=np.float64),
        metrics=metrics,
    )


def write_action_csv(path: Path, evaluation: OfflineEvaluation) -> None:
    """Write one diagnostic row per frame without policy-specific dependencies."""
    path.parent.mkdir(parents=True, exist_ok=True)
    action_dim = evaluation.predictions.shape[1]
    errors = evaluation.predictions - evaluation.targets
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["frame", "episode"]
            + [f"pred_{index}" for index in range(action_dim)]
            + [f"target_{index}" for index in range(action_dim)]
            + [f"error_{index}" for index in range(action_dim)]
            + ["mse", "mae", "rmse"]
        )
        for frame_index, (prediction, target, error, episode) in enumerate(
            zip(
                evaluation.predictions,
                evaluation.targets,
                errors,
                evaluation.episode_indices,
                strict=True,
            )
        ):
            mse = float(np.mean(error**2))
            writer.writerow(
                [frame_index, int(episode)]
                + prediction.tolist()
                + target.tolist()
                + error.tolist()
                + [mse, float(np.mean(np.abs(error))), float(np.sqrt(mse))]
            )
