"""Dependency-light metrics shared by online and offline policy evaluation."""

from __future__ import annotations

from collections import Counter
import math
from typing import Iterable

import numpy as np


def control_decimation(physics_dt: float, control_hz: float) -> int:
    """Return an integer simulator decimation for an exact control rate."""
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("physics_dt must be finite and positive")
    if not math.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")

    physics_hz = 1.0 / physics_dt
    decimation = round(physics_hz / control_hz)
    if decimation < 1:
        raise ValueError(
            f"control_hz={control_hz:g} exceeds physics_hz={physics_hz:g}"
        )
    actual_hz = physics_hz / decimation
    if not math.isclose(actual_hz, control_hz, rel_tol=1.0e-9, abs_tol=1.0e-9):
        raise ValueError(
            f"control_hz={control_hz:g} cannot be represented exactly by "
            f"physics_hz={physics_hz:g}; choose a rate with an integer decimation"
        )
    return decimation


def percentile_nearest_rank(values: Iterable[float], percentile: float) -> float:
    """Return a deterministic nearest-rank percentile for a finite sample."""
    sample = sorted(float(value) for value in values)
    if not sample:
        return 0.0
    if not 0.0 <= percentile <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    rank = max(1, math.ceil((percentile / 100.0) * len(sample)))
    return sample[min(rank, len(sample)) - 1]


def wilson_score_interval(successes: int, trials: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    """Compute a two-sided Wilson interval for a Bernoulli success rate."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("expected 0 <= successes <= trials")
    if trials == 0:
        return 0.0, 0.0
    proportion = successes / trials
    z_squared = z * z
    denominator = 1.0 + z_squared / trials
    center = (proportion + z_squared / (2.0 * trials)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z_squared / (4.0 * trials * trials)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def termination_reason_counts(reasons: Iterable[str]) -> dict[str, int]:
    """Return stable, JSON-ready episode termination counts."""
    return dict(sorted(Counter(str(reason) for reason in reasons).items()))


def action_comparison_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    *,
    inference_latencies_ms: Iterable[float] = (),
) -> dict[str, object]:
    """Summarize teacher-forced action differences for pipeline regression.

    These metrics diagnose schema, preprocessing, normalization, and checkpoint
    regressions.  They are not a substitute for closed-loop task success because
    more than one action can be valid for a manipulation observation.
    """
    predicted = np.asarray(predictions, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    if predicted.ndim != 2 or target.ndim != 2:
        raise ValueError("predictions and targets must both have shape (frames, action_dim)")
    if predicted.shape != target.shape:
        raise ValueError(
            f"prediction shape {predicted.shape} does not match target shape {target.shape}"
        )
    if predicted.shape[0] == 0:
        raise ValueError("at least one action frame is required")
    if not np.isfinite(predicted).all() or not np.isfinite(target).all():
        raise ValueError("predictions and targets must contain only finite values")

    error = predicted - target
    squared = error**2
    absolute = np.abs(error)
    latencies = [float(value) for value in inference_latencies_ms]
    if any(not math.isfinite(value) or value < 0.0 for value in latencies):
        raise ValueError("inference latencies must be finite and non-negative")

    per_dimension = []
    for index in range(predicted.shape[1]):
        dimension_squared = squared[:, index]
        dimension_absolute = absolute[:, index]
        per_dimension.append(
            {
                "index": index,
                "mse": float(np.mean(dimension_squared)),
                "mae": float(np.mean(dimension_absolute)),
                "rmse": float(math.sqrt(np.mean(dimension_squared))),
                "max_abs_error": float(np.max(dimension_absolute)),
            }
        )

    return {
        "num_frames": int(predicted.shape[0]),
        "action_dim": int(predicted.shape[1]),
        "mse": float(np.mean(squared)),
        "mae": float(np.mean(absolute)),
        "rmse": float(math.sqrt(np.mean(squared))),
        "max_abs_error": float(np.max(absolute)),
        "policy_inferences": len(latencies),
        "mean_inference_ms": float(np.mean(latencies)) if latencies else 0.0,
        "p95_inference_ms": percentile_nearest_rank(latencies, 95.0),
        "per_dimension": per_dimension,
    }
