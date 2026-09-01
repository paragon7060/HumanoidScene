import numpy as np
import pytest

from kuavo_isaaclab_scene.eval_metrics import (
    action_comparison_metrics,
    percentile_nearest_rank,
    termination_reason_counts,
    wilson_score_interval,
)


def test_wilson_interval_contains_observed_rate() -> None:
    low, high = wilson_score_interval(8, 10)
    assert low < 0.8 < high
    assert wilson_score_interval(0, 0) == (0.0, 0.0)


def test_percentile_and_reason_counts_are_deterministic() -> None:
    assert percentile_nearest_rank([5.0, 1.0, 3.0, 2.0], 95.0) == 5.0
    assert termination_reason_counts(["timeout", "success", "timeout"]) == {
        "success": 1,
        "timeout": 2,
    }


def test_action_metrics_include_per_dimension_errors() -> None:
    metrics = action_comparison_metrics(
        np.asarray([[1.0, 2.0], [2.0, 4.0]]),
        np.asarray([[0.0, 2.0], [2.0, 2.0]]),
        inference_latencies_ms=[10.0, 30.0],
    )
    assert metrics["mse"] == pytest.approx(1.25)
    assert metrics["mae"] == pytest.approx(0.75)
    assert metrics["p95_inference_ms"] == 30.0
    assert metrics["per_dimension"][0]["mse"] == pytest.approx(0.5)
