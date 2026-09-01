import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kuavo_isaaclab_scene.groot_lerobot_bridge import InferenceSample
from kuavo_isaaclab_scene.offline_policy_eval import (
    evaluate_offline_frames,
    policy_observation_from_frame,
)


class _FakeRunner:
    expected_input_keys = ("observation.state", "observation.images.head")
    output_action_dim = 2

    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def select_action(self, observation):
        assert observation["observation.state"].shape == (1, 2)
        assert observation["observation.images.head"].shape == (1, 3, 2, 2)
        action = observation["observation.state"] + 1.0
        return InferenceSample(action, True, 12.0)


def _frame(episode: int, state: tuple[float, float], action: tuple[float, float]):
    return {
        "episode_index": torch.tensor(episode),
        "observation.state": torch.tensor(state),
        "observation.images.head": torch.zeros((3, 2, 2)),
        "task": "move boxes",
        "action": torch.tensor(action),
    }


def test_policy_observation_fails_early_on_schema_mismatch() -> None:
    with pytest.raises(KeyError, match="missing policy input"):
        policy_observation_from_frame(
            {"observation.state": torch.zeros(2), "task": "x"},
            expected_input_keys=("observation.state", "observation.images.head"),
        )


def test_offline_eval_resets_each_episode_and_compares_actions() -> None:
    runner = _FakeRunner()
    result = evaluate_offline_frames(
        [
            _frame(0, (0.0, 1.0), (1.0, 2.0)),
            _frame(0, (1.0, 2.0), (2.0, 2.0)),
            _frame(1, (2.0, 3.0), (3.0, 4.0)),
        ],
        runner,
    )
    assert runner.resets == 2
    assert result.metrics["episodes_seen"] == 2
    assert result.metrics["num_frames"] == 3
    assert result.metrics["mae"] == pytest.approx(1.0 / 6.0)
    assert np.array_equal(result.episode_indices, np.asarray([0, 0, 1]))
