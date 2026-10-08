"""Closed analyses must use the box pose captured by the live held controller."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from export_eval_q_videos import recorded_held_box_anchor


def test_anchor_follows_first_held_measurement_after_box_moves_during_approach():
    raw = np.zeros((4, 464), dtype=np.float32)
    raw[0, :2] = [1., 2.]
    raw[2, :2] = [1.07, 2.05]
    calls = []

    def measured_anchor(value):
        calls.append(value.clone())
        return value[:, :2].clone()

    prior = SimpleNamespace(coordinates=SimpleNamespace(box_anchor=measured_anchor))
    anchor = recorded_held_box_anchor(prior, {'actor_obs': raw}, 2)
    assert torch.equal(calls[0], torch.from_numpy(raw[2])[None])
    assert torch.equal(anchor, torch.tensor([[1.07, 2.05]]))
    assert not torch.equal(anchor, torch.from_numpy(raw[0, :2])[None])


@pytest.mark.parametrize('start', [-1, 4, True])
def test_anchor_rejects_missing_or_ambiguous_first_held_identity(start):
    with pytest.raises(ValueError):
        recorded_held_box_anchor(None, {'actor_obs': np.zeros((4, 464), np.float32)}, start)
