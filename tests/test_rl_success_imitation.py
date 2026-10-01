"""Preserve measured successful paths inside the existing imitation budget."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import (
    ActorImitationBuffer, AsymmetricReplayBuffer,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.success_imitation import (
    sample_teacher_with_success,
)


def buffers():
    teacher = ActorImitationBuffer(8, 2, 1)
    teacher.add(actor_obs=torch.full((8, 2), 10.), action=torch.full((8, 1), .1))
    success = AsymmetricReplayBuffer(4, 2, 3, 1)
    success.add(**{key: torch.full((4, *value.shape[1:]),
                                 20. if 'obs' in key else .2, dtype=value.dtype)
                   for key, value in success.data.items()})
    return teacher, success


def test_success_fraction_uses_existing_batch_and_only_actor_labels():
    teacher, success = buffers()
    batch, actual = sample_teacher_with_success(teacher, success, 100, 'cpu', .5)
    assert actual == 50
    assert set(batch) == {'actor_obs', 'action'}
    assert batch['action'].shape == (100, 1)
    assert int((batch['actor_obs'][:, 0] == 20).sum()) == 50
    assert torch.all(batch['action'][batch['actor_obs'][:, 0] == 20] == .2)


def test_default_preserves_teacher_sampling_without_success_rows():
    teacher, success = buffers()
    batch, actual = sample_teacher_with_success(teacher, success, 11, 'cpu')
    assert actual == 0 and batch['action'].shape == (11, 1)
    assert torch.all(batch['actor_obs'] == 10)


def test_empty_stratum_falls_back_without_changing_batch_size():
    teacher, success = buffers()
    success.size = 0
    batch, actual = sample_teacher_with_success(teacher, success, 9, 'cpu', .5)
    assert actual == 0 and len(batch['action']) == 9
    teacher.size = 0
    success.size = 4
    batch, actual = sample_teacher_with_success(teacher, success, 9, 'cpu', .5)
    assert actual == 9 and torch.all(batch['actor_obs'] == 20)
    success.size = 0
    with pytest.raises(ValueError, match='No teacher'):
        sample_teacher_with_success(teacher, success, 9, 'cpu', .5)


@pytest.mark.parametrize('count,fraction', [(0, .5), (1, -.1), (1, 1.1)])
def test_invalid_sampling_budget_is_rejected(count, fraction):
    teacher, success = buffers()
    with pytest.raises(ValueError):
        sample_teacher_with_success(teacher, success, count, 'cpu', fraction)
