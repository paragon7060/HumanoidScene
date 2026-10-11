"""Restored SAC learners must not advance each other's saved Adam state."""
from copy import deepcopy
from io import BytesIO
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.sac import SAC, SACConfig
from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
from kuavo_isaaclab_scene.rl.algorithms.hybrid_goal_sac import HybridGoalSAC
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalProjector


def equal_state(before, after):
    if torch.is_tensor(before):
        assert torch.equal(before, after)
    elif isinstance(before, dict):
        assert before.keys() == after.keys()
        for key in before:
            equal_state(before[key], after[key])
    elif isinstance(before, (list, tuple)):
        assert len(before) == len(after)
        for left, right in zip(before, after):
            equal_state(left, right)
    else:
        assert before == after


def fixture(kind):
    config = SACConfig(hidden=8, freeze_actor_normalizer=True, entropy_backup=False)
    if kind == 'continuous':
        make = lambda: SAC(4, 2, config)
        batch = dict(obs=torch.randn(8, 4), next_obs=torch.randn(8, 4), action=torch.zeros(8, 2))
    elif kind == 'asymmetric':
        make = lambda: AsymmetricSAC(4, 7, 2, config)
        batch = dict(actor_obs=torch.randn(8, 4), next_actor_obs=torch.randn(8, 4),
                     critic_obs=torch.randn(8, 7), next_critic_obs=torch.randn(8, 7), action=torch.zeros(8, 2))
    else:
        class Prior:
            def act(self, raw, deterministic=True):
                return raw.new_zeros(len(raw), 24)
        def make():
            return HybridGoalSAC(480, 4, 21, config,
                action_projector=StagedGoalProjector(SimpleNamespace(agent=Prior()), 474, True))
        obs = torch.zeros(8, 480); obs[:, 144] = 1; obs[:, -1] = .05
        actions = torch.zeros(8, 21); actions[:, 19:] = -1
        batch = dict(actor_obs=obs, next_actor_obs=obs.clone(), critic_obs=torch.zeros(8, 4),
                     next_critic_obs=torch.zeros(8, 4), action=actions)
    batch.update(reward=torch.linspace(-1, 1, 8), terminated=torch.ones(8, dtype=torch.bool))
    return make, batch


@pytest.mark.parametrize('kind', ['continuous', 'asymmetric', 'hybrid'])
def test_two_restored_learners_keep_adam_checkpoint_and_each_other_independent(kind):
    torch.manual_seed(104)
    make, batch = fixture(kind)
    source = make(); source.update(batch)
    # Match a loaded CPU checkpoint, rather than live state_dict views.
    serialized = BytesIO(); torch.save(source.checkpoint(), serialized); serialized.seek(0)
    saved = torch.load(serialized, map_location='cpu', weights_only=True)
    before = deepcopy(saved)
    first, second = make(), make()
    first.restore(saved); second.restore(saved)
    torch.manual_seed(117); first.update(batch)
    assert any(not torch.equal(value, before['model'][key]) for key, value in first.state_dict().items())
    equal_state(saved, before)
    equal_state([o.state_dict() for o in second.optimizers], before['optimizers'])
    fresh = make(); fresh.restore(deepcopy(before))
    torch.manual_seed(127); second.update(batch)
    torch.manual_seed(127); fresh.update(batch)
    equal_state(second.state_dict(), fresh.state_dict())
    equal_state([o.state_dict() for o in second.optimizers], [o.state_dict() for o in fresh.optimizers])
