"""Audits must query the actual state-dependent sampler without changing it."""
from copy import deepcopy

import pytest
import torch

from audit_closed_train_servo import diagnose
from audit_servo_actor_gradients import fractions
from test_rl_servo_critic import decoder_fixture
from test_rl_urdf_full_arm_sac import full
from test_rl_urdf_servo_guard_sac import guard


@pytest.mark.parametrize('builder', [guard, full])
def test_audits_use_actual_body_mean_noise_and_offsets_without_mutating_policy(builder):
    raw, encoder = decoder_fixture(8)
    agent = builder(encoder)
    anchor = raw.new_zeros(8, 19)
    # Arms have far absolute targets and a genuinely clipped production servo.
    # Treating the full-support model's residual mean0 as its Gaussian mean
    # would instead synthesize zero targets and a false nonzero Jacobian.
    anchor[:, 1:15] = .7
    agent.executed_body_anchor = lambda measured: anchor
    with torch.no_grad():
        agent.actor.network[-1].weight[:19].zero_()
        agent.actor.network[-1].bias[:19].zero_()
    labels = agent.act(raw, True)
    measured_command = agent.critic_action_features(raw, labels)
    saved = deepcopy(agent.state_dict())
    rng = torch.get_rng_state().clone()
    calls = []
    original = agent.continuous_sample

    def observe(normalized, **kwargs):
        assert kwargs['raw'] is raw
        calls.append(kwargs.copy())
        return original(normalized, **kwargs)

    agent.continuous_sample = observe
    result = diagnose(raw, measured_command, agent, torch.Generator().manual_seed(208))
    assert result['deterministic_body_mean_matches_actual_sampler']
    assert result['noise_and_episode_bias_use_actual_continuous_sampler']
    assert result['arms']['zero_servo_gradient_active_fraction'] == 1
    assert sum(x.get('body_latent_offset') is not None for x in calls) == 8
    assert sum(x.get('noise') is not None for x in calls) == 8
    result = fractions(raw, agent, labels, torch.Generator().manual_seed(209))
    assert result['deterministic_body_mean_matches_actual_sampler']
    assert result['noise_uses_actual_continuous_sampler']
    assert result['all_jacobians_finite']
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(v, agent.state_dict()[k]) for k, v in saved.items())
