"""Actual optimizer rollback, regional protection, and ordinary SAC behavior."""
from copy import deepcopy
from dataclasses import replace

import torch

from test_rl_upright_contact_sac import agent_fixture
from kuavo_isaaclab_scene.rl.algorithms.common import optimize
from kuavo_isaaclab_scene.rl.algorithms.success_update_guard import (
    VARIANT, success_update_guard_config, guarded_actor_step, acceptable)


def fixture():
    raw, _, agent = agent_fixture()
    raw[:, 94:98] = 0.; raw[:4, 94] = 1.; raw[4:, 95] = 1.
    agent.config = replace(agent.config, freeze_actor_normalizer=True)
    agent.actor_success_guard_config = success_update_guard_config(VARIANT)
    with torch.no_grad():
        agent.actor.network[-1].weight.zero_()
        agent.actor.network[-1].bias[:21].zero_()
    labels = agent.act(raw, True)
    assert labels[:, 19:].eq(-1).all()
    return agent, dict(actor_obs=raw, action=labels)


def equal(a, b):
    if isinstance(a, torch.Tensor): return torch.equal(a, b)
    if isinstance(a, dict): return a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)): return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def test_wrong_binary_flip_restores_mature_Adam_parameters_and_moments_exactly():
    a, success = fixture()
    # Exercise the exact binary check even with no useful smooth jaw gradient.
    a.successful_jaw_loss = lambda raw, logits, labels: (logits.new_zeros(()), {})
    a.action_projector.entropy_mask = lambda raw: raw.new_zeros(len(raw), 21)
    for _ in range(5):
        optimize(a.actor_optimizer, -a.actor.network[-1].bias[19], a.actor.parameters())
    with torch.no_grad(): a.actor.network[-1].bias[19] = 0.
    original = deepcopy(a.actor.state_dict()), deepcopy(a.actor_optimizer.state_dict())
    accepted, report = guarded_actor_step(a, -a.actor.network[-1].bias[19], success)
    assert not accepted and report['actor_success_guard_parameter_scale'] == 0.
    assert report['actor_success_guard_regions'] == 2
    assert equal(original[0], a.actor.state_dict()) and equal(original[1], a.actor_optimizer.state_dict())


def test_useful_success_update_is_accepted_and_can_restore_with_its_own_contract():
    a, success = fixture()
    before = a.actor.network[-1].bias.detach().clone()
    accepted, report = guarded_actor_step(a, a.actor.network[-1].bias[19], success)
    assert accepted and report['actor_success_guard_parameter_scale'] == 1.
    assert a.actor.network[-1].bias[19] < before[19]
    restored, _ = fixture(); restored.restore(a.checkpoint())
    assert equal(a.actor.state_dict(), restored.actor.state_dict())
    assert equal(a.actor_optimizer.state_dict(), restored.actor_optimizer.state_dict())


def test_one_regions_improvement_cannot_pay_for_another_regions_regression():
    config = success_update_guard_config(VARIANT)
    before = {'0': dict(body=1., jaw=1., jaw_errors=0), '1': dict(body=1., jaw=1., jaw_errors=0)}
    after = {'0': dict(body=.1, jaw=.1, jaw_errors=0), '1': dict(body=1.1, jaw=1., jaw_errors=0)}
    assert not acceptable(before, after, config)
    after['1'] = dict(body=.9, jaw=1.1, jaw_errors=0)
    assert not acceptable(before, after, config)


def test_missing_actual_success_does_not_freeze_new_regions():
    a, _ = fixture()
    before = a.actor.network[-1].bias.detach().clone()
    accepted, report = guarded_actor_step(a, -a.actor.network[-1].bias[1], None)
    assert accepted and not report and a.actor.network[-1].bias[1] > before[1]


def test_rejected_real_hybrid_update_still_updates_critic_and_skips_temperatures():
    a, success = fixture(); raw = success['actor_obs']
    a.successful_jaw_loss = lambda raw, logits, labels: (logits.new_zeros(()), {})
    a.action_projector.entropy_mask = lambda raw: raw.new_zeros(len(raw), 21)
    # Force an Adam direction that flips an initially correct open jaw.
    for _ in range(5): optimize(a.actor_optimizer, -a.actor.network[-1].bias[19], a.actor.parameters())
    with torch.no_grad(): a.actor.network[-1].bias[19] = 0.
    a.actor_jaw_regularization = lambda logits, near: (-100. * logits[:, 0].mean(), {})
    before = deepcopy(a.actor.state_dict()), deepcopy(a.actor_optimizer.state_dict())
    q = deepcopy(a.q1.state_dict()); temperatures = [p.detach().clone() for p in (a.log_alpha, a.log_alpha_discrete)]
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=success['action'],
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8, 4), reward=torch.ones(8),
        terminated=torch.ones(8, dtype=torch.bool))
    report = a.update(batch, successful_train=success, success_goal_weight=2., success_jaw_weight=.05)
    assert not report['actor_updated'] and not report['actor_success_guard_accepted']
    assert equal(before[0], a.actor.state_dict()) and equal(before[1], a.actor_optimizer.state_dict())
    assert not equal(q, a.q1.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(temperatures, (a.log_alpha, a.log_alpha_discrete)))


def test_projecting_actual_Adam_displacement_preserves_success_instead_of_only_shrinking_it():
    torch.manual_seed(0)
    a, success = fixture()
    accepted, report = guarded_actor_step(a, -a.actor.network[-1].bias[19], success)
    assert accepted and report['actor_success_guard_projected_constraints'] > 0
    assert report['actor_success_guard_parameter_scale'] == 1.
    assert report['actor_success_guard_after']['0']['jaw_errors'] == 0
    assert report['actor_success_guard_after']['1']['jaw_errors'] == 0
    assert torch.equal(a.act(success['actor_obs'], True), success['action'])
