"""Prepared jaw correction: fixed bounds and untouched body/std/Adam."""
from copy import deepcopy

import pytest
import torch

from test_rl_regional_guard_metric_cache import regional_fixture
from test_rl_region_size_guard import size_fixture
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_output_repair import jaw_output_correction


def failed_jaw(factory):
    agent = factory()
    batch, groups = cohort.cohort_batch(agent)
    with torch.no_grad():
        agent.actor.network.heads[0][-1].bias[19] += .001
    values, correct = cohort.metrics(agent, batch, groups)
    assert not cohort.within_ceilings(values, correct, agent.success_guard_memory,
                                     agent.actor_success_guard_config)
    return agent, batch, groups


@pytest.mark.parametrize('factory', [regional_fixture, size_fixture])
def test_jaw_repair_preserves_body_std_hidden_optimizer_memory_and_RNG(factory):
    agent, batch, groups = failed_jaw(factory)
    head = agent.actor.network.heads[0]
    parameters = list(head.parameters())
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.actor_optimizer.state_dict()), deepcopy(agent.success_guard_memory)
    rng = torch.get_rng_state().clone()
    delta, kinds = jaw_output_correction(agent, batch, groups, parameters)
    assert delta is not None and kinds['body'] == 0 and kinds['jaw_label'] > 0
    assert equal(before[0], agent.actor.state_dict())
    assert equal(before[1], agent.actor_optimizer.state_dict())
    assert equal(before[2], agent.success_guard_memory) and torch.equal(rng, torch.get_rng_state())
    assert torch.isfinite(delta).all() and delta.norm() > 0
    offset = 0
    for p in parameters:
        part = delta[offset:offset+p.numel()].reshape_as(p); offset += p.numel()
        if p is head[-1].weight or p is head[-1].bias:
            assert not part[:19].any() and not part[21:].any()
        else:
            assert not part.any()
    with torch.no_grad():
        offset = 0
        for p in parameters:
            p.add_(delta[offset:offset+p.numel()].reshape_as(p)); offset += p.numel()
    values, correct = cohort.metrics(agent, batch, groups)
    assert cohort.within_ceilings(values, correct, agent.success_guard_memory,
                                 agent.actor_success_guard_config)
    assert equal(before[1], agent.actor_optimizer.state_dict())


def test_actual_body_violation_forbids_jaw_only_repair():
    agent, batch, groups = failed_jaw(size_fixture)
    with torch.no_grad(): agent.actor.network.heads[0][-1].bias[1] += 1.
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.success_guard_memory)
    delta, kinds = jaw_output_correction(agent, batch, groups, list(agent.actor.network.heads[0].parameters()))
    assert delta is None and kinds['body'] > 0
    assert equal(before[0], agent.actor.state_dict()) and equal(before[1], agent.success_guard_memory)


def test_valid_candidate_has_no_jaw_correction():
    agent = size_fixture(); batch, groups = cohort.cohort_batch(agent)
    delta, kinds = jaw_output_correction(agent, batch, groups, list(agent.actor.network.heads[0].parameters()))
    assert delta is None and kinds == dict(body=0, jaw_loss=0, jaw_label=0)


def test_partial_or_two_heads_are_rejected():
    agent, batch, groups = failed_jaw(size_fixture)
    first, other = [list(h.parameters()) for h in agent.actor.network.heads[:2]]
    for parameters in (first[:-1], first+other, first+[first[0]]):
        with pytest.raises(ValueError, match='complete independent head'):
            jaw_output_correction(agent, batch, groups, parameters)
