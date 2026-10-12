"""Size isolation must preserve original ceilings, jaw checks and Adam rollback."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_rl_regional_guard_metric_cache import regional_fixture
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms import success_update_guard as guard
from kuavo_isaaclab_scene.rl.algorithms.common import optimize
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_size_actor import (
    SCOPES, SOURCE_HEADS, split_region_size_actor_and_Adam,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.independent_actor_heads import (
    head_labels, parameter_partitions,
)


def size_fixture():
    agent = regional_fixture()
    episodes = {}
    for old in agent.success_guard_memory['entries']:
        item = deepcopy(old)
        region = item['outcome']['layout']['target_region']
        size = 'medium' if region == 'shelf_2_right' else 'small'
        item['outcome']['layout']['target_box_type'] = size
        raw = item['actor_obs']
        raw[:, 89:91] = raw.new_tensor([0., 1.] if size == 'medium' else [1., 0.])
        raw[:, 91:94] = raw.new_tensor([.32, .22, .185] if size == 'medium' else [.266, .185, .13])
        item['rows'] = dict(actor_obs=raw,
                           action=torch.cat([agent.act(raw[:8], True), agent.act(raw[8:], True)]).detach())
        episodes[region] = [item]
    agent.success_guard_memory = cohort.empty_memory()
    cohort.synchronize(agent, SimpleNamespace(episodes=episodes))
    split_region_size_actor_and_Adam(agent)
    return agent


def test_Adam_migration_clones_mature_steps_and_moments_without_aliases_or_other_changes():
    agent = regional_fixture()
    source, optimizer = agent.actor.network, agent.actor_optimizer
    for i, p in enumerate(source.parameters()):
        optimizer.state[p] = dict(step=torch.tensor(float(i+7)),
                                  exp_avg=torch.full_like(p, .001*(i+1)),
                                  exp_avg_sq=torch.full_like(p, .002*(i+1)))
    before = deepcopy(agent.checkpoint())
    memory = deepcopy(agent.success_guard_memory)
    split_region_size_actor_and_Adam(agent)
    after = agent.checkpoint()
    for k, v in before['model'].items():
        if not k.startswith('actor.network.'):
            assert equal(v, after['model'][k])
    assert equal(before['optimizers'][1:], after['optimizers'][1:])
    assert equal(memory, agent.success_guard_memory)
    for i, source_index in enumerate(SOURCE_HEADS):
        for new, old in zip(agent.actor.network.heads[i].parameters(), source.heads[source_index].parameters()):
            assert torch.equal(new, old) and new.data_ptr() != old.data_ptr()
            assert equal(optimizer.state[old], agent.actor_optimizer.state[new])
            for key in optimizer.state[old]:
                assert optimizer.state[old][key].data_ptr() != agent.actor_optimizer.state[new][key].data_ptr()
    # Even cloned heads' CPU step counters must have independent storage.
    for first, clone in ((0, 4), (1, 5)):
        for a, b in zip(agent.actor.network.heads[first].parameters(), agent.actor.network.heads[clone].parameters()):
            assert agent.actor_optimizer.state[a]['step'].data_ptr() != agent.actor_optimizer.state[b]['step'].data_ptr()
    optimize(agent.actor_optimizer, -agent.actor.network.heads[4][-1].bias[1], agent.actor.parameters())
    assert equal(before['optimizers'][0], optimizer.state_dict())
    assert all(torch.equal(v, source.state_dict()[k.removeprefix('actor.network.')])
               for k, v in before['model'].items() if k.startswith('actor.network.'))


@pytest.mark.parametrize('fault', ('group', 'normalizer', 'nonfinite'))
def test_failed_migration_does_not_partly_install_actor_or_Adam(fault):
    agent = regional_fixture()
    if fault == 'group': agent.actor_optimizer.param_groups[0]['params'].pop()
    elif fault == 'normalizer': agent.config = replace(agent.config, freeze_actor_normalizer=False)
    else:
        with torch.no_grad(): next(agent.actor.parameters()).flatten()[0] = float('nan')
    actor, optimizer = agent.actor.network, agent.actor_optimizer
    with pytest.raises(ValueError): split_region_size_actor_and_Adam(agent)
    assert agent.actor.network is actor and agent.actor_optimizer is optimizer


def test_size_cache_matches_whole_cohort_after_commits_and_unprotected_size_changes():
    agent = size_fixture()
    batch, groups = cohort.cohort_batch(agent)
    values, correct = cohort.metrics(agent, batch, groups)
    cache = cohort.RegionalMetricCache(agent, batch, groups, values, correct)
    assert len(cache.groups) == 6 and not cache.groups['shelf_2_right/small']
    for index in (4, 0, 1, 5):
        with torch.no_grad(): agent.actor.network.heads[index][-1].bias[1] += .00001
        candidate, jaws = cache.evaluate('/'.join(SCOPES[index]))
        full, full_jaws = cohort.metrics(agent, batch, groups)
        assert equal(candidate, full) and torch.equal(jaws, full_jaws)
        cache.commit('/'.join(SCOPES[index]))


def test_rejected_medium_partition_restores_mature_Adam_while_other_size_can_update(monkeypatch):
    agent = size_fixture()
    agent.action_projector.entropy_mask = lambda raw: raw.new_zeros(len(raw), 21)
    protected, small = agent.actor.network.heads[0], agent.actor.network.heads[4]
    for _ in range(5):
        optimize(agent.actor_optimizer, -protected[-1].bias[19], agent.actor.parameters())
    with torch.no_grad(): protected[-1].bias[19] = 0.
    before = deepcopy(protected.state_dict()), deepcopy(agent.actor_optimizer.state_dict())
    moments = {p: deepcopy(agent.actor_optimizer.state.get(p)) for p in protected.parameters()}
    limits = deepcopy(agent.success_guard_memory['ceilings'])
    other = small[-1].bias.detach().clone()
    # Simulate a protected head for which no candidate repair is feasible.
    # Keep the real ceiling/jaw checks as well, including the final whole check.
    original = cohort.within_ceilings
    monkeypatch.setattr(cohort, 'within_ceilings', lambda *args:
                        equal(before[0], protected.state_dict()) and original(*args))
    accepted, report = guard.guarded_actor_step(agent, -protected[-1].bias[19]-small[-1].bias[1], None)
    assert accepted
    scales = report['actor_success_guard_region_parameter_scales']
    assert scales['shelf_2_right/medium'] == 0 and scales['shelf_2_right/small'] == 1
    assert equal(before[0], protected.state_dict()) and not torch.equal(other, small[-1].bias)
    assert all(equal(state, agent.actor_optimizer.state.get(p)) for p, state in moments.items())
    assert agent.success_guard_memory['ceilings'] == limits
    batch, groups = cohort.cohort_batch(agent)
    values, correct = cohort.metrics(agent, batch, groups)
    assert cohort.within_ceilings(values, correct, agent.success_guard_memory, agent.actor_success_guard_config)
    assert report['actor_success_guard_candidate_metric_cache']['final_whole_cohort_check_executed']


def test_final_whole_cohort_can_roll_back_six_heads_and_all_Adam(monkeypatch):
    agent = size_fixture()
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.actor_optimizer.state_dict())
    original = cohort.metrics
    calls = []
    def metrics(*args):
        values, correct = original(*args); calls.append(True)
        if len(calls) == 2:
            correct = correct.clone(); correct[0, 0] = False
        return values, correct
    monkeypatch.setattr(cohort, 'metrics', metrics)
    accepted, report = guard.guarded_actor_step(agent, -agent.actor.network.heads[4][-1].bias[1], None)
    assert len(calls) == 2 and not accepted
    assert report['actor_success_guard_candidate_metric_cache']['final_whole_cohort_check_executed']
    assert equal(before[0], agent.actor.state_dict()) and equal(before[1], agent.actor_optimizer.state_dict())


@pytest.mark.parametrize('direction', ('body', 'jaw', 'zero'))
def test_cached_and_uncached_six_head_search_decisions_and_Adam_are_identical(monkeypatch, direction):
    cached = size_fixture(); full = deepcopy(cached)
    def proposal(agent):
        loss = -agent.actor.network.heads[4][-1].bias[1]
        if direction == 'jaw': loss = loss-agent.actor.network.heads[0][-1].bias[19]
        elif direction == 'zero': loss = loss*0
        return guard.guarded_actor_step(agent, loss, None)
    accepted, report = proposal(cached)
    monkeypatch.setattr(cohort, 'RegionalMetricCache', lambda *args: None)
    full_accepted, full_report = proposal(full)
    assert accepted == full_accepted and equal(cached.actor.state_dict(), full.actor.state_dict())
    assert equal(cached.actor_optimizer.state_dict(), full.actor_optimizer.state_dict())
    assert equal(cached.success_guard_memory, full.success_guard_memory)
    for key in ('actor_success_guard_region_parameter_scales', 'actor_success_guard_candidate_checks',
                'actor_success_guard_region_actual_delta_norms'):
        assert equal(report[key], full_report[key])


def test_size_output_repair_touches_only_one_head_body_mean_and_rejects_incomplete_partition():
    agent = size_fixture(); head = agent.actor.network.heads[0]
    with torch.no_grad(): head[-1].bias[1] += .03
    batch, groups = cohort.cohort_batch(agent)
    params = list(head.parameters())
    correction, kinds = cohort.quantized_body_output_correction(agent, batch, groups, params)
    assert correction is not None and kinds['body'] and torch.isfinite(correction).all()
    offset = 0
    for p in params:
        part = correction[offset:offset+p.numel()].reshape_as(p); offset += p.numel()
        if p is head[-1].weight or p is head[-1].bias: assert not part[19:].any()
        else: assert not part.any()
    assert len(parameter_partitions(agent.actor)) == len(head_labels(agent.actor.network)) == 6
    with pytest.raises(ValueError, match='complete regional'):
        cohort.quantized_body_output_correction(agent, batch, groups, params[-2:])
