"""Candidate caching must preserve actual hard-guard and Adam decisions."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from test_rl_success_cohort_guard import sample
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms.success_update_guard import guarded_actor_step
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import install_regional_actor
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS


def regional_fixture():
    torch.manual_seed(7143)
    agent, bank = sample(cohort.TUBE_VARIANT)
    agent.config = replace(agent.config, actor_lr=agent.actor_optimizer.param_groups[0]['lr'])
    install_regional_actor(agent)
    seed = bank.episodes[REGIONS[0]][0]
    entries = {}
    for index, region in enumerate(REGIONS[:2]):
        item = deepcopy(seed)
        item['identity'] = f'cache/wave1/env{index}/seed{60000000+index}'
        item['outcome'].update(environment=index)
        item['outcome']['layout'].update(target_region=region, seed=60000000+index)
        raw = seed['rows']['actor_obs'].repeat(18, 1)
        raw[:, 94:98] = 0.; raw[:, 94+index] = 1.
        # Store each fixed approach/tail shape exactly as the production guard
        # will decode it, rather than introducing a mixed-batch rounding change.
        parts = [agent.act(raw[:8], True), agent.act(raw[8:], True)]
        item['rows'] = dict(actor_obs=raw, action=torch.cat(parts).detach())
        entries[region] = [item]
    agent.success_guard_memory = cohort.empty_memory()
    cohort.synchronize(agent, SimpleNamespace(episodes=entries))
    return agent


def test_cached_candidates_match_full_cohort_after_each_committed_region():
    agent = regional_fixture()
    batch, groups = cohort.cohort_batch(agent)
    values, correct = cohort.metrics(agent, batch, groups)
    cache = cohort.RegionalMetricCache(agent, batch, groups, values, correct)
    rng = torch.get_rng_state().clone()
    for index, region in enumerate(REGIONS[:2]):
        with torch.no_grad():
            agent.actor.network.heads[index][-1].bias[1] += .00001
        candidate, jaws = cache.evaluate(region)
        expected, expected_jaws = cohort.metrics(agent, batch, groups)
        assert equal(candidate, expected) and torch.equal(jaws, expected_jaws)
        cache.commit(region)
    assert torch.equal(torch.get_rng_state(), rng)
    # Only two of four fixed phases need a forward for each regional candidate.
    assert len(groups) == 4 and cache.phase_forwards == 2*cache.evaluations


@pytest.mark.parametrize('direction', ('body_and_wrong_jaw', 'large_body', 'zero'))
def test_cached_and_uncached_search_have_identical_parameters_moments_and_checks(monkeypatch, direction):
    cached = regional_fixture(); original = deepcopy(cached)
    old_memory = deepcopy(cached.success_guard_memory)

    def proposal(agent):
        heads = agent.actor.network.heads
        loss = -heads[0][-1].bias[1] + heads[1][-1].bias[19]
        if direction == 'body_and_wrong_jaw':
            loss = loss-heads[0][-1].bias[19]
        elif direction == 'large_body':
            agent.actor_optimizer.param_groups[0]['lr'] = .03
        else:
            loss = loss*0.
        return guarded_actor_step(agent, loss, None)

    accepted, report = proposal(cached)
    monkeypatch.setattr(cohort, 'RegionalMetricCache', lambda *args: None)
    old_accepted, old_report = proposal(original)
    assert accepted == old_accepted
    assert equal(cached.actor.state_dict(), original.actor.state_dict())
    assert equal(cached.actor_optimizer.state_dict(), original.actor_optimizer.state_dict())
    assert equal(cached.success_guard_memory, original.success_guard_memory)
    assert cached.success_guard_memory['ceilings'] == old_memory['ceilings']
    for key in ('actor_success_guard_candidate_checks', 'actor_success_guard_region_parameter_scales',
                'actor_success_guard_region_actual_delta_norms', 'actor_success_guard_candidate_repairs',
                'actor_success_guard_quantized_body_fallback_regions'):
        assert equal(report[key], old_report[key]), key
    if accepted:
        batch, groups = cohort.cohort_batch(cached)
        values, jaws = cohort.metrics(cached, batch, groups)
        assert cohort.within_ceilings(values, jaws, cached.success_guard_memory, cached.actor_success_guard_config)
    cache_report = report['actor_success_guard_candidate_metric_cache']
    assert cache_report['final_whole_cohort_check_required_for_acceptance']
    assert cache_report['final_whole_cohort_check_executed'] == accepted


def test_final_uncached_check_can_reject_even_when_cached_candidate_passed(monkeypatch):
    agent = regional_fixture()
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.actor_optimizer.state_dict())
    original = cohort.metrics
    calls = []

    def metrics(*args):
        values, correct = original(*args)
        calls.append(True)
        if len(calls) == 2:
            # Simulate a final independent check finding a protected jaw failure.
            correct = correct.clone(); correct[0, 0] = False
        return values, correct

    monkeypatch.setattr(cohort, 'metrics', metrics)
    accepted, report = guarded_actor_step(agent, agent.actor.network.heads[0][-1].bias[19], None)
    assert len(calls) == 2 and not accepted
    assert report['actor_success_guard_candidate_metric_cache']['final_whole_cohort_check_executed']
    assert not report['actor_success_guard_accepted']
    assert equal(before[0], agent.actor.state_dict())
    assert equal(before[1], agent.actor_optimizer.state_dict())
