"""Rounded jaw repair keeps fixed ceilings, Adam rollback and legacy defaults."""
from copy import deepcopy

import pytest
import torch

from test_rl_region_size_guard import size_fixture
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms import success_update_guard as guard


def test_actual_rounded_jaw_repair_preserves_nonjaw_candidate_rows_and_fixed_limits():
    agent = size_fixture()
    head = agent.actor.network.heads[0]
    parameters = list(head.parameters())
    previous = [p.detach().clone() for p in parameters]
    proposal = [p.detach().clone() for p in parameters]
    proposal[-1][19] += .001  # Wrong protected jaw, including after scale1/64.
    proposal[-1][22] += .005  # A real useful Adam direction must remain.
    norm = torch.cat([(new-old).flatten() for new, old in zip(proposal, previous)]).norm()
    batch, groups = cohort.cohort_batch(agent)
    limits, optimizer, rng = deepcopy(agent.success_guard_memory), deepcopy(agent.actor_optimizer.state_dict()), torch.get_rng_state().clone()
    scale, report = guard.repair_scaled_jaw_candidate(agent, batch, groups, parameters,
        previous, proposal, norm, lambda: cohort.metrics(agent, batch, groups))
    assert scale and report['attempted'] and report['accepted']
    assert report['actual_delta_norm'] <= float(norm)*(1+1e-5)+1e-10
    for p, old, new in zip(parameters, previous, proposal):
        expected = old+scale*(new-old)
        if p is head[-1].weight or p is head[-1].bias:
            assert torch.equal(p[:19], expected[:19]) and torch.equal(p[21:], expected[21:])
        else:
            assert torch.equal(p, expected)
    assert head[-1].bias[22] != previous[-1][22]
    assert all(t['body'] == 0 for t in report['trials'])
    values, jaws = cohort.metrics(agent, batch, groups)
    assert cohort.within_ceilings(values, jaws, limits, agent.actor_success_guard_config)
    assert equal(limits, agent.success_guard_memory) and equal(optimizer, agent.actor_optimizer.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


def test_body_violation_forbids_rounded_jaw_fallback_and_restores_candidate():
    agent = size_fixture(); head = agent.actor.network.heads[0]
    parameters = list(head.parameters())
    previous = [p.detach().clone() for p in parameters]
    proposal = [p.detach().clone() for p in parameters]
    proposal[-1][1] += 100.
    proposal[-1][19] += 1.
    norm = torch.cat([(new-old).flatten() for new, old in zip(proposal, previous)]).norm()
    batch, groups = cohort.cohort_batch(agent)
    scale, report = guard.repair_scaled_jaw_candidate(agent, batch, groups, parameters,
        previous, proposal, norm, lambda: cohort.metrics(agent, batch, groups))
    assert scale == 0 and not report['attempted'] and not report['accepted']
    assert all(torch.equal(p, old) for p, old in zip(parameters, previous))


def test_accepted_ordinary_Adam_candidate_is_exact_with_optin_and_never_calls_jaw_repair(monkeypatch):
    off = size_fixture(); on = deepcopy(off); on.actor_rounded_jaw_repair = True
    def forbidden(*args):
        raise AssertionError('Already accepted ordinary candidate must stay exact')
    monkeypatch.setattr(guard, 'repair_scaled_jaw_candidate', forbidden)
    rng = torch.get_rng_state().clone()
    accepted, report = guard.guarded_actor_step(off, -off.actor.network.heads[4][-1].bias[1], None)
    end_rng = torch.get_rng_state().clone(); torch.set_rng_state(rng)
    other_accepted, other = guard.guarded_actor_step(on, -on.actor.network.heads[4][-1].bias[1], None)
    assert accepted and other_accepted and torch.equal(end_rng, torch.get_rng_state())
    assert equal(off.actor.state_dict(), on.actor.state_dict())
    assert equal(off.actor_optimizer.state_dict(), on.actor_optimizer.state_dict())
    assert equal(off.success_guard_memory, on.success_guard_memory)
    assert other.pop('actor_success_guard_rounded_jaw_fallback_regions') == {}
    assert equal(report, other)


def test_enabled_final_uncached_failure_rolls_back_all_heads_and_Adam(monkeypatch):
    agent = size_fixture(); agent.actor_rounded_jaw_repair = True
    before = deepcopy(agent.actor.state_dict()), deepcopy(agent.actor_optimizer.state_dict())
    original, calls = cohort.metrics, []
    def metrics(*args):
        values, correct = original(*args); calls.append(True)
        if len(calls) == 2:
            correct = correct.clone(); correct[0, 0] = False
        return values, correct
    monkeypatch.setattr(cohort, 'metrics', metrics)
    accepted, report = guard.guarded_actor_step(agent, -agent.actor.network.heads[4][-1].bias[1], None)
    assert not accepted and report['actor_success_guard_candidate_metric_cache']['final_whole_cohort_check_executed']
    assert equal(before[0], agent.actor.state_dict()) and equal(before[1], agent.actor_optimizer.state_dict())


def test_optional_checkpoint_repair_roundtrip_and_legacy_restore_disable_it():
    agent = size_fixture()
    accepted, _ = guard.guarded_actor_step(agent, -agent.actor.network.heads[4][-1].bias[1], None)
    assert accepted
    legacy = deepcopy(agent.checkpoint())
    assert 'actor_rounded_jaw_repair' not in legacy
    agent.actor_rounded_jaw_repair = True
    saved = deepcopy(agent.checkpoint())
    assert saved['actor_rounded_jaw_repair'] == 'rejected-rounded-jaw-output-v1'
    restored = deepcopy(agent); restored.restore(saved)
    assert restored.actor_rounded_jaw_repair
    assert equal(saved, restored.checkpoint())
    restored.restore(legacy)
    assert not restored.actor_rounded_jaw_repair and equal(legacy, restored.checkpoint())


@pytest.mark.parametrize('fault', ['unknown', 'missing_guard'])
def test_invalid_saved_repair_is_rejected_before_mutating_model_or_optimizer(fault):
    agent = size_fixture(); before = deepcopy(agent.checkpoint()); invalid = deepcopy(before)
    invalid['actor_rounded_jaw_repair'] = 'wrong' if fault == 'unknown' else 'rejected-rounded-jaw-output-v1'
    if fault == 'missing_guard': invalid['hybrid_contract'].pop('actor_success_guard')
    with pytest.raises(ValueError, match='rounded jaw repair'):
        agent.restore(invalid)
    assert equal(before, agent.checkpoint())
