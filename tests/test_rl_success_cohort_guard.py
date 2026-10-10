"""Persistent provenance, per-hand correctness, and cumulative update limits."""
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch

from test_rl_success_update_guard import fixture, equal
from kuavo_isaaclab_scene.rl.algorithms.success_update_guard import (
    guarded_actor_step, success_update_guard_config)
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort


def sample(variant=cohort.VARIANT):
    agent, success = fixture()
    agent.actor_success_guard_config = success_update_guard_config(variant)
    raw = success['actor_obs'][:4].clone(); raw[:, -6] = 1.
    labels = agent.act(raw, True).detach()
    outcome = dict(wave=1, split='train', environment=0, initial_layout_valid=True, complete=True,
        layout=dict(target_region='shelf_2_right', target_box_type='small', seed=60000000),
        result=dict(success=True, unsafe=False, pinching=[True, True], stable_hands=[True, True],
            opposing_flaps=True, proof_lift=True, hold_time_s=.25, rack_clearance_m=.01,
            staged_base=dict(phase='held_grasp', manipulation_start=1)))
    episode = dict(identity='test/wave1/env0/seed60000000', outcome=outcome,
                   rows=dict(actor_obs=raw, action=labels))
    bank = SimpleNamespace(episodes={'shelf_2_right': [episode]})
    cohort.synchronize(agent, bank)
    return agent, bank


def test_rejected_region_restores_mature_Adam_while_unprotected_region_can_update():
    from dataclasses import replace
    from kuavo_isaaclab_scene.rl.algorithms.common import optimize
    from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import install_regional_actor
    a,bank=sample(cohort.STABLE_VARIANT)
    a.config=replace(a.config,actor_lr=a.actor_optimizer.param_groups[0]['lr'])
    install_regional_actor(a)
    a.action_projector.entropy_mask=lambda raw:raw.new_zeros(len(raw),21)
    heads=a.actor.network.heads
    for _ in range(5):
        optimize(a.actor_optimizer,-heads[0][-1].bias[19]-heads[1][-1].bias[19],a.actor.parameters())
    with torch.no_grad():heads[0][-1].bias[19]=0.
    a.success_guard_memory=cohort.empty_memory()
    cohort.synchronize(a,bank)
    original=deepcopy(heads[0].state_dict())
    moments={p:deepcopy(a.actor_optimizer.state.get(p)) for p in heads[0].parameters()}
    other=heads[1][-1].bias.detach().clone()
    accepted,report=guarded_actor_step(a,-heads[0][-1].bias[19]-heads[1][-1].bias[19],None)
    assert accepted
    assert report['actor_success_guard_region_parameter_scales']['shelf_2_right']==0.
    assert report['actor_success_guard_region_parameter_scales']['shelf_2_left']==1.
    assert equal(original,heads[0].state_dict())
    assert all(equal(before,a.actor_optimizer.state.get(p)) for p,before in moments.items())
    assert heads[1][-1].bias[19]>other[19]
    batch,groups=cohort.cohort_batch(a);values,correct=cohort.metrics(a,batch,groups)
    assert cohort.within_ceilings(values,correct,a.success_guard_memory,a.actor_success_guard_config)


def test_stable_guard_does_not_count_zero_parameter_motion_as_an_actor_update():
    a,_=sample(cohort.STABLE_VARIANT)
    before=deepcopy(a.actor_optimizer.state_dict())
    accepted,report=guarded_actor_step(a,a.actor.network[-1].bias.sum()*0.,None)
    assert not accepted and report['actor_success_guard_parameter_delta_norm']==0.
    assert equal(before,a.actor_optimizer.state_dict())


def test_first_success_cohort_survives_eviction_and_repeated_synchronization():
    a, bank = sample(); before = deepcopy(a.success_guard_memory)
    bank.episodes['shelf_2_right'] = []
    cohort.synchronize(a, bank)
    assert equal(before, a.success_guard_memory)
    batch, groups = cohort.cohort_batch(a)
    assert len(batch['action']) == 4 and len(groups) == 1
    assert cohort.report(a)['episodes'] == 1


def test_correct_jaws_cannot_be_swapped_even_if_total_error_count_is_unchanged():
    a, _ = sample(); batch, groups = cohort.cohort_batch(a)
    values, correct = cohort.metrics(a, batch, groups)
    memory = deepcopy(a.success_guard_memory)
    memory['entries'][0]['protected_correct_jaws'][0, 0] = False
    before = correct.clone(); before[0, 0] = False
    after = correct.clone(); after[1, 0] = False
    assert int((~before).sum()) == int((~after).sum())
    assert not cohort.within_ceilings(values, after, memory, a.actor_success_guard_config)


def test_guard_jaw_checks_use_the_same_zero_logit_boundary_as_production():
    a, _ = sample(); batch, groups = cohort.cohort_batch(a)
    _, correct = cohort.metrics(a, batch, groups)
    assert torch.equal(correct, a.act(batch['actor_obs'], True)[:, 19:] == batch['action'][:, 19:])
    assert correct.all()


def test_absolute_slack_cannot_accumulate_across_many_accepted_steps():
    a, _ = sample(); batch, groups = cohort.cohort_batch(a)
    values, correct = cohort.metrics(a, batch, groups); key = next(iter(values))
    memory = deepcopy(a.success_guard_memory)
    memory['ceilings'][key]['left_arm'] = 0.
    candidate = deepcopy(values); candidate[key]['left_arm'] = .9e-9
    assert cohort.within_ceilings(candidate, correct, memory, a.actor_success_guard_config)
    cohort.tighten(memory, candidate, correct)
    candidate[key]['left_arm'] = 1.8e-9
    assert not cohort.within_ceilings(candidate, correct, memory, a.actor_success_guard_config)
    assert memory['ceilings'][key]['left_arm'] == 0.


def test_one_arm_improvement_cannot_pay_for_the_other_arms_regression():
    a, _ = sample(); batch, groups = cohort.cohort_batch(a)
    values, correct = cohort.metrics(a, batch, groups); key = next(iter(values))
    memory = deepcopy(a.success_guard_memory)
    memory['ceilings'][key].update(left_arm=1., right_arm=1.)
    candidate = deepcopy(values); candidate[key].update(left_arm=.1, right_arm=1.01)
    assert not cohort.within_ceilings(candidate, correct, memory, a.actor_success_guard_config)


@pytest.mark.parametrize('variant',[cohort.VARIANT,cohort.STABLE_VARIANT])
def test_persistent_paths_and_counters_round_trip_with_the_actual_optimizer(variant):
    a, _ = sample(variant)
    accepted, report = guarded_actor_step(a, a.actor.network[-1].bias[19], None)
    assert accepted and report['actor_success_guard_cohort']['episodes'] == 1
    b, _ = sample(variant); b.restore(a.checkpoint())
    assert equal(a.success_guard_memory, b.success_guard_memory)
    assert equal(a.actor.state_dict(), b.actor.state_dict())
    assert equal(a.actor_optimizer.state_dict(), b.actor_optimizer.state_dict())
    assert a.success_guard_statistics == b.success_guard_statistics
    assert a.success_guard_statistics['accepted'] == 1


def test_DEV_or_unsafe_success_cannot_enter_the_persistent_cohort():
    a, bank = sample()
    a.success_guard_memory = cohort.empty_memory()
    bank.episodes['shelf_2_right'][0]['outcome']['split'] = 'validation'
    with pytest.raises(ValueError, match='TRAIN only'):
        cohort.synchronize(a, bank)
    assert not a.success_guard_memory['entries']


def test_corrupt_checkpoint_path_or_missing_persistent_memory_is_rejected():
    a, _ = sample(); saved = a.checkpoint()
    b, _ = sample(); bad = deepcopy(saved)
    bad['success_guard_memory']['entries'][0]['outcome']['result']['unsafe'] = True
    with pytest.raises(ValueError, match='Unsafe'):
        b.restore(bad)
    bad = deepcopy(saved); bad.pop('success_guard_memory')
    with pytest.raises(ValueError, match='missing'):
        b.restore(bad)


def test_fresh_restored_checkpoint_starts_aggregate_counters_on_first_proposal():
    a, _ = sample(); b, _ = sample()
    saved = a.checkpoint(); saved['success_guard_statistics'] = {}
    b.restore(saved)
    accepted, _ = guarded_actor_step(b, b.actor.network[-1].bias[19], None)
    assert accepted and b.success_guard_statistics['attempted'] == 1


def test_rejected_cohort_update_restores_actor_and_Adam_but_keeps_critic_learning():
    from kuavo_isaaclab_scene.rl.algorithms.common import optimize
    a, _ = sample(); success, _ = cohort.cohort_batch(a); raw = success['actor_obs']
    a.action_projector.entropy_mask = lambda obs: obs.new_zeros(len(obs), 21)
    for _ in range(5):
        optimize(a.actor_optimizer, -a.actor.network[-1].bias[19], a.actor.parameters())
    with torch.no_grad(): a.actor.network[-1].bias[19] = 0.
    a.actor_jaw_regularization = lambda logits, near: (-100. * logits[:, 0].mean(), {})
    before = deepcopy(a.actor.state_dict()), deepcopy(a.actor_optimizer.state_dict())
    q = deepcopy(a.q1.state_dict())
    temperatures = [p.detach().clone() for p in (a.log_alpha, a.log_alpha_discrete)]
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(len(raw), 4), action=success['action'],
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(len(raw), 4), reward=torch.ones(len(raw)),
        terminated=torch.ones(len(raw), dtype=torch.bool))
    report = a.update(batch, successful_train=success, success_goal_weight=2., success_jaw_weight=.05)
    assert not report['actor_updated'] and a.success_guard_statistics['rejected'] == 1
    assert equal(before[0], a.actor.state_dict()) and equal(before[1], a.actor_optimizer.state_dict())
    assert not equal(q, a.q1.state_dict())
    assert all(torch.equal(x, y) for x, y in zip(temperatures, (a.log_alpha, a.log_alpha_discrete)))


def test_stable_guard_fixed_path_forward_does_not_change_when_cohort_grows():
    a, bank = sample(cohort.STABLE_VARIANT)
    original=a.continuous_parameters
    # Model GEMM kernels can vary at different batch sizes. Deliberately
    # amplify that dependence so the regression is deterministic on CPU.
    def batch_sensitive(normalized, raw):
        mean,std,logits=original(normalized,raw)
        return mean,std,logits+len(raw)*1e-4
    a.continuous_parameters=batch_sensitive
    a.success_guard_memory=cohort.empty_memory();cohort.synchronize(a,bank)
    old=deepcopy(a.success_guard_memory['ceilings'])
    other=deepcopy(bank.episodes['shelf_2_right'][0]);other['identity']='test/wave2/env0/seed60000000'
    other['outcome']['wave']=2
    other['rows']={k:v[:3].clone() for k,v in other['rows'].items()}
    bank.episodes['shelf_2_right'].append(other);cohort.synchronize(a,bank)
    batch,groups=cohort.cohort_batch(a);values,correct=cohort.metrics(a,batch,groups)
    assert all(values[k]==v for k,v in old.items())
    assert all(a.success_guard_memory['ceilings'][k]==v for k,v in old.items())
    assert cohort.within_ceilings(values,correct,a.success_guard_memory,a.actor_success_guard_config)
    losses,_=cohort.losses_and_jaws(a,batch,groups)
    assert all(v.dtype==torch.float64 for terms in losses.values() for v in terms.values())
