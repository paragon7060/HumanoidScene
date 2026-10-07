"""Actual completed returns, zero bootstrap, fair path sampling and resume."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    EPISODE_RETURN_VARIANT, VARIANT, MeasuredTrainCreditBank, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_train_credit import completed_episode_returns
from test_rl_measured_train_credit import path
from test_rl_actual_flap_residual_sac import pilots


def bank():
    return MeasuredTrainCreditBank(518, 577, .999, measured_credit_config(EPISODE_RETURN_VARIANT))


def test_complete_return_uses_real_terminal_and_does_not_relabel_online_rows():
    rows, _ = path(n=4)
    original = deepcopy(rows)
    result = completed_episode_returns(rows, gamma=.5)
    torch.testing.assert_close(result['reward'], torch.tensor([2., 2., 0., -6.]))
    assert result['terminated'].all() and result['bootstrap_discount'].eq(0).all()
    assert result['n_steps'].tolist() == [4, 3, 2, 1]
    assert result['next_actor_obs'][:, 0].tolist() == [4.] * 4
    assert result['next_critic_obs'][:, 0].tolist() == [4.] * 4
    assert all(torch.equal(value, original[key]) for key, value in rows.items())


def test_full900_step_return_matches_recorded_episode_and_no_16_step_cap():
    rows, _ = path(n=900)
    result = completed_episode_returns(rows, gamma=1.)
    expected = rows['reward'].flip(0).cumsum(0).flip(0)
    torch.testing.assert_close(result['reward'], expected, rtol=0, atol=0)
    assert result['n_steps'][0] == 900 and result['n_steps'][-1] == 1
    assert result['terminated'].all() and result['bootstrap_discount'].eq(0).all()


@pytest.mark.parametrize('mutation', [
    lambda r: r['terminated'].__setitem__(-1, False),
    lambda r: r['terminated'].__setitem__(1, True),
    lambda r: r['next_actor_obs'].__setitem__((1, 0), 123.),
])
def test_incomplete_reset_or_missing_row_cannot_receive_completed_return(mutation):
    rows, _ = path(n=5)
    mutation(rows)
    with pytest.raises(ValueError):
        completed_episode_returns(rows, gamma=.999)


def test_path_uniform_sampling_does_not_let_long_failure_dominate_success():
    b = bank()
    for env, success, n in ((0, True, 3), (1, False, 90)):
        rows, outcome = path(success, env=env, n=n)
        for k in ('actor_obs', 'next_actor_obs', 'critic_obs', 'next_critic_obs'):
            rows[k][:, 1] = env
        b.add_episode(rows, outcome, source_run='same_real_TRAIN_fixture')
    torch.manual_seed(418)
    sampled = b.sample(2048, 'cpu')
    short_fraction = sampled['actor_obs'][:, 1].eq(0).float().mean()
    assert .45 < short_fraction < .55
    assert sampled['terminated'].all() and sampled['bootstrap_discount'].eq(0).all()
    assert b.report()['bootstrap_allowed'] is False
    restored = bank()
    restored.restore(b.state())
    assert restored.report() == b.report()


def test_full_return_target_never_evaluates_terminal_next_state_and_keeps_actor(tmp_path):
    _, pilot, *_ = pilots(tmp_path)
    pilot.agent.measured_train_credit_enabled = True
    pilot.agent.measured_train_credit_config = measured_credit_config(EPISODE_RETURN_VARIANT)
    b = bank()
    rows, outcome = path(n=40)
    b.add_episode(rows, outcome, source_run='fixture')
    aux = b.sample(64, 'cpu')
    aux['next_actor_obs'].zero_()
    aux['next_critic_obs'].fill_(1e20)
    pilot.agent.validate_critic_auxiliary(aux, .1)
    target, stats = pilot.agent.critic_target(aux, torch.tensor(0.), torch.tensor(0.),
        bootstrap_discount=aux['bootstrap_discount'])
    torch.testing.assert_close(target, aux['reward'])
    assert stats['bootstrapped_rows'] == 0
    before = deepcopy(pilot.agent.actor.state_dict())
    report = pilot.agent.update(rows, critic_auxiliary=aux, critic_auxiliary_weight=.1, update_actor=False)
    assert report['measured_episode_return_rows'] == 64
    assert report['measured_episode_return_bootstrapped_rows'] == 0
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
    assert all(torch.equal(v, before[k]) for k, v in pilot.agent.actor.state_dict().items())
    invalid = deepcopy(aux)
    invalid['terminated'][0] = False
    with pytest.raises(ValueError, match='without bootstrap'):
        pilot.agent.validate_critic_auxiliary(invalid, .1)
    pilot.agent.measured_train_credit_config = measured_credit_config(VARIANT)
    aux['n_steps'][0] = 40
    with pytest.raises(ValueError, match='Malformed'):
        pilot.agent.validate_critic_auxiliary(aux, .1)


def test_episode_return_objective_and_bank_resume_without_mixing_nstep_contract(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    _, old, warm, physical, stage, _ = pilots(tmp_path)
    old.directory.mkdir()
    old.save(final=True)
    cp = next(old.directory.glob('checkpoint_*.pt'))
    enabled = ActualFlapResidualSACPilot(warm, physical, tmp_path/'enabled', stage,
        checkpoint=cp, measured_train_credit=EPISODE_RETURN_VARIANT)
    assert enabled.contract == old.contract
    assert all(torch.equal(v, enabled.agent.state_dict()[k]) for k, v in old.agent.state_dict().items())
    rows, outcome = path(n=40)
    enabled.measured_credit_bank.add_episode(rows, outcome, source_run='fixture')
    enabled.directory.mkdir()
    enabled.save(final=True)
    saved = next(enabled.directory.glob('checkpoint_*.pt'))
    restored = ActualFlapResidualSACPilot(warm, physical, tmp_path/'restored', stage, checkpoint=saved)
    assert restored.measured_train_credit == measured_credit_config(EPISODE_RETURN_VARIANT)
    assert restored.agent.measured_train_credit_config == restored.measured_train_credit
    assert restored.measured_credit_bank.report() == enabled.measured_credit_bank.report()
    with pytest.raises(ValueError, match='differs from checkpoint'):
        ActualFlapResidualSACPilot(warm, physical, tmp_path/'invalid', stage,
            checkpoint=saved, measured_train_credit=VARIANT)


def actor_inputs():
    from test_rl_body_policy_spread import initializer_inputs
    from prepare_gentle_servo_actor import prepare as quarter_prepare
    from prepare_servo_retention_actor import prepare as retention_prepare
    state, replay = initializer_inputs()
    state, replay, _ = quarter_prepare(state, replay)
    state['hybrid_contract'] = {'fixture_only': True}
    state, replay = retention_prepare(state, replay)
    state['body_anchor_state'] = {'model': {'weight': torch.zeros(2)}}
    state['frozen_actor_prior'] = {'model': {'weight': torch.ones(2)}}
    source = deepcopy(state)
    source['actor_updates'], source['critic_updates'] = 31, 201
    source['model']['actor.network.4.bias'][0] = .13
    source['model']['q1.weight'].fill_(3.)
    source['successful_train_transitions']['episodes']['middle_left'] = [{'old_rows': 'not_imported'}]
    return state, replay, source


def test_actor_only_return_initializer_preserves_fresh_Q_and_excludes_old_reward_banks():
    from prepare_episode_return_actor import prepare
    from prepare_actual_success_actor_tail import identical
    initial, replay, source = actor_inputs()
    before, replay_before = deepcopy(initial), deepcopy(replay)
    state, experience = prepare(initial, replay, source)
    assert identical(initial, before) and identical(replay, replay_before)
    assert state['model']['actor.network.4.bias'][0] == .13
    assert torch.equal(state['model']['q1.weight'], initial['model']['q1.weight'])
    assert state['actor_updates'] == state['critic_updates'] == 0
    assert not any(state['successful_train_transitions']['episodes'].values())
    assert not any(experience['measured_train_credit_bank']['episodes'].values())
    assert state['measured_train_credit'] == experience['measured_train_credit_bank']['config'] == measured_credit_config(EPISODE_RETURN_VARIANT)


@pytest.mark.parametrize('bad', ['trained_Q', 'reward_bank', 'different_credit', 'different_jaw_prior', 'nonfinite_Q'])
def test_actor_only_return_initializer_rejects_contamination_or_incompatible_behavior(bad):
    from prepare_episode_return_actor import prepare
    initial, replay, source = actor_inputs()
    if bad == 'trained_Q': initial['critic_updates'] = 1
    if bad == 'reward_bank': replay['measured_train_credit_bank']['episodes']['middle_left'] = [{}]
    if bad == 'different_credit': replay['measured_train_credit'] = {}
    if bad == 'different_jaw_prior': source['frozen_actor_prior']['model']['weight'][0] = 4.
    if bad == 'nonfinite_Q': source['model']['q1.weight'][0] = float('nan')
    with pytest.raises(ValueError): prepare(initial, replay, source)
