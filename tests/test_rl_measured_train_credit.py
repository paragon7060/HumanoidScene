"""Actual path separation, finite failures, n-step endpoints and resume semantics."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    MeasuredTrainCreditBank, VARIANT, add_measured_training_wave, measured_credit_config,
)
from test_rl_actual_flap_residual_sac import pilots


def path(success=False, env=0, n=5):
    states = torch.arange(n + 1).float()
    rows = {k: torch.zeros(n, d) for k, d in (
        ('actor_obs', 518), ('next_actor_obs', 518),
        ('critic_obs', 577), ('next_critic_obs', 577), ('action', 21))}
    for current, following in (('actor_obs', 'next_actor_obs'), ('critic_obs', 'next_critic_obs')):
        rows[current][:, 0] = states[:-1]
        rows[following][:, 0] = states[1:]
        rows[current][:, -6] = rows[following][:, -6] = 1
    rows['actor_obs'][:, 94] = rows['next_actor_obs'][:, 94] = 1
    rows['action'][:, 19:] = -1
    rows['reward'] = torch.tensor([*range(1, n), 8. if success else -6.])
    rows['terminated'] = torch.tensor([False] * (n - 1) + [True])
    outcome = dict(wave=1, split='train', environment=env, initial_layout_valid=True, complete=True,
        layout=dict(seed=120000 + env, target_region='shelf_2_right'),
        result=dict(success=success, unsafe=not success, numerical_failure=False, invalid_reset=False,
            staged_base=dict(phase='held_grasp', manipulation_start=70)))
    return rows, outcome


def bank():
    return MeasuredTrainCreditBank(518, 577, .999, measured_credit_config(VARIANT))


def test_actual_success_and_finite_unsafe_failure_keep_rewards_and_terminal():
    b = bank()
    for env, success in enumerate((False, True)):
        rows, outcome = path(success, env)
        b.add_episode(rows, outcome, source_run='unit_fixture')
        stored = b.episodes['shelf_2_right'][-1]['rows']
        for k in rows:
            assert torch.equal(rows[k], stored[k])
    assert b.report()['by_region']['shelf_2_right']['failures'] == 1
    assert b.report()['by_region']['shelf_2_right']['successes'] == 1
    sampled = b.sample(128, 'cpu')
    assert sampled['terminated'].all() and sampled['bootstrap_discount'].eq(0).all()
    assert (sampled['reward'] < 0).any() and (sampled['reward'] > 0).any()
    restored = bank()
    restored.restore(b.state())
    assert restored.report() == b.report()


@pytest.mark.parametrize('mutation', [
    lambda o: o.update(split='validation'), lambda o: o.update(split='holdout'),
    lambda o: o.update(initial_layout_valid=False), lambda o: o.update(complete=False),
    lambda o: o['result'].update(numerical_failure=True),
    lambda o: o['result'].update(invalid_reset=True),
    lambda o: o['result']['staged_base'].update(waypoint_probe={}),
])
def test_reject_evaluation_replaced_broken_or_changed_controller_path(mutation):
    b = bank()
    rows, outcome = path()
    mutation(outcome)
    with pytest.raises(ValueError, match='matching TRAIN'):
        b.add_episode(rows, outcome, source_run='fixture')
    assert b.size == 0


def test_cannot_jump_across_quarantined_row_or_partial_reset():
    rows, outcome = path()
    rows['next_actor_obs'][1, 0] = 100
    with pytest.raises(ValueError, match='discontinuous'):
        bank().add_episode(rows, outcome, source_run='fixture')
    rows, outcome = path()
    rows['terminated'][1] = True
    with pytest.raises(ValueError, match='one complete actual terminal'):
        bank().add_episode(rows, outcome, source_run='fixture')


def test_nonterminal_windows_bootstrap_from_measured_endpoint_at_gamma_to_16(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_train_credit import discounted_episode_windows
    _, pilot, *_ = pilots(tmp_path)
    pilot.agent.measured_train_credit_enabled = True
    class ConstantQ(torch.nn.Module):
        def forward(self, x): return x.new_full((len(x), 1), 2.)
    pilot.agent.target1 = pilot.agent.target2 = ConstantQ()
    rows, _ = path(n=20)
    windows = discounted_episode_windows(rows, horizon=16, gamma=.999)
    pilot.agent.validate_critic_auxiliary(windows, .1)
    target, stats = pilot.agent.critic_target(windows, torch.tensor(0.), torch.tensor(0.),
        bootstrap_discount=windows['bootstrap_discount'])
    torch.testing.assert_close(target[0], windows['reward'][0] + 2 * .999**16)
    assert stats['bootstrapped_rows'] == 4 and windows['next_actor_obs'][0, 0] == 16
    assert target[-1] == -6 and windows['bootstrap_discount'][-1] == 0


def test_capacity_preserves_rare_success_with_real_failed_paths():
    b = bank()
    rows, outcome = path(True, n=900)
    b.add_episode(rows, outcome, source_run='fixture')
    for env in range(1, 12):
        rows, outcome = path(False, env, n=900)
        b.add_episode(rows, outcome, source_run='fixture')
    report = b.report()['by_region']['shelf_2_right']
    assert report['successes'] == 1 and report['failures'] == 8 and b.size == 8100


def test_wave_identifies_individual_environments_and_reports_skipped_paths():
    failed, failure = path(env=7)
    successful, success = path(True, env=4)
    batches = [(torch.tensor([7, 4]), {k: torch.stack((failed[k][j], successful[k][j])) for k in failed})
        for j in range(5)]
    b = bank()
    report = add_measured_training_wave(b, {'split': 'train'}, [failure, success], batches,
        source_run='fixture')
    assert report == dict(added=2, skipped={})
    assert torch.equal(b.episodes['shelf_2_right'][0]['rows']['reward'], failed['reward'])
    with pytest.raises(ValueError, match='Evaluation'):
        add_measured_training_wave(b, {'split': 'validation'}, [failure], batches, source_run='fixture')


def test_bounded_auxiliary_defaults_off_and_masks_unusable_terminal_next(tmp_path):
    _, pilot, *_ = pilots(tmp_path)
    rows, outcome = path()
    b = bank()
    b.add_episode(rows, outcome, source_run='fixture')
    aux = b.sample(64, 'cpu')
    # These placeholders must never invoke the constrained body decoder.
    aux['next_actor_obs'].zero_()
    aux['next_critic_obs'].fill_(1e20)
    with pytest.raises(ValueError, match='does not support'):
        pilot.agent.update(rows, critic_auxiliary=aux, critic_auxiliary_weight=.1, update_actor=False)
    pilot.agent.measured_train_credit_enabled = True
    actor_before = deepcopy(pilot.agent.actor.state_dict())
    rewards_before = rows['reward'].clone()
    report = pilot.agent.update(rows, critic_auxiliary=aux, critic_auxiliary_weight=.1, update_actor=False)
    assert report['measured_nstep_rows'] == 64 and report['measured_nstep_terminal_rows'] == 64
    assert report['measured_nstep_bootstrapped_rows'] == 0
    assert report['one_step_q_loss'] > 0 and report['measured_nstep_weight'] == .1
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
    assert all(torch.equal(v, actor_before[k]) for k, v in pilot.agent.actor.state_dict().items())
    assert torch.equal(rows['reward'], rewards_before)
    aux['bootstrap_discount'][0] = .999
    with pytest.raises(ValueError, match='actual horizon'):
        pilot.agent.validate_critic_auxiliary(aux, .1)


def test_opt_in_preserves_model_optimizer_contract_and_resume_bank(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    _, old, warm, physical, stage, _ = pilots(tmp_path)
    old.directory.mkdir()
    raw = torch.zeros(64, 464); raw[:, 144] = 1
    extra = torch.zeros(64, 38); critic = torch.zeros(64, 530)
    previous = old.act(raw, critic, 0, supplemental=extra)[1]
    old.observe(previous, raw, critic, torch.ones(64), torch.ones(64, dtype=torch.bool), 0, supplemental=extra)
    old.save(final=True)
    checkpoint = next(old.directory.glob('checkpoint_*.pt'))
    enabled = ActualFlapResidualSACPilot(warm, physical, tmp_path/'enabled', stage,
        checkpoint=checkpoint, measured_train_credit=VARIANT)
    assert enabled.contract == old.contract
    assert (enabled.actor_updates, enabled.critic_updates, enabled.replay.size) == (
        old.actor_updates, old.critic_updates, old.replay.size)
    for key, value in old.agent.state_dict().items():
        assert torch.equal(value, enabled.agent.state_dict()[key])
    for before, after in zip(old.agent.optimizers, enabled.agent.optimizers):
        a, b = before.state_dict(), after.state_dict()
        assert a['param_groups'] == b['param_groups'] and a['state'].keys() == b['state'].keys()
        for parameter, values in a['state'].items():
            for key, value in values.items():
                assert torch.equal(value, b['state'][parameter][key])
    rows, outcome = path()
    enabled.measured_credit_bank.add_episode(rows, outcome, source_run='fixture')
    enabled.directory.mkdir()
    enabled.save(final=True)
    cp = next(enabled.directory.glob('checkpoint_*.pt'))
    restored = ActualFlapResidualSACPilot(warm, physical, tmp_path/'restored', stage, checkpoint=cp)
    assert restored.measured_credit_bank.report() == enabled.measured_credit_bank.report()
    assert restored.measured_train_credit == measured_credit_config(VARIANT)
    assert restored.contract == old.contract
    with pytest.raises(ValueError, match='differs from checkpoint'):
        ActualFlapResidualSACPilot(warm, physical, tmp_path/'invalid', stage,
            checkpoint=cp, measured_train_credit='one-step')
