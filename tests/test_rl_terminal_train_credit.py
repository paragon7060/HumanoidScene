"""Actual terminal labels, balanced regions and strict sampling resume provenance."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    MeasuredTrainCreditBank, VARIANT, TERMINAL_VARIANT, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS
from test_rl_measured_train_credit import path


def bank(variant=TERMINAL_VARIANT):
    return MeasuredTrainCreditBank(518, 577, .999, measured_credit_config(variant))


def add(b, region, env, n=40, success=False):
    rows, outcome = path(success=success, env=env, n=n)
    outcome['layout']['target_region'] = region
    for key in ('actor_obs', 'next_actor_obs'):
        rows[key][:, 94:98] = 0
        rows[key][:, 94 + REGIONS.index(region)] = 1
    b.add_episode(rows, outcome, source_run='terminal_fixture')
    return rows


def test_long_paths_guarantee_real_terminal_one_step_labels_and_balanced_regions():
    torch.manual_seed(7)
    b = bank()
    for env, region in enumerate(REGIONS):
        add(b, region, env)
    batch = b.sample(64, 'cpu')
    final = batch['n_steps'].eq(1)
    assert final.sum() >= 16
    assert batch['reward'][final].eq(-6).all()
    assert batch['terminated'][final].all()
    assert batch['bootstrap_discount'][final].eq(0).all()
    assert batch['actor_obs'][final, 0].eq(39).all()
    assert batch['next_actor_obs'][final, 0].eq(40).all()
    assert batch['action'][final, 19:].eq(-1).all()
    regions = batch['actor_obs'][:, 94:98].argmax(-1)
    assert all(regions.eq(i).sum() == 16 for i in range(4))
    assert all((regions.eq(i) & final).sum() >= 4 for i in range(4))
    assert (batch['n_steps'].eq(16) & ~batch['terminated']).any()


def test_terminal_pool_balances_episodes_instead_of_length_and_keeps_success_rewards():
    torch.manual_seed(19)
    b = bank()
    original = add(b, REGIONS[0], 0, n=2, success=True)
    add(b, REGIONS[0], 1, n=800)
    batch = b.sample(256, 'cpu')
    final = batch['n_steps'].eq(1)
    assert final.sum() >= 64
    # A uniform-row-only sampler almost never sees the two-row success.
    assert (final & batch['reward'].eq(8)).sum() >= 8
    assert (final & batch['reward'].eq(-6)).sum() >= 8
    assert torch.equal(b.episodes[REGIONS[0]][0]['rows']['reward'], original['reward'])


def test_sampling_contract_is_saved_restored_and_cannot_be_silently_downgraded():
    b = bank()
    add(b, REGIONS[0], 0)
    saved = deepcopy(b.state())
    restored = bank()
    restored.restore(saved)
    assert restored.report() == b.report()
    assert restored.config['terminal_batch_fraction'] == .25
    with pytest.raises(ValueError, match='differs'):
        bank(VARIANT).restore(saved)
    with pytest.raises(ValueError, match='differs'):
        bank().restore(bank(VARIANT).state())


def test_legacy_sampler_and_report_are_unchanged():
    b = bank(VARIANT)
    add(b, REGIONS[0], 0, n=800)
    assert 'terminal_batch_fraction' not in b.config
    assert 'terminal_batch_fraction' not in b.report()
    torch.manual_seed(41)
    ordinary = b.sample(64, 'cpu')
    torch.manual_seed(41)
    legacy = b._sample_uniform_rows(64, 'cpu')
    assert all(torch.equal(ordinary[k], legacy[k]) for k in ordinary)


def test_actual_pilot_restores_terminal_sampler_and_rejects_legacy_override(tmp_path):
    from test_rl_actual_flap_residual_sac import pilots
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    _, old, warm, physical, stage, _ = pilots(tmp_path)
    old.directory.mkdir()
    old.save(final=True)
    source = next(old.directory.glob('checkpoint_*.pt'))
    enabled = ActualFlapResidualSACPilot(warm, physical, tmp_path/'terminal', stage,
        checkpoint=source, measured_train_credit=TERMINAL_VARIANT)
    assert enabled.contract == old.contract
    assert all(torch.equal(v, old.agent.state_dict()[k]) for k, v in enabled.agent.state_dict().items())
    add(enabled.measured_credit_bank, REGIONS[0], 0)
    enabled.directory.mkdir()
    enabled.save(final=True)
    checkpoint = next(enabled.directory.glob('checkpoint_*.pt'))
    restored = ActualFlapResidualSACPilot(warm, physical, tmp_path/'restored', stage,
        checkpoint=checkpoint)
    assert restored.measured_train_credit == measured_credit_config(TERMINAL_VARIANT)
    assert restored.measured_credit_bank.report() == enabled.measured_credit_bank.report()
    with pytest.raises(ValueError, match='differs from checkpoint'):
        ActualFlapResidualSACPilot(warm, physical, tmp_path/'invalid', stage,
            checkpoint=checkpoint, measured_train_credit=VARIANT)
