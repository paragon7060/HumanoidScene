"""Scarce real success, region fallback, physical evidence and return horizon."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    BALANCED_RETURN_VARIANT, EPISODE_RETURN_VARIANT, MeasuredTrainCreditBank,
    measured_credit_config, validate_measured_credit_batch,
)
from test_rl_measured_train_credit import path


def complete_path(success, env, region=0, n=40):
    rows, outcome = path(success, env, n)
    name = ('shelf_2_right', 'shelf_2_left', 'shelf_3_right', 'shelf_3_left')[region]
    outcome['layout']['target_region'] = name
    for key in ('actor_obs', 'next_actor_obs'):
        rows[key][:, 94:98] = 0
        rows[key][:, 94+region] = 1
    for key in ('actor_obs', 'next_actor_obs', 'critic_obs', 'next_critic_obs'):
        rows[key][:, 1] = env
    if success:
        outcome['result'].update(pinching=[True, True], stable_hands=[True, True],
            opposing_flaps=True, proof_lift=True, hold_time_s=.267,
            rack_clearance_m=.01, unsafe_causes={})
    return rows, outcome


def test_balance_scarce_success_per_region_and_failure_only_fallback():
    config = measured_credit_config(BALANCED_RETURN_VARIANT)
    bank = MeasuredTrainCreditBank(518, 577, .999, config)
    originals = []
    for env, success, region in [(0, True, 0), *[(i, False, 0) for i in range(1, 9)],
                                  (9, False, 1)]:
        rows, outcome = complete_path(success, env, region)
        originals.append((deepcopy(rows), rows))
        bank.add_episode(rows, outcome, source_run='real_TRAIN_fixture')
    sampled = bank.sample(128, 'cpu')
    ids = sampled['actor_obs'][:, 1]
    assert ids.eq(0).sum() == 32
    assert ids.eq(9).sum() == 64
    assert sampled['actor_obs'][:, 94:98].argmax(-1).eq(0).sum() == 64
    assert sampled['terminated'].all() and sampled['bootstrap_discount'].eq(0).all()
    assert all(torch.equal(original[k], actual[k]) for original, actual in originals for k in original)
    agent = SimpleNamespace(actor_obs_dim=518, critic_obs_dim=577,
        config=SimpleNamespace(gamma=.999), measured_train_credit_config=config)
    validate_measured_credit_batch(agent, sampled, .1)
    assert sampled['n_steps'].max() > 16
    restored = MeasuredTrainCreditBank(518, 577, .999, config)
    restored.restore(bank.state())
    assert restored.report() == bank.report()
    ordinary = MeasuredTrainCreditBank(518, 577, .999, measured_credit_config(EPISODE_RETURN_VARIANT))
    with pytest.raises(ValueError, match='differs'):
        ordinary.restore(bank.state())


@pytest.mark.parametrize('mutation', [
    lambda o: o.update(split='validation'),
    lambda o: o['result'].update(unsafe=True),
    lambda o: o['result'].update(proof_lift=False),
    lambda o: o['result'].update(hold_time_s=.1),
])
def test_balanced_success_requires_actual_safe_TRAIN_physical_evidence(mutation):
    bank = MeasuredTrainCreditBank(518, 577, .999, measured_credit_config(BALANCED_RETURN_VARIANT))
    rows, outcome = complete_path(True, 0)
    mutation(outcome)
    with pytest.raises(ValueError):
        bank.add_episode(rows, outcome, source_run='fixture')
    assert bank.size == 0
