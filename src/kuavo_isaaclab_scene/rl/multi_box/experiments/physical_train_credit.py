"""Experimental multi-step credit from completed, literal successful TRAIN paths.

The intermediate actions belong to the recorded behavior, not the current SAC
policy. This uncorrected auxiliary target is intentionally kept separate from
the online one-step Bellman target and never used as an evaluation outcome.
"""
from copy import deepcopy
import math

import torch

VARIANT = 'native-nstep10'


def training_credit_config(variant):
    if variant in (None, 'one-step'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown physical TRAIN credit variant')
    return dict(
        name='completed_safe_native_TRAIN_nstep10_retention_v1',
        horizon=10, batch_size=64, terminal_window_fraction=.25, critic_weight=1.,
        actor_body_weight=10., actor_jaw_weight=.1, prior_body_weight=1.,
        prior_fade_actor_updates=5000,
        online_target='unchanged_one_step_soft_SAC',
        auxiliary_target='actual_discounted_behavior_rewards_plus_endpoint_soft_SAC_value',
        off_policy_correction=False, intermediate_entropy_included=False,
        episode_crossing_allowed=False, evaluation_import_allowed=False,
    )


def resolve_training_credit(saved_contract, requested_variant):
    if saved_contract is None:
        return training_credit_config(requested_variant)
    stored = saved_contract.get('training_credit')
    if stored is not None and stored != training_credit_config(VARIANT):
        raise ValueError('Saved physical TRAIN credit configuration differs')
    if requested_variant is not None and training_credit_config(requested_variant) != stored:
        raise ValueError('Requested physical TRAIN credit differs from checkpoint')
    return deepcopy(stored)


def discounted_episode_windows(rows, *, horizon, gamma):
    """Build forward windows; each first action and endpoint remain recorded.

    This accepts a whole successful episode already validated by the TRAIN
    bank, then additionally rejects discontinuous observation chains. Short
    terminal windows stop at the actual terminal and have zero bootstrap.
    """
    if type(horizon) is not int or not 1 <= horizon <= 900 or not math.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError('Finite discount and bounded integer horizon required')
    n = len(rows['reward'])
    terminal = rows['terminated']
    if not n or terminal.dtype != torch.bool or not bool(terminal[-1]) or bool(terminal[:-1].any()):
        raise ValueError('Multi-step credit requires one complete actual terminal episode')
    for current, following in (('actor_obs', 'next_actor_obs'), ('critic_obs', 'next_critic_obs')):
        if not torch.equal(rows[following][:-1], rows[current][1:]):
            raise ValueError('Multi-step credit cannot join discontinuous physical states')
    reward = rows['reward']
    starts = torch.arange(n, device=reward.device)
    offsets = torch.arange(horizon, device=reward.device)
    indices = starts[:, None] + offsets
    valid = indices < n
    discounts = gamma ** offsets.to(reward)
    discounted = (reward[indices.clamp_max(n - 1)] * discounts * valid).sum(-1)
    steps = (n - starts).clamp_max(horizon)
    endpoints = starts + steps - 1
    terminated = terminal[endpoints]
    bootstrap_discount = gamma ** steps.to(reward)
    bootstrap_discount = bootstrap_discount.masked_fill(terminated, 0.)
    return dict(
        actor_obs=rows['actor_obs'], critic_obs=rows['critic_obs'], action=rows['action'],
        next_actor_obs=rows['next_actor_obs'][endpoints],
        next_critic_obs=rows['next_critic_obs'][endpoints], reward=discounted,
        terminated=terminated, bootstrap_discount=bootstrap_discount, n_steps=steps,
    )
