"""Joint hand readiness and sustained actual pinch as absorbing potentials."""
from copy import deepcopy
import math

import torch


def staged_contact_contract():
    return dict(name='precision_and_pinch_hold_progress_v1',
        precision_weight=1., pinch_hold_weight=1.,
        precision_distance_scale_m=.03, precision_capture_scale_m=.025,
        precision='per_hand_reach_times_axis_alignment_times_capture_then_weak_hand',
        pinch_hold_seconds=.25, pinch_hold='continuous_actual_opposed_two_pad_pinch_on_same_flap',
        final_relative_pose_stability_success_predicate_unchanged=True,
        aggregation='0.25*(left+right)+0.5*min(left,right)',
        both_hands_on_same_flap_cannot_get_bilateral_score=True,
        reward='gamma_times_next_potential_minus_previous',
        terminal_potential_zero=True, reset_and_assignment_rebase=True,
        new_sensors_or_physics_substeps=False,
        source_Q_replay_optimizer_or_reward_labels_import_allowed=False)


def without_staged_contact(contract):
    profile = contract.get('reward_profile', {})
    if 'staged_contact' not in profile:
        return contract
    if profile['staged_contact'] != staged_contact_contract():
        raise ValueError('Unknown staged contact reward')
    result = deepcopy(contract)
    result['reward_profile'].pop('staged_contact')
    from .contact_profile import frozen_actor_reward_contract
    frozen_actor_reward_contract(result)
    if 'absorbing_geometry' not in profile or 'precision_capture' not in profile:
        raise ValueError('Staged contact requires reviewed absorbing precision geometry')
    return result


def with_staged_contact_profile(contract):
    from .contact_profile import frozen_actor_reward_contract
    frozen_actor_reward_contract(contract)
    profile = contract['reward_profile']
    if 'staged_contact' in profile or 'absorbing_geometry' not in profile or 'precision_capture' not in profile:
        raise ValueError('Fresh staged reward requires existing absorbing precision contact')
    result = deepcopy(contract)
    result['reward_profile']['staged_contact'] = staged_contact_contract()
    without_staged_contact(result)
    return result


def actor_only_physics_equal(source, destination):
    """Permit only this reward change for actor-only source compatibility.

    Q and replay compatibility must still use the full exact physical contract.
    """
    if source == destination:
        return True
    if 'staged_contact' not in destination.get('reward_profile', {}):
        return False
    return without_staged_contact(destination) == source


def precision_readiness(distance, alignment_cos, capture_error):
    score = (torch.exp(-distance.clamp_min(0) / .03)
        * alignment_cos.clamp(0, 1).square()
        * torch.exp(-capture_error.clamp_min(0) / .025))
    return .25 * score.sum(-1) + .5 * score.amin(-1)


class StagedContactProgress:
    def __init__(self, count, device, *, discount, config):
        if config != staged_contact_contract() or not 0 < discount <= 1:
            raise ValueError('Exact staged contact reward and discount required')
        self.discount = discount
        self.initialized = torch.zeros(count, dtype=torch.bool, device=device)
        self.pinching_time = torch.zeros(count, 2, device=device)
        self.previous_flap = torch.full((count, 2), -1, dtype=torch.long, device=device)
        self.previous_precision = torch.zeros(count, device=device)
        self.previous_hold = torch.zeros(count, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        self.initialized[ids] = False
        self.pinching_time[ids] = 0.
        self.previous_flap[ids] = -1
        self.previous_precision[ids] = self.previous_hold[ids] = 0.

    def step(self, *, precision, hand_pinching, hand_flap_index,
             trainable, terminated, assignment_changed, dt):
        if not math.isfinite(dt) or dt <= 0 or dt > .1:
            raise ValueError('A finite actual control interval in (0,.1] is required')
        n = len(self.initialized)
        if precision.shape != (n,) or any(x.shape != (n, 2) for x in
                (hand_pinching, hand_flap_index)) \
                or any(x.shape != (n,) or x.dtype != torch.bool for x in
                       (trainable, terminated, assignment_changed)) \
                or hand_pinching.dtype != torch.bool \
                or hand_flap_index.dtype != torch.long:
            raise ValueError('Staged contact requires matching measured per-hand inputs')
        precision = torch.nan_to_num(precision, nan=0., posinf=0., neginf=0.).clamp(0, 1)
        valid = hand_pinching & (hand_flap_index >= 0) & (hand_flap_index < 2)
        valid &= trainable[:, None]
        previous_time = torch.where(hand_flap_index == self.previous_flap, self.pinching_time, 0.)
        self.pinching_time = torch.where(valid, (previous_time + dt).clamp_max(.25), 0.)
        self.previous_flap.copy_(torch.where(valid, hand_flap_index, -1))
        held = (self.pinching_time / .25).clamp(0, 1)
        distinct = (hand_flap_index[:, 0] != hand_flap_index[:, 1]) & valid.all(-1)
        hold = .25 * held.sum(-1) + .5 * torch.where(distinct, held.amin(-1), 0.)
        hold = torch.where(valid.all(-1) & ~distinct, .25 * held.amax(-1), hold)
        current_precision = torch.where(trainable & ~terminated, precision, 0.)
        current_hold = torch.where(trainable & ~terminated, hold, 0.)
        previous_precision = torch.where(self.initialized & (~assignment_changed | terminated),
            self.previous_precision, self.discount * current_precision)
        previous_hold = torch.where(self.initialized, self.previous_hold, self.discount * current_hold)
        result = dict(
            precision_readiness_progress=self.discount * current_precision - previous_precision,
            pinch_hold_progress=self.discount * current_hold - previous_hold)
        self.previous_precision.copy_(current_precision)
        self.previous_hold.copy_(current_hold)
        self.initialized.copy_(trainable)
        return {k: torch.where(trainable, v, 0.) for k, v in result.items()}
