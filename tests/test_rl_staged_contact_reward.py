"""No phantom contact/hold, terminal reward cycles, and partial reset."""
from copy import deepcopy

import pytest
import torch

from test_rl_contact_reward_profile import base_contract
from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
    with_contact_reward_profile, frozen_actor_reward_contract, configured_reward_weights)
from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import with_precision_capture_profile
from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import with_absorbing_geometry_profile
from kuavo_isaaclab_scene.rl.multi_box.rewards.staged_contact import (
    StagedContactProgress, staged_contact_contract, precision_readiness,
    with_staged_contact_profile, actor_only_physics_equal)


def tracker(n=1):
    return StagedContactProgress(n, 'cpu', discount=.999, config=staged_contact_contract())


def inputs(n=1):
    return dict(precision=torch.zeros(n), hand_pinching=torch.zeros(n, 2, dtype=torch.bool),
        hand_flap_index=torch.tensor([[0, 1]]).expand(n, -1).clone(),
        trainable=torch.ones(n, dtype=torch.bool), terminated=torch.zeros(n, dtype=torch.bool),
        assignment_changed=torch.zeros(n, dtype=torch.bool), dt=1/30)


def test_precision_requires_reach_alignment_and_capture_in_the_same_hand():
    distance = torch.tensor([[0., 0.], [0., 1.], [0., 0.]])
    alignment = torch.tensor([[1., 1.], [1., 1.], [1., 0.]])
    capture = torch.zeros(3, 2)
    result = precision_readiness(distance, alignment, capture)
    assert result[0] == 1 and result[1] == pytest.approx(.25) and result[2] == .25
    capture[0] = .1
    assert precision_readiness(distance, alignment, capture)[0] < .02


def test_no_command_or_distance_can_accumulate_actual_pinch_hold():
    t = tracker(); data = inputs(); data['precision'].fill_(1.)
    for _ in range(10): assert t.step(**data)['pinch_hold_progress'] == 0
    assert not t.pinching_time.any()
    data['hand_pinching'][0, 0] = True
    for _ in range(8): t.step(**data)
    assert t.previous_hold.item() == .25
    data['hand_pinching'][0, 1] = True
    for _ in range(8): t.step(**data)
    assert t.previous_hold.item() == 1.


def test_same_flap_two_hands_is_not_bilateral_and_flap_change_restarts_hold():
    t = tracker(); data = inputs(); data['hand_pinching'].fill_(True); data['hand_flap_index'].zero_()
    for _ in range(10): t.step(**data)
    assert t.previous_hold.item() == .25
    data['hand_flap_index'][0, 1] = 1
    t.step(**data)
    assert t.pinching_time[0, 1] == pytest.approx(1/30)
    assert t.previous_hold.item() < .4


def test_acquire_lose_timeout_cycle_has_zero_discounted_shaping_return():
    t = tracker(); data = inputs(); t.step(**data)
    rewards = []
    for i in range(14):
        data['precision'].fill_(1. if i < 11 else 0.)
        data['hand_pinching'].fill_(i < 11)
        data['terminated'].fill_(i == 13)
        rewards.append(sum(t.step(**data).values()).item())
    assert sum(.999**i * r for i, r in enumerate(rewards)) == pytest.approx(0., abs=2e-6)


def test_partial_reset_and_terminal_assignment_switch_close_existing_potentials():
    t = tracker(2); data = inputs(2); t.step(**data)
    data['precision'].fill_(1.); data['hand_pinching'].fill_(True)
    for _ in range(10): t.step(**data)
    t.reset(torch.tensor([1])); data['terminated'][0] = True; data['assignment_changed'].fill_(True)
    report = t.step(**data)
    assert report['precision_readiness_progress'][0] == -1.
    assert report['pinch_hold_progress'][0] == -1.
    assert report['precision_readiness_progress'][1] == 0 and report['pinch_hold_progress'][1] == 0


def test_reward_migration_only_allows_declared_actor_only_change():
    source = base_contract(); source['reward_profile']['capture_scale_m'] = .10
    old = with_absorbing_geometry_profile(with_precision_capture_profile(with_contact_reward_profile(source)))
    original = deepcopy(old); new = with_staged_contact_profile(old)
    assert old == original and actor_only_physics_equal(old, new)
    assert frozen_actor_reward_contract(new) == frozen_actor_reward_contract(old)
    assert configured_reward_weights(new['reward_profile']) == configured_reward_weights(old['reward_profile'])
    bad = deepcopy(new); bad['terminal_contract']['safety'] = 'changed'
    assert not actor_only_physics_equal(old, bad)
    bad = deepcopy(new); bad['reward_profile']['staged_contact']['precision_weight'] = 9.
    with pytest.raises(ValueError): frozen_actor_reward_contract(bad)
