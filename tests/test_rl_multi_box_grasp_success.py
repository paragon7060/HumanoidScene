"""CPU checks for the reward-independent privileged grasp predicate."""

from dataclasses import replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.success import (
    GraspSuccessConfig,
    GraspSuccessInput,
    GraspSuccessTracker,
)


def sample(num_envs=4):
    return GraspSuccessInput(
        hand_pinching=torch.ones(num_envs, 2, dtype=torch.bool),
        hand_flap_index=torch.tensor([[0, 1]]).repeat(num_envs, 1),
        relative_pose_stable=torch.ones(num_envs, 2, dtype=torch.bool),
        rack_clearance_m=torch.full((num_envs,), 0.008),
    )


def test_success_requires_continuous_opposing_bilateral_proof_lift():
    tracker = GraspSuccessTracker(4, "cpu")
    values = sample()
    for _ in range(4):
        result = tracker.update(values, 0.05)
        assert not result.success.any()
    result = tracker.update(values, 0.05)
    assert result.success.all()

    values.hand_flap_index[0] = torch.tensor([0, 0])
    values.hand_pinching[1, 1] = False
    values.relative_pose_stable[2, 0] = False
    values.rack_clearance_m[3] = 0.0079
    result = tracker.update(values, 0.05)
    assert not result.instantaneous.any()
    assert torch.equal(result.hold_time_s, torch.zeros(4))


def test_brief_loss_resets_hold_instead_of_latching_success():
    tracker = GraspSuccessTracker(1, "cpu")
    values = sample(1)
    tracker.update(values, 0.20)
    values.hand_pinching[0, 0] = False
    assert tracker.update(values, 0.01).hold_time_s.item() == 0
    values.hand_pinching[0, 0] = True
    assert not tracker.update(values, 0.20).success.item()


def test_partial_reset_cache_refresh_does_not_advance_other_environment_hold():
    tracker = GraspSuccessTracker(2, "cpu")
    values = sample(2)
    for step in range(4):
        tracker.update(values, 0.05, step_id=step)
    tracker.reset(torch.tensor([1]))
    values.hand_pinching[1] = False
    # Refreshing after environment 1 resets cannot fabricate a fifth physical
    # tick for environment 0 and consume its success reward early.
    duplicate = tracker.update(values, 0.05, step_id=3)
    assert duplicate.hold_time_s[0].item() == pytest.approx(0.20)
    assert not duplicate.success.any()
    actual_next = tracker.update(values, 0.05, step_id=4)
    assert actual_next.success.tolist() == [True, False]


def test_only_approved_flap_pair_and_proof_lift_ranges_are_accepted():
    GraspSuccessConfig().validate()
    with pytest.raises(ValueError, match="opposing"):
        replace(GraspSuccessConfig(), flap_names=("flap_front", "flap_back")).validate()
    with pytest.raises(ValueError, match="5-10 mm"):
        replace(GraspSuccessConfig(), proof_lift_m=0.02).validate()
    with pytest.raises(ValueError, match="0.2-0.3"):
        replace(GraspSuccessConfig(), hold_seconds=0.5).validate()
