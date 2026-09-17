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


def test_only_approved_flap_pair_and_proof_lift_ranges_are_accepted():
    GraspSuccessConfig().validate()
    with pytest.raises(ValueError, match="opposing"):
        replace(GraspSuccessConfig(), flap_names=("flap_front", "flap_back")).validate()
    with pytest.raises(ValueError, match="5-10 mm"):
        replace(GraspSuccessConfig(), proof_lift_m=0.02).validate()
    with pytest.raises(ValueError, match="0.2-0.3"):
        replace(GraspSuccessConfig(), hold_seconds=0.5).validate()
