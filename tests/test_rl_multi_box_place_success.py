"""CPU checks for reward-independent privileged place success."""

import math

import torch

from kuavo_isaaclab_scene.rl.multi_box.success import PlaceSuccessInput, PlaceSuccessTracker


def sample(num_envs=7):
    corners = torch.tensor([
        [-0.20, -0.10], [-0.20, 0.10], [0.20, 0.10], [0.20, -0.10],
    ]).repeat(num_envs, 1, 1)
    return PlaceSuccessInput(
        belt_support=torch.ones(num_envs, dtype=torch.bool),
        gripper_grasping=torch.zeros(num_envs, 2, dtype=torch.bool),
        gripper_box_distance_m=torch.full((num_envs, 2), 0.02),
        box_footprint_corners_belt=corners,
        belt_half_extents_xy=torch.tensor([1.275, 0.34]),
        overlaps_placed_box=torch.zeros(num_envs, dtype=torch.bool),
        long_axis_yaw_error_rad=torch.zeros(num_envs),
        linear_speed_mps=torch.full((num_envs,), 0.05),
        angular_speed_radps=torch.full((num_envs,), 0.20),
    )


def test_place_requires_every_approved_condition_continuously():
    values = sample()
    values.belt_support[0] = False
    values.gripper_grasping[1, 0] = True
    values.gripper_box_distance_m[2, 1] = 0.019
    values.box_footprint_corners_belt[3, 0, 1] = 0.341
    values.overlaps_placed_box[4] = True
    values.long_axis_yaw_error_rad[5] = math.radians(10.1)
    values.linear_speed_mps[6] = 0.051
    result = PlaceSuccessTracker(7, "cpu").update(values, 0.5)
    assert not result.success.any()
    assert result.instantaneous.tolist() == [False] * 7


def test_place_accepts_parallel_or_antiparallel_long_axis_after_half_second():
    values = sample(2)
    values.long_axis_yaw_error_rad[:] = torch.tensor([
        math.radians(10.0), math.pi - math.radians(10.0),
    ])
    tracker = PlaceSuccessTracker(2, "cpu")
    assert not tracker.update(values, 0.25).success.any()
    result = tracker.update(values, 0.25)
    assert result.success.all()


def test_place_hold_resets_after_contact_or_motion_loss():
    values = sample(1)
    tracker = PlaceSuccessTracker(1, "cpu")
    tracker.update(values, 0.4)
    values.belt_support[0] = False
    assert tracker.update(values, 0.01).hold_time_s.item() == 0
    values.belt_support[0] = True
    assert not tracker.update(values, 0.4).success.item()
