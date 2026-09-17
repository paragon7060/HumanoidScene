"""CPU checks for reward-independent privileged carry success."""

from dataclasses import replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.success import (
    CarrySuccessConfig,
    CarrySuccessInput,
    carry_success,
)


def sample(num_envs=5):
    corners = torch.tensor([
        [-0.20, -0.10], [-0.20, 0.10], [0.20, 0.10], [0.20, -0.10],
    ]).repeat(num_envs, 1, 1)
    return CarrySuccessInput(
        grasp_maintained=torch.ones(num_envs, dtype=torch.bool),
        box_footprint_corners_belt=corners,
        belt_half_extents_xy=torch.tensor([1.275, 0.34]),
        box_bottom_height_m=torch.full((num_envs,), 0.10),
        overlaps_placed_box=torch.zeros(num_envs, dtype=torch.bool),
    )


def test_carry_requires_grasp_full_footprint_free_space_and_height_range():
    values = sample()
    values.grasp_maintained[0] = False
    values.box_footprint_corners_belt[1, 0, 0] = 1.276
    values.overlaps_placed_box[2] = True
    values.box_bottom_height_m[3] = 0.049
    result = carry_success(values)
    assert result.success.tolist() == [False, False, False, False, True]
    assert not result.grasp_maintained[0]
    assert not result.footprint_inside_belt[1]
    assert not result.free_space[2]
    assert not result.pre_place_height[3]


def test_carry_accepts_approved_height_boundaries_and_batched_belt_extents():
    values = sample(2)
    values.box_bottom_height_m[:] = torch.tensor([0.05, 0.15])
    values = replace(values, belt_half_extents_xy=torch.tensor([[1.0, 0.3], [1.2, 0.3]]))
    assert carry_success(values).success.all()


def test_carry_height_config_must_be_ordered_and_finite():
    with pytest.raises(ValueError, match="ordered"):
        replace(CarrySuccessConfig(), min_pre_place_height_m=0.20).validate()
    with pytest.raises(ValueError, match="finite"):
        replace(CarrySuccessConfig(), max_pre_place_height_m=float("nan")).validate()
