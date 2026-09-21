"""Map randomized physical asset poses into 12 deployable box tokens."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.state.perception import (
    NUM_PHYSICAL_BOX_ASSETS,
    simulated_perception_frame,
)
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M


def _identity(*shape):
    values = torch.zeros(*shape, 7)
    values[..., 3] = 1.0
    return values


def _source():
    active = torch.zeros(2, 12, dtype=torch.bool)
    active[0, 1] = True
    active[1, 7] = True
    types = torch.full((2, 12), -1, dtype=torch.long)
    types[0, 1] = 0
    types[1, 7] = 1
    regions = torch.full_like(types, -1)
    regions[0, 1] = 0
    regions[1, 7] = 1
    pool = torch.full_like(types, -1)
    pool[0, 1] = 5
    pool[1, 7] = 10
    physical = _identity(2, NUM_PHYSICAL_BOX_ASSETS)
    physical[0, 5, :3] = torch.tensor([1.0, 2.0, 3.0])
    physical[1, 10, :3] = torch.tensor([-1.0, -2.0, 4.0])
    return dict(active=active, box_type_id=types, rack_region_id=regions,
                pool_id=pool, physical_box_poses_world=physical,
                rack_pose_world=_identity(2), conveyor_pose_world=_identity(2))


def test_simulated_perception_gathers_per_environment_pool_poses_and_masks_inactive():
    frame = simulated_perception_frame(**_source())
    assert frame.boxes.pose_world[0, 1, :3].tolist() == [1.0, 2.0, 3.0]
    assert frame.boxes.pose_world[1, 7, :3].tolist() == [-1.0, -2.0, 4.0]
    assert frame.boxes.pose_world[0, 0].tolist() == [0., 0., 0., 1., 0., 0., 0.]
    assert frame.boxes.pose_confidence[0, 1].item() == 1.0
    assert frame.boxes.pose_confidence[0, 0].item() == 0.0
    torch.testing.assert_close(frame.boxes.size_m[0, 1], torch.tensor(BOX_DIMENSIONS_M["small"]))
    torch.testing.assert_close(frame.boxes.size_m[1, 7], torch.tensor(BOX_DIMENSIONS_M["medium"]))
    assert frame.boxes.size_m[0, 0].tolist() == [0.0, 0.0, 0.0]


def test_active_logical_box_requires_valid_pool_and_type():
    source = _source()
    source["pool_id"][0, 1] = -1
    with pytest.raises(ValueError, match="physical pool"):
        simulated_perception_frame(**source)
    source["pool_id"][0, 1] = 5
    source["box_type_id"][0, 1] = -1
    with pytest.raises(ValueError, match="box type"):
        simulated_perception_frame(**source)
