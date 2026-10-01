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


def test_invalid_physical_box_pose_has_zero_confidence_and_finite_observation():
    source = _source()
    source["physical_box_poses_world"][1, 10, 3:] = 0.0
    frame = simulated_perception_frame(**source)
    assert frame.boxes.pose_confidence[0, 1].item() == 1.0
    assert frame.boxes.pose_confidence[1, 7].item() == 0.0
    assert frame.boxes.pose_world[1, 7].tolist() == [0., 0., 0., 1., 0., 0., 0.]


def test_articulated_panel_poses_follow_logical_pool_mapping_and_inactive_mask():
    source = _source()
    panels = _identity(2, NUM_PHYSICAL_BOX_ASSETS, 2)
    panels[0, 5, 0, :3] = torch.tensor([1., 2., 3.2])
    panels[0, 5, 1, :3] = torch.tensor([1., 2.3, 3.2])
    panels[1, 10, :, 2] = 4.4
    frame = simulated_perception_frame(**source, physical_flap_center_poses_world=panels)
    torch.testing.assert_close(frame.boxes.flap_pose_world[0, 1], panels[0, 5])
    torch.testing.assert_close(frame.boxes.flap_pose_world[1, 7], panels[1, 10])
    assert frame.boxes.flap_pose_confidence[0, 1].tolist() == [1., 1.]
    assert frame.boxes.flap_pose_confidence[0, 0].tolist() == [0., 0.]
    assert frame.boxes.flap_pose_world[0, 0, :, 3].tolist() == [1., 1.]


def test_bad_panel_pose_is_untrusted_without_hiding_valid_box_pose():
    source = _source()
    panels = _identity(2, NUM_PHYSICAL_BOX_ASSETS, 2)
    panels[0, 5, 0, 3:] = 0
    panels[1, 10, 1, 0] = float("nan")
    frame = simulated_perception_frame(**source, physical_flap_center_poses_world=panels)
    assert frame.boxes.flap_pose_confidence[0, 1].tolist() == [0., 1.]
    assert frame.boxes.flap_pose_confidence[1, 7].tolist() == [1., 0.]
    assert frame.boxes.pose_confidence[0, 1] == 1
    assert torch.isfinite(frame.boxes.flap_pose_world).all()
