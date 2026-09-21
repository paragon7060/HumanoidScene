"""CPU tests for the v2 rack spawn planner; never launch Isaac Sim."""

from dataclasses import replace
import math

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import (
    BOX_TYPE_IDS,
    DEPTH_GAP_M,
    LOW_LEVEL_FRONT_DEPTH_JITTER_M,
    RACK_BOX_FRONT_REFERENCE_DEPTH_RAW,
    RACK_BOX_REAR_MARGIN_M,
    REGION_DEPTHS_RAW,
    logical_cells,
    physical_asset_names,
    physical_asset_types,
    sample_spawn_batch,
)
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M
from kuavo_isaaclab_scene.workcell.rack_box_layout import RACK_RAMP_BACK_DEPTH_RAW


def generator(seed=7):
    return torch.Generator(device="cpu").manual_seed(seed)


def test_low_level_spawns_one_box_across_all_regions_and_allowed_types():
    spec = replace(MultiBoxSpec(), strategy="staged", skill="grasp")
    batch = sample_spawn_batch(spec, 512, generator=generator())
    assert torch.equal(batch.counts, torch.ones_like(batch.counts))
    assert torch.equal(batch.active.sum(-1), torch.ones(512, dtype=torch.long))
    selected_regions = batch.region_ids[batch.active]
    assert set(selected_regions.tolist()) == {0, 1, 2, 3}
    for region_id in (2, 3):
        mask = batch.active & (batch.region_ids == region_id)
        assert set(batch.box_type_ids[mask].tolist()) == {BOX_TYPE_IDS["small"]}
    shelf_two = batch.active & (batch.region_ids < 2)
    assert set(batch.box_type_ids[shelf_two].tolist()) == {
        BOX_TYPE_IDS["small"], BOX_TYPE_IDS["medium"]}
    depths = -batch.rack_local_positions[..., 1][batch.active]
    assert bool((depths >= RACK_BOX_FRONT_REFERENCE_DEPTH_RAW
                 - LOW_LEVEL_FRONT_DEPTH_JITTER_M - 1e-7).all())
    assert bool((depths <= RACK_BOX_FRONT_REFERENCE_DEPTH_RAW
                 + LOW_LEVEL_FRONT_DEPTH_JITTER_M + 1e-7).all())


def test_full_task_count_is_random_and_region_load_is_balanced():
    batch = sample_spawn_batch(MultiBoxSpec(), 512, generator=generator(11))
    assert int(batch.counts.min()) >= 1 and int(batch.counts.max()) <= 12
    assert len(torch.unique(batch.counts)) > 1
    assert torch.equal(batch.active.sum(-1), batch.counts)
    for env_id in range(len(batch.counts)):
        loads = torch.stack([(batch.region_ids[env_id] == region).sum() for region in range(4)])
        assert int(loads.max() - loads.min()) <= 1


def test_full_task_packs_each_region_from_front_using_actual_box_depths():
    batch = sample_spawn_batch(MultiBoxSpec(), 512, generator=generator(13))
    dimensions = torch.tensor([BOX_DIMENSIONS_M[name] for name in ("small", "medium")])
    for env_id in range(len(batch.counts)):
        for region_id in range(4):
            mask = batch.active[env_id] & (batch.region_ids[env_id] == region_id)
            if not bool(mask.any()):
                continue
            depths = -batch.rack_local_positions[env_id, mask, 1]
            half_depths = dimensions[batch.box_type_ids[env_id, mask], 1] / 2.0
            order = torch.argsort(depths)
            depths = depths[order]
            half_depths = half_depths[order]
            assert float(depths[0]) == pytest.approx(RACK_BOX_FRONT_REFERENCE_DEPTH_RAW)
            if len(depths) > 1:
                expected = half_depths[:-1] + DEPTH_GAP_M + half_depths[1:]
                torch.testing.assert_close(depths[1:] - depths[:-1], expected)
            assert float(depths[-1] + half_depths[-1]) <= (
                RACK_RAMP_BACK_DEPTH_RAW - RACK_BOX_REAR_MARGIN_M + 1e-6)


def test_all_twelve_boxes_have_unique_physical_assets_and_nonoverlapping_poses():
    spec = replace(MultiBoxSpec(), full_spawn_count_range=(12, 12))
    batch = sample_spawn_batch(spec, 64, generator=generator(19))
    cells = logical_cells(spec)
    dimensions = torch.tensor([BOX_DIMENSIONS_M[name] for name in ("small", "medium")])
    assert len(physical_asset_names()) == 18
    assert physical_asset_types() == ("small",) * 6 + ("medium",) * 6 + ("small",) * 6
    assert batch.active.all()
    for env_id in range(64):
        assert len(set(batch.pool_ids[env_id].tolist())) == 12
        for first in range(12):
            for second in range(first + 1, 12):
                if cells[first].shelf != cells[second].shelf:
                    continue
                delta = (batch.rack_local_positions[env_id, first, :2]
                         - batch.rack_local_positions[env_id, second, :2]).abs()
                first_size = dimensions[batch.box_type_ids[env_id, first], :2]
                second_size = dimensions[batch.box_type_ids[env_id, second], :2]
                # Separation along either rack-local horizontal axis prevents overlap.
                assert bool((delta >= (first_size + second_size) / 2).any())


def test_spawn_poses_and_anchor_jitter_are_finite_bounded_and_reproducible():
    spec = MultiBoxSpec()
    assert math.degrees(spec.rack_yaw_jitter) == pytest.approx(15.0)
    first = sample_spawn_batch(spec, 128, generator=generator(29))
    second = sample_spawn_batch(spec, 128, generator=generator(29))
    for name in first.__dataclass_fields__:
        assert torch.equal(getattr(first, name), getattr(second, name))
    assert torch.isfinite(first.rack_local_positions[first.active]).all()
    norms = first.rack_local_quaternions[first.active].norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6)
    assert (first.rack_xy_delta.abs() <= torch.tensor(spec.rack_xy_jitter) + 1e-7).all()
    assert (first.rack_yaw_delta.abs() <= spec.rack_yaw_jitter + 1e-7).all()
    assert math.degrees(float(first.rack_yaw_delta.min())) < -14.0
    assert math.degrees(float(first.rack_yaw_delta.max())) > 14.0
    assert (first.conveyor_xy_delta.abs() <= torch.tensor(spec.conveyor_xy_jitter) + 1e-7).all()
    assert (first.conveyor_yaw_delta.abs() <= spec.conveyor_yaw_jitter + 1e-7).all()


def test_depth_cells_use_measured_front_reference_and_largest_box_spacing():
    assert REGION_DEPTHS_RAW == pytest.approx((0.21, 0.47, 0.73))
    medium_half_depth = BOX_DIMENSIONS_M["medium"][1] / 2.0
    assert REGION_DEPTHS_RAW[-1] + medium_half_depth < 0.85102


def test_roller_clearance_raises_box_roots_without_changing_horizontal_reference():
    spec = replace(MultiBoxSpec(), strategy="staged", skill="grasp")
    plain = sample_spawn_batch(spec, 32, generator=generator(41))
    rollers = sample_spawn_batch(
        spec, 32, generator=generator(41), rack_surface_extra_clearance_m=0.01)
    torch.testing.assert_close(
        rollers.rack_local_positions[..., :2], plain.rack_local_positions[..., :2])
    dz = rollers.rack_local_positions[..., 2] - plain.rack_local_positions[..., 2]
    torch.testing.assert_close(dz[plain.active], torch.full_like(dz[plain.active], 0.01))
