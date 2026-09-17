"""Pure tensor planner for randomized rack-box scenes.

The planner describes twelve logical boxes.  Shelf-2 logical positions map to
separate small/medium physical candidates so each vectorized environment may
choose its box type independently without respawning USD assets.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..spec import BOX_TYPES, MAX_BOXES, MultiBoxSpec
from ...scenes.layout import RACK_SLOPE_RAD
from ....workcell.rack_box_layout import (
    BOX_DIMENSIONS_M,
    RACK_RAMP_BACK_DEPTH_RAW,
    RACK_SHELF_CENTER_LOCAL_X_RAW,
    RACK_SURFACE_CLEARANCE_M,
)
from ....workcell.workcell_layout import RACK_RAW_TIER_RANGES


POSITIONS_PER_REGION = 3
REGION_DEPTHS_RAW = (0.16, 0.455, 0.75)
COLUMN_GAP_M = 0.04
BOX_TYPE_IDS = {name: index for index, name in enumerate(BOX_TYPES)}


@dataclass(frozen=True)
class RackCell:
    logical_id: int
    region_id: int
    region_name: str
    shelf: int
    side: str
    depth_index: int

    def local_pose(self, box_type: str, rack_scale=(1.0, 1.0, 1.0)):
        """Return rack-local root position/quaternion for one physical box."""
        if box_type not in BOX_DIMENSIONS_M:
            raise ValueError(f"Unknown physical box type: {box_type}")
        sx, sy, sz = (float(value) for value in rack_scale)
        max_width = BOX_DIMENSIONS_M["medium"][0]
        # From the robot-facing rack front, local -X is right and +X is left.
        side_sign = -1.0 if self.side == "right" else 1.0
        local_x = RACK_SHELF_CENTER_LOCAL_X_RAW * sx + side_sign * (max_width + COLUMN_GAP_M) / 2.0
        depth_raw = REGION_DEPTHS_RAW[self.depth_index]
        local_y = -depth_raw * sy
        surface_raw = RACK_RAW_TIER_RANGES[self.shelf - 1][1] - math.tan(RACK_SLOPE_RAD) * (
            RACK_RAMP_BACK_DEPTH_RAW - depth_raw)
        # The physical wrapper's body root sits about 0.5% of body height above its bottom.
        bottom_offset = 0.005 * BOX_DIMENSIONS_M[box_type][2]
        local_z = surface_raw * sz + bottom_offset + RACK_SURFACE_CLEARANCE_M
        half_angle = -RACK_SLOPE_RAD / 2.0
        return ((local_x, local_y, local_z),
                (math.cos(half_angle), math.sin(half_angle), 0.0, 0.0))


@dataclass
class SpawnBatch:
    counts: torch.Tensor
    active: torch.Tensor
    box_type_ids: torch.Tensor
    region_ids: torch.Tensor
    pool_ids: torch.Tensor
    rack_local_positions: torch.Tensor
    rack_local_quaternions: torch.Tensor
    rack_xy_delta: torch.Tensor
    rack_yaw_delta: torch.Tensor
    conveyor_xy_delta: torch.Tensor
    conveyor_yaw_delta: torch.Tensor


def logical_cells(spec: MultiBoxSpec) -> tuple[RackCell, ...]:
    """Three depth positions in each of the four semantic rack regions."""
    cells = tuple(
        RackCell(region_id * POSITIONS_PER_REGION + depth_index,
                 region_id, region.name, region.shelf, region.side, depth_index)
        for region_id, region in enumerate(spec.rack_regions)
        for depth_index in range(POSITIONS_PER_REGION)
    )
    if len(cells) != spec.max_boxes:
        raise ValueError(
            f"Rack cells ({len(cells)}) must equal the logical N_max ({spec.max_boxes}).")
    return cells


def physical_asset_names() -> tuple[str, ...]:
    """Eighteen fixed assets backing twelve environment-specific logical boxes."""
    return tuple(
        [f"mb_s2_small_{index}" for index in range(6)]
        + [f"mb_s2_medium_{index}" for index in range(6)]
        + [f"mb_s3_small_{index}" for index in range(6)]
    )


def physical_asset_types() -> tuple[str, ...]:
    return ("small",) * 6 + ("medium",) * 6 + ("small",) * 6


def physical_pool_id(cell: RackCell, box_type_id: int) -> int:
    if cell.shelf == 2:
        shelf_index = cell.region_id * POSITIONS_PER_REGION + cell.depth_index
        if box_type_id == BOX_TYPE_IDS["small"]:
            return shelf_index
        if box_type_id == BOX_TYPE_IDS["medium"]:
            return 6 + shelf_index
        raise ValueError("Shelf 2 supports only small and medium boxes.")
    if cell.shelf == 3 and box_type_id == BOX_TYPE_IDS["small"]:
        shelf_index = (cell.region_id - 2) * POSITIONS_PER_REGION + cell.depth_index
        return 12 + shelf_index
    raise ValueError("Shelf 3 supports only small boxes.")


def _symmetric(shape, limits, *, device, generator):
    limit = torch.as_tensor(limits, dtype=torch.float32, device=device)
    return (2.0 * torch.rand(shape, device=device, generator=generator) - 1.0) * limit


def _balanced_cell_order(cells, *, device, generator):
    """Randomize cells while keeping region counts within one of each other."""
    depth_orders = [torch.randperm(POSITIONS_PER_REGION, device=device, generator=generator).tolist()
                    for _ in range(4)]
    result = []
    for round_index in range(POSITIONS_PER_REGION):
        region_order = torch.randperm(4, device=device, generator=generator).tolist()
        for region_id in region_order:
            depth = depth_orders[region_id][round_index]
            result.append(cells[region_id * POSITIONS_PER_REGION + depth].logical_id)
    return result


def sample_spawn_batch(
    spec: MultiBoxSpec,
    num_envs: int,
    *,
    device: str | torch.device = "cpu",
    generator: torch.Generator | None = None,
    rack_scale=(1.0, 1.0, 1.0),
) -> SpawnBatch:
    """Sample active logical boxes, physical variants, poses, and anchor jitter."""
    spec.validate()
    if num_envs < 1:
        raise ValueError("num_envs must be positive.")
    device = torch.device(device)
    cells = logical_cells(spec)
    low, high = spec.spawn_count_range
    if low == high:
        counts = torch.full((num_envs,), low, dtype=torch.long, device=device)
    else:
        counts = torch.randint(low, high + 1, (num_envs,), device=device, generator=generator)

    active = torch.zeros(num_envs, MAX_BOXES, dtype=torch.bool, device=device)
    box_type_ids = torch.full((num_envs, MAX_BOXES), -1, dtype=torch.long, device=device)
    region_ids = torch.full_like(box_type_ids, -1)
    pool_ids = torch.full_like(box_type_ids, -1)
    positions = torch.zeros(num_envs, MAX_BOXES, 3, dtype=torch.float32, device=device)
    quaternions = torch.zeros(num_envs, MAX_BOXES, 4, dtype=torch.float32, device=device)

    for env_id, count in enumerate(counts.tolist()):
        selected = _balanced_cell_order(cells, device=device, generator=generator)[:count]
        for logical_id in selected:
            cell = cells[logical_id]
            region = spec.rack_regions[cell.region_id]
            type_choice = int(torch.randint(
                len(region.allowed_box_types), (1,), device=device, generator=generator).item())
            box_type = region.allowed_box_types[type_choice]
            type_id = BOX_TYPE_IDS[box_type]
            position, quaternion = cell.local_pose(box_type, rack_scale)
            active[env_id, logical_id] = True
            box_type_ids[env_id, logical_id] = type_id
            region_ids[env_id, logical_id] = cell.region_id
            pool_ids[env_id, logical_id] = physical_pool_id(cell, type_id)
            positions[env_id, logical_id] = torch.tensor(position, device=device)
            quaternions[env_id, logical_id] = torch.tensor(quaternion, device=device)

    return SpawnBatch(
        counts=counts,
        active=active,
        box_type_ids=box_type_ids,
        region_ids=region_ids,
        pool_ids=pool_ids,
        rack_local_positions=positions,
        rack_local_quaternions=quaternions,
        rack_xy_delta=_symmetric((num_envs, 2), spec.rack_xy_jitter,
                                 device=device, generator=generator),
        rack_yaw_delta=_symmetric((num_envs,), spec.rack_yaw_jitter,
                                  device=device, generator=generator),
        conveyor_xy_delta=_symmetric((num_envs, 2), spec.conveyor_xy_jitter,
                                     device=device, generator=generator),
        conveyor_yaw_delta=_symmetric((num_envs,), spec.conveyor_yaw_jitter,
                                      device=device, generator=generator),
    )
