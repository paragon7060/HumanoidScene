"""Deployable perception frame and an explicit simulator-truth source adapter.

The policy consumes ``PerceptionFrame``.  This module maps exact Isaac poses
into that interface for initial training; a real perception backend can supply
the same frame without importing simulator-only state or contact measurements.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..spec import BOX_TYPES, MAX_BOXES
from ..scene.spawn import physical_asset_names
from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from .schema import DeployableBoxState


NUM_PHYSICAL_BOX_ASSETS = len(physical_asset_names())


@dataclass(frozen=True)
class PerceptionFrame:
    boxes: DeployableBoxState
    rack_pose_world: torch.Tensor
    conveyor_pose_world: torch.Tensor

    def validate(self, num_envs: int) -> None:
        self.boxes.validate(num_envs)
        for name in ("rack_pose_world", "conveyor_pose_world"):
            pose = getattr(self, name)
            if pose.shape != (num_envs, 7) or not pose.is_floating_point():
                raise ValueError(f"{name} must be floating point [num_envs, 7].")
        devices = {self.boxes.pose_world.device, self.rack_pose_world.device,
                   self.conveyor_pose_world.device}
        if len(devices) != 1:
            raise ValueError("Perception tensors must share one device.")


def simulated_perception_frame(
    *,
    active: torch.Tensor,
    box_type_id: torch.Tensor,
    rack_region_id: torch.Tensor,
    pool_id: torch.Tensor,
    physical_box_poses_world: torch.Tensor,
    rack_pose_world: torch.Tensor,
    conveyor_pose_world: torch.Tensor,
) -> PerceptionFrame:
    """Gather one active physical asset per logical box with confidence one.

    This function is intentionally named for the simulator.  It does not add
    random pose noise, hidden velocity, contact force, or success labels.
    """
    if active.ndim != 2 or active.shape[1] != MAX_BOXES or active.dtype != torch.bool:
        raise ValueError("active must be boolean [num_envs, 12].")
    n = active.shape[0]
    indices = (("box_type_id", box_type_id), ("rack_region_id", rack_region_id),
               ("pool_id", pool_id))
    for name, value in indices:
        if value.shape != active.shape or value.dtype != torch.long:
            raise ValueError(f"{name} must be torch.long [num_envs, 12].")
    expected_pool_shape = (n, NUM_PHYSICAL_BOX_ASSETS, 7)
    if physical_box_poses_world.shape != expected_pool_shape \
            or not physical_box_poses_world.is_floating_point():
        raise ValueError(f"physical_box_poses_world must be floating point {expected_pool_shape}.")
    devices = {value.device for value in (active, box_type_id, rack_region_id,
               pool_id, physical_box_poses_world, rack_pose_world, conveyor_pose_world)}
    if len(devices) != 1:
        raise ValueError("Simulator pose tensors must share one device.")
    if bool((active & ((pool_id < 0) | (pool_id >= NUM_PHYSICAL_BOX_ASSETS))).any()):
        raise ValueError("An active logical box has no valid physical pool asset.")
    if bool((active & ((box_type_id < 0) | (box_type_id >= len(BOX_TYPES)))).any()):
        raise ValueError("An active logical box has no valid box type.")
    if bool((active & ((rack_region_id < 0) | (rack_region_id >= 4))).any()):
        raise ValueError("An active logical box has no valid rack region.")

    rows = torch.arange(n, device=active.device)[:, None]
    gathered = physical_box_poses_world[rows, pool_id.clamp(0, NUM_PHYSICAL_BOX_ASSETS - 1)]
    identity_pose = torch.zeros_like(gathered)
    identity_pose[..., 3] = 1.0
    pose = torch.where(active[..., None], gathered, identity_pose)
    sizes_by_type = torch.as_tensor(
        [BOX_DIMENSIONS_M[name] for name in BOX_TYPES],
        dtype=pose.dtype, device=pose.device)
    sizes = sizes_by_type[box_type_id.clamp(0, len(BOX_TYPES) - 1)] \
        * active[..., None]
    frame = PerceptionFrame(
        boxes=DeployableBoxState(
            active=active.clone(),
            box_type_id=box_type_id.clone(),
            rack_region_id=rack_region_id.clone(),
            size_m=sizes,
            pose_world=pose,
            pose_confidence=active.to(pose.dtype),
        ),
        rack_pose_world=rack_pose_world.clone(),
        conveyor_pose_world=conveyor_pose_world.clone(),
    )
    frame.validate(n)
    return frame
