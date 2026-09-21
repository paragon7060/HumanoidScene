"""Read-only carry-success probe using the v2 conveyor geometry."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.belt import BELT_HALF_EXTENTS_XY
from ..success import CarrySuccessInput, CarrySuccessResult, carry_success
from .grasp_probe import GraspProbeResult


@dataclass(frozen=True)
class CarryProbeResult:
    result: CarrySuccessResult
    grasp_previously_completed: bool

    def report(self) -> str:
        value = self.result
        return (
            f"CARRY PROBE | grasp-seen={int(self.grasp_previously_completed)} "
            f"maintained={int(value.grasp_maintained[0])} "
            f"tilt<=20deg:{int(value.box_tilt_ok[0])} "
            f"belt={int(value.footprint_inside_belt[0])} "
            f"free={int(value.free_space[0])} "
            f"height=5-15cm:{int(value.pre_place_height[0])} "
            f"success={int(value.success[0])}"
        )


class QuestCarryProbe:
    """Requires an observed grasp success before carry can be counted."""

    def __init__(self, device: str | torch.device):
        self.device = torch.device(device)
        self.belt_half_extents_xy = torch.tensor(BELT_HALF_EXTENTS_XY, device=self.device)
        self.grasp_previously_completed = False
        self.target_logical_id = -1

    def reset(self) -> None:
        self.grasp_previously_completed = False
        self.target_logical_id = -1

    def update(self, snapshot, grasp: GraspProbeResult) -> CarryProbeResult:
        if snapshot.target_logical_id != self.target_logical_id:
            self.reset()
            self.target_logical_id = snapshot.target_logical_id
        self.grasp_previously_completed |= bool(grasp.success.success[0].item())
        # Relative-pose stability belongs to the short proof-lift used to
        # establish the grasp.  Requiring the same 10 mm / 10 degree anchor
        # throughout transport rejects a physically valid pinch after normal
        # box compliance or small in-hand settling.  Once grasp success has
        # been observed, privileged carry truth only requires the approved
        # opposing-flap contacts to remain present.
        maintained = (
            grasp.success.bilateral_pinch
            & grasp.success.opposing_flaps
            & self.grasp_previously_completed
        )
        result = carry_success(CarrySuccessInput(
            grasp_maintained=maintained,
            box_footprint_corners_belt=snapshot.box_footprint_corners_belt,
            belt_half_extents_xy=self.belt_half_extents_xy,
            box_bottom_height_m=snapshot.box_bottom_height_m,
            box_tilt_rad=snapshot.box_tilt_rad,
            # In the inspection scene, placed-box state has not been wired.
            # Conservatively reject overlap with any other box near the belt.
            overlaps_placed_box=snapshot.overlaps_other_belt_box,
        ))
        return CarryProbeResult(result, self.grasp_previously_completed)
