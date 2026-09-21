"""Read-only, physically supported place-success probe for Quest inspection."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..geometry.belt import BELT_HALF_EXTENTS_XY
from ..success import PlaceSuccessInput, PlaceSuccessResult, PlaceSuccessTracker
from .carry_probe import CarryProbeResult
from .grasp_probe import GraspProbeResult


MIN_BELT_BODY_FORCE_N = 0.2


@dataclass(frozen=True)
class PlaceProbeResult:
    result: PlaceSuccessResult
    carry_previously_completed: bool
    measured_belt_force_n: torch.Tensor

    def report(self) -> str:
        value = self.result
        return (
            f"PLACE PROBE | carry-seen={int(self.carry_previously_completed)} "
            f"belt-F={float(self.measured_belt_force_n[0]):.2f}N "
            f"support={int(value.supported[0])} "
            f"released-clear={int(value.released_and_clear[0])} "
            f"belt={int(value.footprint_inside_belt[0])} "
            f"free={int(value.free_space[0])} "
            f"parallel={int(value.axis_parallel[0])} "
            f"still={int(value.motion_stable[0])} "
            f"hold={float(value.hold_time_s[0]):.2f}s "
            f"success={int(value.success[0])}"
        )


class QuestPlaceProbe:
    """Requires a prior carry success without driving control or reward."""

    def __init__(self, device: str | torch.device, num_envs: int = 1):
        self.device = torch.device(device)
        self.belt_half_extents_xy = torch.tensor(BELT_HALF_EXTENTS_XY, device=self.device)
        self.tracker = PlaceSuccessTracker(num_envs, device)
        self.carry_previously_completed = False
        self.target_logical_id = -1

    def reset(self) -> None:
        self.tracker.reset()
        self.carry_previously_completed = False
        self.target_logical_id = -1

    def update(self, snapshot, grasp: GraspProbeResult,
               carry: CarryProbeResult, dt: float) -> PlaceProbeResult:
        if snapshot.target_logical_id != self.target_logical_id:
            self.reset()
            self.target_logical_id = snapshot.target_logical_id
        self.carry_previously_completed |= bool(carry.result.success[0].item())
        supported = (
            snapshot.belt_sensor_available
            & torch.isfinite(snapshot.belt_body_force_n)
            & (snapshot.belt_body_force_n >= MIN_BELT_BODY_FORCE_N)
            & self.carry_previously_completed
        )
        raw = snapshot.raw_by_phase["place"]
        result = self.tracker.update(PlaceSuccessInput(
            belt_support=supported,
            gripper_grasping=grasp.pinch.hand_pinching,
            gripper_box_distance_m=snapshot.gripper_box_distance_m,
            box_footprint_corners_belt=snapshot.box_footprint_corners_belt,
            belt_half_extents_xy=self.belt_half_extents_xy,
            overlaps_placed_box=snapshot.overlaps_other_belt_box,
            long_axis_yaw_error_rad=raw.long_axis_error_rad,
            linear_speed_mps=raw.linear_speed_mps,
            angular_speed_radps=raw.angular_speed_radps,
        ), dt)
        return PlaceProbeResult(result, self.carry_previously_completed,
                                snapshot.belt_body_force_n)
