"""Privileged carry-success predicate for a selected box over the conveyor."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..geometry import footprint_inside_rectangle


@dataclass(frozen=True)
class CarrySuccessConfig:
    """Approved pre-place height interval above the belt surface."""

    min_pre_place_height_m: float = 0.05
    max_pre_place_height_m: float = 0.15
    max_box_tilt_rad: float = math.radians(20.0)

    def validate(self) -> None:
        values = (
            self.min_pre_place_height_m,
            self.max_pre_place_height_m,
            self.max_box_tilt_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Carry pre-place heights must be finite.")
        if not 0 <= self.min_pre_place_height_m < self.max_pre_place_height_m:
            raise ValueError("Carry pre-place height range must be ordered and nonnegative.")
        if not 0 < self.max_box_tilt_rad < math.pi / 2:
            raise ValueError("Carry box tilt limit must be between zero and 90 degrees.")


@dataclass(frozen=True)
class CarrySuccessInput:
    """Measurements expressed in the randomized conveyor's local frame.

    ``box_footprint_corners_belt`` contains the four bottom footprint corners.
    ``belt_half_extents_xy`` describes the usable belt rectangle, not its frame
    or rails.  This keeps the predicate valid when the conveyor is translated
    or yaw-rotated at reset.
    """

    grasp_maintained: torch.Tensor
    box_footprint_corners_belt: torch.Tensor
    belt_half_extents_xy: torch.Tensor
    box_bottom_height_m: torch.Tensor
    box_tilt_rad: torch.Tensor
    overlaps_placed_box: torch.Tensor

    def validate(self) -> None:
        n = len(self.grasp_maintained)
        if self.grasp_maintained.shape != (n,) or self.grasp_maintained.dtype != torch.bool:
            raise ValueError("grasp_maintained must be boolean [num_envs].")
        if self.box_footprint_corners_belt.shape != (n, 4, 2):
            raise ValueError("box footprint corners must have shape [num_envs, 4, 2].")
        if self.belt_half_extents_xy.shape not in ((2,), (n, 2)):
            raise ValueError("belt_half_extents_xy must have shape [2] or [num_envs, 2].")
        if self.box_bottom_height_m.shape != (n,) or not self.box_bottom_height_m.is_floating_point():
            raise ValueError("box_bottom_height_m must be floating point [num_envs].")
        if self.box_tilt_rad.shape != (n,) or not self.box_tilt_rad.is_floating_point():
            raise ValueError("box_tilt_rad must be floating point [num_envs].")
        if self.overlaps_placed_box.shape != (n,) or self.overlaps_placed_box.dtype != torch.bool:
            raise ValueError("overlaps_placed_box must be boolean [num_envs].")
        if not self.box_footprint_corners_belt.is_floating_point():
            raise TypeError("box footprint corners must be floating point.")
        if not self.belt_half_extents_xy.is_floating_point():
            raise TypeError("belt half extents must be floating point.")
        devices = {
            self.grasp_maintained.device,
            self.box_footprint_corners_belt.device,
            self.belt_half_extents_xy.device,
            self.box_bottom_height_m.device,
            self.box_tilt_rad.device,
            self.overlaps_placed_box.device,
        }
        if len(devices) != 1:
            raise ValueError("All carry-success measurements must share one device.")


@dataclass(frozen=True)
class CarrySuccessResult:
    grasp_maintained: torch.Tensor
    box_tilt_ok: torch.Tensor
    footprint_inside_belt: torch.Tensor
    free_space: torch.Tensor
    pre_place_height: torch.Tensor
    success: torch.Tensor


def carry_success(measurements: CarrySuccessInput, config=None) -> CarrySuccessResult:
    """Return carry success without assigning reward or changing task phase."""
    config = config or CarrySuccessConfig()
    config.validate()
    measurements.validate()

    inside = footprint_inside_rectangle(
        measurements.box_footprint_corners_belt,
        measurements.belt_half_extents_xy,
    )
    free = ~measurements.overlaps_placed_box
    height = measurements.box_bottom_height_m
    pre_place = torch.isfinite(height) & (height >= config.min_pre_place_height_m) \
        & (height <= config.max_pre_place_height_m)
    tilt_ok = torch.isfinite(measurements.box_tilt_rad) \
        & (measurements.box_tilt_rad <= config.max_box_tilt_rad + 1e-6)
    success = measurements.grasp_maintained & tilt_ok & inside & free & pre_place
    return CarrySuccessResult(
        measurements.grasp_maintained,
        tilt_ok,
        inside,
        free,
        pre_place,
        success,
    )
