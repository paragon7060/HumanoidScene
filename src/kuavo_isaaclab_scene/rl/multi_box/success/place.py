"""Privileged, reward-independent place-success predicate."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..geometry import footprint_inside_rectangle, unsigned_axis_angle_error


@dataclass(frozen=True)
class PlaceSuccessConfig:
    gripper_clearance_m: float = 0.02
    max_parallel_error_rad: float = math.radians(10.0)
    max_linear_speed_mps: float = 0.05
    max_angular_speed_radps: float = 0.20
    hold_seconds: float = 0.50

    def validate(self) -> None:
        values = (
            self.gripper_clearance_m,
            self.max_parallel_error_rad,
            self.max_linear_speed_mps,
            self.max_angular_speed_radps,
            self.hold_seconds,
        )
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("Place thresholds must be finite and positive.")
        if self.max_parallel_error_rad >= math.pi / 2:
            raise ValueError("Parallel-axis tolerance must be less than 90 degrees.")


@dataclass(frozen=True)
class PlaceSuccessInput:
    """Privileged measurements for one selected box per environment.

    The gripper distance is the nearest finger-pad distance to any collision
    surface on the selected box.  ``long_axis_yaw_error_rad`` is measured in
    the randomized conveyor frame.  Placed-box overlap must exclude the box
    currently being evaluated.
    """

    belt_support: torch.Tensor
    gripper_grasping: torch.Tensor
    gripper_box_distance_m: torch.Tensor
    box_footprint_corners_belt: torch.Tensor
    belt_half_extents_xy: torch.Tensor
    overlaps_placed_box: torch.Tensor
    long_axis_yaw_error_rad: torch.Tensor
    linear_speed_mps: torch.Tensor
    angular_speed_radps: torch.Tensor

    def validate(self) -> None:
        n = len(self.belt_support)
        bool_vectors = {
            "belt_support": self.belt_support,
            "overlaps_placed_box": self.overlaps_placed_box,
        }
        for name, value in bool_vectors.items():
            if value.shape != (n,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean [num_envs].")
        if self.gripper_grasping.shape != (n, 2) or self.gripper_grasping.dtype != torch.bool:
            raise ValueError("gripper_grasping must be boolean [num_envs, 2].")
        if self.gripper_box_distance_m.shape != (n, 2):
            raise ValueError("gripper_box_distance_m must have shape [num_envs, 2].")
        if self.box_footprint_corners_belt.shape != (n, 4, 2):
            raise ValueError("box footprint corners must have shape [num_envs, 4, 2].")
        if self.belt_half_extents_xy.shape not in ((2,), (n, 2)):
            raise ValueError("belt_half_extents_xy must have shape [2] or [num_envs, 2].")
        for name in (
            "gripper_box_distance_m",
            "box_footprint_corners_belt",
            "belt_half_extents_xy",
            "long_axis_yaw_error_rad",
            "linear_speed_mps",
            "angular_speed_radps",
        ):
            value = getattr(self, name)
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point.")
            if name not in ("gripper_box_distance_m", "box_footprint_corners_belt", "belt_half_extents_xy") \
                    and value.shape != (n,):
                raise ValueError(f"{name} must have shape [num_envs].")
        devices = {value.device for value in vars(self).values()}
        if len(devices) != 1:
            raise ValueError("All place-success measurements must share one device.")


@dataclass(frozen=True)
class PlaceSuccessResult:
    supported: torch.Tensor
    released_and_clear: torch.Tensor
    footprint_inside_belt: torch.Tensor
    free_space: torch.Tensor
    axis_parallel: torch.Tensor
    motion_stable: torch.Tensor
    instantaneous: torch.Tensor
    hold_time_s: torch.Tensor
    success: torch.Tensor


class PlaceSuccessTracker:
    def __init__(self, num_envs: int, device: str | torch.device, config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.config = config or PlaceSuccessConfig()
        self.config.validate()
        self.hold_time_s = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.hold_time_s.zero_()
        else:
            self.hold_time_s[env_ids] = 0.0

    def update(self, measurements: PlaceSuccessInput, dt: float) -> PlaceSuccessResult:
        measurements.validate()
        if len(measurements.belt_support) != len(self.hold_time_s):
            raise ValueError("Place-success batch size does not match the tracker.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")

        finite_distance = torch.isfinite(measurements.gripper_box_distance_m).all(dim=-1)
        released = (~measurements.gripper_grasping.any(dim=-1)) & finite_distance \
            & (measurements.gripper_box_distance_m >= self.config.gripper_clearance_m).all(dim=-1)
        inside = footprint_inside_rectangle(
            measurements.box_footprint_corners_belt,
            measurements.belt_half_extents_xy,
        )
        free = ~measurements.overlaps_placed_box
        angle_error = unsigned_axis_angle_error(measurements.long_axis_yaw_error_rad)
        parallel = torch.isfinite(angle_error) \
            & (angle_error <= self.config.max_parallel_error_rad + 1e-6)
        speeds_finite = torch.isfinite(measurements.linear_speed_mps) \
            & torch.isfinite(measurements.angular_speed_radps)
        stable = speeds_finite \
            & (measurements.linear_speed_mps <= self.config.max_linear_speed_mps) \
            & (measurements.angular_speed_radps <= self.config.max_angular_speed_radps)
        instantaneous = measurements.belt_support & released & inside & free & parallel & stable
        self.hold_time_s = torch.where(
            instantaneous, self.hold_time_s + dt, torch.zeros_like(self.hold_time_s))
        success = self.hold_time_s >= self.config.hold_seconds
        return PlaceSuccessResult(
            measurements.belt_support,
            released,
            inside,
            free,
            parallel,
            stable,
            instantaneous,
            self.hold_time_s.clone(),
            success,
        )
