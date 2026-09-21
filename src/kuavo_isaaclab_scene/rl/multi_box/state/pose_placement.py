"""Deployable, pose/proprio-only estimate of boxes left on the conveyor.

This is a policy selection hint, not the authoritative place-success test.
No simulator contact, force, object velocity, or success flag enters here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..geometry.belt import BELT_HALF_EXTENTS_XY, unsigned_axis_angle_error
from ..geometry.pose import quat_apply, relative_pose
from .perception import PerceptionFrame
from .placement import DeployablePlacementEstimateState
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schema import DeployableRobotState


BELT_HALF_THICKNESS_M = 0.015
BOX_ROOT_BOTTOM_OFFSET_FRACTION = 0.005


@dataclass(frozen=True)
class PosePlacementEstimateConfig:
    belt_height_tolerance_m: float = 0.02
    max_parallel_error_rad: float = math.radians(10.0)
    hold_seconds: float = 0.50
    # The remaining thresholds will be validated with VR/real telemetry.
    max_closed_fraction_for_release: float = 0.20
    max_height_drift_m: float = 0.005
    min_pose_confidence: float = 0.5

    def validate(self) -> None:
        values = vars(self)
        if not all(math.isfinite(value) and value > 0 for value in values.values()):
            raise ValueError("Pose placement thresholds must be finite and positive.")
        if self.max_parallel_error_rad >= math.pi / 2:
            raise ValueError("Long-axis tolerance must be less than 90 degrees.")
        if self.max_closed_fraction_for_release >= 1 or self.min_pose_confidence >= 1:
            raise ValueError("Release fraction and confidence threshold must be below one.")


@dataclass(frozen=True)
class PosePlacementEvidence:
    bottom_height_m: torch.Tensor
    footprint_inside: torch.Tensor
    belt_height_ok: torch.Tensor
    axis_parallel: torch.Tensor
    grippers_released: torch.Tensor
    no_overlap: torch.Tensor
    height_stable: torch.Tensor
    geometry_candidate: torch.Tensor


def _box_geometry_in_belt(perception: PerceptionFrame):
    """Four body-bottom corners in the belt's 6D frame, [N,12,4,3]."""
    boxes = perception.boxes
    box_in_belt = relative_pose(perception.conveyor_pose_world, boxes.pose_world)
    signs = boxes.pose_world.new_tensor((
        (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5), (0.5, 0.5)))
    xy = signs[None, None] * boxes.size_m[..., None, :2]
    z = (-BOX_ROOT_BOTTOM_OFFSET_FRACTION * boxes.size_m[..., 2:3]) \
        .unsqueeze(-2).expand(-1, -1, 4, -1)
    local_corners = torch.cat((xy, z), dim=-1)
    quat = box_in_belt[..., 3:].unsqueeze(-2).expand(-1, -1, 4, -1)
    corners = box_in_belt[..., :3].unsqueeze(-2) + quat_apply(quat, local_corners)
    return box_in_belt, corners


def pose_placement_evidence(
    perception: PerceptionFrame, robot: DeployableRobotState,
    config: PosePlacementEstimateConfig,
) -> tuple[PosePlacementEvidence, torch.Tensor]:
    """Geometry and release checks available from real perception/proprio."""
    n = perception.boxes.active.shape[0]
    perception.validate(n)
    robot.validate(n)
    config.validate()
    boxes = perception.boxes
    box_in_belt, corners = _box_geometry_in_belt(perception)
    half_extents = corners.new_tensor(BELT_HALF_EXTENTS_XY)
    finite_corners = torch.isfinite(corners).all(dim=(-1, -2))
    inside = finite_corners & (corners[..., :2].abs() <= half_extents).all(dim=(-1, -2))
    bottom_z = corners[..., 2].amin(dim=-1)
    height_ok = torch.isfinite(bottom_z) & (
        (bottom_z - BELT_HALF_THICKNESS_M).abs() <= config.belt_height_tolerance_m)
    long_axis = quat_apply(
        box_in_belt[..., 3:],
        torch.tensor((1.0, 0.0, 0.0), device=boxes.pose_world.device,
                     dtype=boxes.pose_world.dtype).expand(n, boxes.active.shape[1], -1))
    axis_error = unsigned_axis_angle_error(torch.atan2(long_axis[..., 1], long_axis[..., 0]))
    parallel = torch.isfinite(axis_error) & (axis_error <= config.max_parallel_error_rad)
    released = (robot.gripper_position <= config.max_closed_fraction_for_release).all(-1)

    # Reject any other confidently observed box occupying the same belt space.
    low, high = corners[..., :2].amin(-2), corners[..., :2].amax(-2)
    overlaps = ((low[:, :, None] < high[:, None, :])
                & (low[:, None, :] < high[:, :, None])).all(-1)
    count = boxes.active.shape[1]
    eye = torch.eye(count, device=boxes.active.device, dtype=torch.bool)[None]
    other_on_belt = boxes.active & (boxes.pose_confidence >= config.min_pose_confidence) \
        & height_ok
    blocked = (overlaps & other_on_belt[:, None, :] & ~eye).any(-1)
    known = boxes.active & (boxes.pose_confidence >= config.min_pose_confidence)
    geometry_candidate = known & inside & height_ok & parallel & ~blocked
    return (PosePlacementEvidence(
        bottom_height_m=bottom_z - BELT_HALF_THICKNESS_M,
        footprint_inside=inside,
        belt_height_ok=height_ok,
        axis_parallel=parallel,
        grippers_released=released[:, None].expand_as(known),
        no_overlap=~blocked,
        height_stable=torch.zeros_like(known),
        geometry_candidate=geometry_candidate,
    ), bottom_z)


class PosePlacementEstimateTracker:
    """Require an observed release and 0.5 s of stable on-belt pose."""

    def __init__(self, num_envs: int, device: str | torch.device, config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.config = config or PosePlacementEstimateConfig()
        self.config.validate()
        self.hold_time_s = torch.zeros(num_envs, 12, device=device)
        self.anchor_bottom_z = torch.zeros_like(self.hold_time_s)
        self.placed = torch.zeros(num_envs, 12, dtype=torch.bool, device=device)
        self.release_latched = torch.zeros_like(self.placed)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.hold_time_s[ids] = 0.0
        self.anchor_bottom_z[ids] = 0.0
        self.placed[ids] = False
        self.release_latched[ids] = False

    def update(self, perception: PerceptionFrame, robot: DeployableRobotState,
               dt: float) -> tuple[DeployablePlacementEstimateState, PosePlacementEvidence]:
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")
        evidence, bottom_z = pose_placement_evidence(perception, robot, self.config)
        if evidence.geometry_candidate.shape != self.placed.shape:
            raise ValueError("Pose placement batch size differs from tracker.")
        was_tracking = self.hold_time_s > 0
        stable = ~was_tracking | (
            (bottom_z - self.anchor_bottom_z).abs() <= self.config.max_height_drift_m)
        geometry_stable = evidence.geometry_candidate & stable
        # A placed box remains placed while the grippers close on the next
        # target. Its release evidence is revoked if its pose is disturbed.
        self.release_latched = geometry_stable & (
            self.release_latched | evidence.grippers_released)
        continuous = geometry_stable & self.release_latched
        self.anchor_bottom_z = torch.where(
            evidence.geometry_candidate & ~stable | (evidence.geometry_candidate & ~was_tracking),
            bottom_z, self.anchor_bottom_z)
        self.hold_time_s = torch.where(
            continuous, self.hold_time_s + dt, torch.zeros_like(self.hold_time_s))
        self.placed = continuous & (self.hold_time_s >= self.config.hold_seconds)
        active = perception.boxes.active
        placement = DeployablePlacementEstimateState(
            active=active.clone(), placed=self.placed.clone(),
            selectable=active & ~self.placed,
            hold_time_s=self.hold_time_s.clone(),
        )
        evidence = PosePlacementEvidence(
            **{**vars(evidence), "height_stable": stable})
        return placement, evidence
