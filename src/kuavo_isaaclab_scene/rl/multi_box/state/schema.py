"""Tensor schemas separating privileged truth from deployable estimates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..hierarchy import SkillControlState
from ..spec import MAX_BOXES
from .placement import PlacementSetState


def _shape(value: torch.Tensor, expected: tuple[int, ...], name: str, *, dtype=None) -> None:
    if value.shape != expected:
        raise ValueError(f"{name} must have shape {list(expected)}.")
    if dtype is not None and value.dtype != dtype:
        raise TypeError(f"{name} must use {dtype}.")


def _floating(value: torch.Tensor, expected: tuple[int, ...], name: str) -> None:
    _shape(value, expected, name)
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point.")


@dataclass(frozen=True)
class DeployableBoxState:
    """Perception outputs available in both simulation and real deployment."""

    active: torch.Tensor
    box_type_id: torch.Tensor
    rack_region_id: torch.Tensor
    size_m: torch.Tensor
    pose_world: torch.Tensor
    pose_confidence: torch.Tensor

    def validate(self, num_envs: int) -> None:
        prefix = (num_envs, MAX_BOXES)
        _shape(self.active, prefix, "active", dtype=torch.bool)
        _shape(self.box_type_id, prefix, "box_type_id", dtype=torch.long)
        _shape(self.rack_region_id, prefix, "rack_region_id", dtype=torch.long)
        _floating(self.size_m, (*prefix, 3), "size_m")
        # Pose is xyz + unit quaternion wxyz. Observation code may later encode
        # orientation as rotation-6D without losing the state reference frame.
        _floating(self.pose_world, (*prefix, 7), "pose_world")
        _floating(self.pose_confidence, prefix, "pose_confidence")


@dataclass(frozen=True)
class DeployableRobotState:
    """Robot proprioception expected from the real controller interface."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    base_pose_world: torch.Tensor
    base_twist_world: torch.Tensor
    tcp_pose_world: torch.Tensor
    gripper_position: torch.Tensor
    gripper_command: torch.Tensor

    def validate(self, num_envs: int) -> None:
        if self.joint_pos.ndim != 2 or self.joint_pos.shape[0] != num_envs:
            raise ValueError("joint_pos must have shape [num_envs, num_joints].")
        _floating(self.joint_pos, self.joint_pos.shape, "joint_pos")
        _floating(self.joint_vel, self.joint_pos.shape, "joint_vel")
        _floating(self.base_pose_world, (num_envs, 7), "base_pose_world")
        _floating(self.base_twist_world, (num_envs, 6), "base_twist_world")
        _floating(self.tcp_pose_world, (num_envs, 2, 7), "tcp_pose_world")
        _floating(self.gripper_position, (num_envs, 2), "gripper_position")
        _floating(self.gripper_command, (num_envs, 2), "gripper_command")


@dataclass(frozen=True)
class DeployableTaskState:
    """State that can be reconstructed from real sensors and perception."""

    boxes: DeployableBoxState
    robot: DeployableRobotState
    rack_pose_world: torch.Tensor
    conveyor_pose_world: torch.Tensor
    placement: PlacementSetState
    control: SkillControlState
    transition_confidence: torch.Tensor

    def validate(self, num_envs: int) -> None:
        self.boxes.validate(num_envs)
        self.robot.validate(num_envs)
        _floating(self.rack_pose_world, (num_envs, 7), "rack_pose_world")
        _floating(self.conveyor_pose_world, (num_envs, 7), "conveyor_pose_world")
        _floating(self.transition_confidence, (num_envs, 3), "transition_confidence")
        masks = (self.placement.active, self.placement.placed, self.placement.selectable)
        if any(value.shape != (num_envs, MAX_BOXES) for value in masks):
            raise ValueError("Placement masks must have shape [num_envs, 12].")
        if self.control.target_box.shape != (num_envs,) \
                or self.control.current_skill.shape != (num_envs,):
            raise ValueError("Control target and skill must have shape [num_envs].")


@dataclass(frozen=True)
class PrivilegedTaskState:
    """Simulator truth used by success, diagnostics, and later reward design.

    None of these fields becomes an actor observation merely by being present
    here.  In particular box velocities and contact forces remain privileged.
    """

    box_pose_world: torch.Tensor
    box_pose_rack: torch.Tensor
    box_pose_conveyor: torch.Tensor
    box_linear_velocity: torch.Tensor
    box_angular_velocity: torch.Tensor
    hand_pinching: torch.Tensor
    hand_flap_index: torch.Tensor
    hand_box_pose_stable: torch.Tensor
    finger_contact_force_n: torch.Tensor
    rack_clearance_m: torch.Tensor
    belt_support: torch.Tensor
    footprint_corners_belt: torch.Tensor
    overlaps_placed_box: torch.Tensor
    valid_place: torch.Tensor
    rack_contact_force_n: torch.Tensor
    self_contact_force_n: torch.Tensor
    obstacle_contact_force_n: torch.Tensor
    base_distance_m: torch.Tensor

    def validate(self, num_envs: int) -> None:
        prefix = (num_envs, MAX_BOXES)
        for name in ("box_pose_world", "box_pose_rack", "box_pose_conveyor"):
            _floating(getattr(self, name), (*prefix, 7), name)
        for name in ("box_linear_velocity", "box_angular_velocity"):
            _floating(getattr(self, name), (*prefix, 3), name)
        _shape(self.hand_pinching, (*prefix, 2), "hand_pinching", dtype=torch.bool)
        _shape(self.hand_flap_index, (*prefix, 2), "hand_flap_index", dtype=torch.long)
        _shape(self.hand_box_pose_stable, (*prefix, 2), "hand_box_pose_stable", dtype=torch.bool)
        _floating(self.finger_contact_force_n, (*prefix, 2, 2), "finger_contact_force_n")
        _floating(self.rack_clearance_m, prefix, "rack_clearance_m")
        _shape(self.belt_support, prefix, "belt_support", dtype=torch.bool)
        _floating(self.footprint_corners_belt, (*prefix, 4, 2), "footprint_corners_belt")
        _shape(self.overlaps_placed_box, prefix, "overlaps_placed_box", dtype=torch.bool)
        _shape(self.valid_place, prefix, "valid_place", dtype=torch.bool)
        for name in (
            "rack_contact_force_n",
            "self_contact_force_n",
            "obstacle_contact_force_n",
            "base_distance_m",
        ):
            _floating(getattr(self, name), (num_envs,), name)


@dataclass(frozen=True)
class MultiBoxState:
    deployable: DeployableTaskState
    privileged: PrivilegedTaskState

    def validate(self, num_envs: int) -> None:
        self.deployable.validate(num_envs)
        self.privileged.validate(num_envs)

        tensors = []
        for group in (vars(self.deployable.boxes), vars(self.deployable.robot),
                      vars(self.privileged)):
            tensors.extend(value for value in group.values() if isinstance(value, torch.Tensor))
        tensors.extend((self.deployable.rack_pose_world,
                        self.deployable.conveyor_pose_world,
                        self.deployable.transition_confidence))
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("All multi-box state tensors must share one device.")
