"""Real-controller-compatible robot proprioception for the v2 actor."""

from __future__ import annotations

import torch

from ..geometry.pose import quat_apply
from .schema import ACTUATED_BODY_JOINTS, DeployableRobotState


def closure_fraction(
    measured_joint_pos: torch.Tensor,
    open_joint_pos: torch.Tensor,
    closed_joint_pos: torch.Tensor,
) -> torch.Tensor:
    """Project both measured driver angles onto open=0 / closed=1."""
    if measured_joint_pos.ndim != 2 or measured_joint_pos.shape[1] != 2:
        raise ValueError("Measured gripper driver joints must be [num_envs, 2].")
    if open_joint_pos.shape != (2,) or closed_joint_pos.shape != (2,):
        raise ValueError("Open and closed gripper commands must each have two joints.")
    if len({value.device for value in (measured_joint_pos, open_joint_pos, closed_joint_pos)}) != 1:
        raise ValueError("Gripper driver angles and commands must share one device.")
    direction = closed_joint_pos - open_joint_pos
    denominator = direction.square().sum()
    if bool((denominator < 1e-8).item()):
        raise ValueError("Open and closed gripper driver angles must differ.")
    return (((measured_joint_pos - open_joint_pos) * direction).sum(-1)
            / denominator).clamp(0.0, 1.0)


def kinematic_base_twist_world(base_quat_world: torch.Tensor,
                               planar_velocity_local: torch.Tensor) -> torch.Tensor:
    """World-frame twist of the simulator's ideal kinematic planar drive."""
    if base_quat_world.ndim != 2 or base_quat_world.shape[1] != 4:
        raise ValueError("Base quaternion must be [num_envs, 4].")
    n = len(base_quat_world)
    if planar_velocity_local.shape != (n, 3):
        raise ValueError("Planar drive velocity must be [num_envs, 3].")
    zero = torch.zeros(n, 1, device=planar_velocity_local.device,
                       dtype=planar_velocity_local.dtype)
    linear_local = torch.cat((planar_velocity_local[:, :2], zero), dim=-1)
    angular_local = torch.cat((zero, zero, planar_velocity_local[:, 2:3]), dim=-1)
    return torch.cat((quat_apply(base_quat_world, linear_local),
                      quat_apply(base_quat_world, angular_local)), dim=-1)


def robot_state_from_sensors(
    *, joint_pos: torch.Tensor, joint_vel: torch.Tensor,
    base_pose_world: torch.Tensor, base_twist_world: torch.Tensor,
    tcp_pose_world: torch.Tensor, gripper_position: torch.Tensor,
    gripper_command: torch.Tensor,
) -> DeployableRobotState:
    """Common robot-state interface for Isaac and eventual real telemetry.

    ``joint_pos``/``joint_vel`` follow ACTUATED_BODY_JOINTS. Gripper position
    and command are left/right closure fractions, 0=open and 1=closed.
    """
    if joint_pos.ndim != 2 or joint_pos.shape[1] != len(ACTUATED_BODY_JOINTS):
        raise ValueError("Body joints must follow the fixed 20-joint policy order.")
    result = DeployableRobotState(
        joint_pos=joint_pos.clone(), joint_vel=joint_vel.clone(),
        base_pose_world=base_pose_world.clone(),
        base_twist_world=base_twist_world.clone(),
        tcp_pose_world=tcp_pose_world.clone(),
        gripper_position=gripper_position.clone(),
        gripper_command=gripper_command.clone(),
    )
    result.validate(len(joint_pos))
    devices = {value.device for value in vars(result).values()}
    if len(devices) != 1:
        raise ValueError("Robot proprioception tensors must share one device.")
    return result
