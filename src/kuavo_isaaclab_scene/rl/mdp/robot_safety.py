"""Numerical safety checks for robot articulation state."""

import torch


def robot_motion_unsafe(robot, max_joint_speed=100.0):
    """Catch articulation blow-ups before their finite values overflow rewards."""
    position = robot.data.joint_pos
    velocity = robot.data.joint_vel
    finite = torch.isfinite(position).all(-1) & torch.isfinite(velocity).all(-1)
    return ~finite | (velocity.abs().amax(-1) > max_joint_speed)
