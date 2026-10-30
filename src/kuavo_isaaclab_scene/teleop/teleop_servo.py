"""Bounded simulation-only arm servo profiles (no Isaac Sim dependency)."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ArmResponse:
    name: str
    smoothing_s: float
    pose_gain: float
    damping: float
    max_velocity: float
    max_acceleration: float


SMOOTH = ArmResponse("smooth", .045, 2.5, .08, 1.5, 12.)
RESPONSIVE = ArmResponse("responsive", .015, 10., .05, 2.5, 20.)


def arm_response_profile(selection, mapping, input_mode):
    if selection == "auto":
        selection = "responsive" if mapping in {"absolute", "scaled"} and input_mode == "controllers" else "smooth"
    if selection not in {"smooth", "responsive"}:
        raise ValueError(f"Unknown arm response: {selection}")
    return RESPONSIVE if selection == "responsive" else SMOOTH


def joint_servo_step(
    jac,
    error,
    joints,
    rest,
    limits,
    previous_velocity,
    command,
    dt,
    response,
    *,
    posture=None,
    posture_weight=.15,
    direct_indices=(),
    direct_gain=0.,
):
    """Damped pose-error feedback with velocity/acceleration and anti-windup bounds."""
    ident = torch.eye(6, device=jac.device, dtype=jac.dtype).expand(jac.shape[0], -1, -1)
    inverse = jac.transpose(1, 2) @ torch.linalg.solve(
        jac @ jac.transpose(1, 2) + response.damping ** 2 * ident, ident)
    velocity = (inverse @ (error * response.pose_gain).unsqueeze(-1)).squeeze(-1)
    nullspace = torch.eye(joints.shape[-1], device=jac.device, dtype=jac.dtype) - inverse @ jac
    posture = rest if posture is None else posture
    velocity += posture_weight * (
        nullspace @ (posture - joints).unsqueeze(-1)
    ).squeeze(-1)
    if direct_indices and direct_gain > 0.:
        direct_error = posture[:, direct_indices] - joints[:, direct_indices]
        velocity[:, direct_indices] += direct_gain * direct_error
    cap = response.max_velocity
    velocity *= cap / velocity.abs().amax(-1, keepdim=True).clamp_min(cap)
    velocity = torch.clamp(velocity, previous_velocity - response.max_acceleration * dt,
                           previous_velocity + response.max_acceleration * dt)
    velocity = torch.clamp(velocity, (limits[..., 0] - joints) / dt, (limits[..., 1] - joints) / dt)
    command = torch.clamp(command + velocity * dt, joints - .10, joints + .10)
    return velocity, torch.clamp(command, limits[..., 0], limits[..., 1])
