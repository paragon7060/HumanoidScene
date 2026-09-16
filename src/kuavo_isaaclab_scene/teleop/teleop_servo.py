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
    command_delay_s: float = 0.0


SMOOTH = ArmResponse("smooth", .045, 2.5, .08, 1.5, 12.)
RESPONSIVE = ArmResponse("responsive", .015, 10., .05, 2.5, 20.)
REAL = ArmResponse("real", .015, 10., .05, 2.5, 20., .200)


class ActionDelay:
    """Fixed control-tick input delay with explicit reset for safety transitions."""

    def __init__(self):
        self._queue = []
        self._steps = 0

    def reset(self):
        self._queue.clear()

    def step(self, actions, delay_s, dt):
        steps = max(0, round(delay_s / dt))
        if steps != self._steps:
            self._steps = steps
            self.reset()
        if steps == 0:
            return actions
        self._queue.append(actions.clone())
        if len(self._queue) <= steps:
            return torch.zeros_like(actions)
        return self._queue.pop(0)


def arm_response_profile(selection, mapping, input_mode):
    if selection == "auto":
        selection = "responsive" if mapping in {"absolute", "scaled"} and input_mode == "controllers" else "smooth"
    if selection not in {"smooth", "responsive", "real"}:
        raise ValueError(f"Unknown arm response: {selection}")
    return {"smooth": SMOOTH, "responsive": RESPONSIVE, "real": REAL}[selection]


def joint_servo_step(jac, error, joints, rest, limits, previous_velocity, command, dt, response):
    """Resolved-rate feedback anchored to measured joints, with bounded velocity.

    ``command`` remains in the interface for callers retaining their drive
    target, but must not integrate measured pose error onto that old target.
    """
    ident = torch.eye(6, device=jac.device, dtype=jac.dtype).expand(jac.shape[0], -1, -1)
    inverse = jac.transpose(1, 2) @ torch.linalg.solve(
        jac @ jac.transpose(1, 2) + response.damping ** 2 * ident, ident)
    velocity = (inverse @ (error * response.pose_gain).unsqueeze(-1)).squeeze(-1)
    nullspace = torch.eye(joints.shape[-1], device=jac.device, dtype=jac.dtype) - inverse @ jac
    velocity += .15 * (nullspace @ (rest - joints).unsqueeze(-1)).squeeze(-1)
    cap = response.max_velocity
    velocity *= cap / velocity.abs().amax(-1, keepdim=True).clamp_min(cap)
    velocity = torch.clamp(velocity, previous_velocity - response.max_acceleration * dt,
                           previous_velocity + response.max_acceleration * dt)
    velocity = torch.clamp(velocity, (limits[..., 0] - joints) / dt, (limits[..., 1] - joints) / dt)
    command = torch.clamp(joints + velocity * dt, joints - .10, joints + .10)
    return velocity, torch.clamp(command, limits[..., 0], limits[..., 1])
