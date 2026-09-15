"""Shared physical-state guards, independent of collision/task success settings."""

import torch
from functools import wraps


def box_safety_checks(centers, poses, velocities, initial_z, origins, spec):
    height = centers[..., 2] - origins[:, None, 2] - initial_z
    non_finite = ~(torch.isfinite(centers).all(dim=(-1, -2))
                   & torch.isfinite(poses).all(dim=(-1, -2))
                   & torch.isfinite(velocities).all(dim=(-1, -2)))
    # Clamp before taking norms: the guard itself must not overflow.
    linear = velocities[..., :3].abs().clamp_max(1e6).norm(dim=-1)
    angular = velocities[..., 3:].abs().clamp_max(1e6).norm(dim=-1)
    return {
        "box_over_lift": (height >= spec.max_box_lift_height).any(-1),
        "box_non_finite": non_finite,
        "box_excessive_speed": ((linear > spec.max_box_linear_speed)
                                | (angular > spec.max_box_angular_speed)).any(-1),
    }


def failed(checks):
    return checks["box_over_lift"] | checks["box_non_finite"] | checks["box_excessive_speed"]


def guard_box_reward(func):
    """Unsafe terminal steps keep the failure cost, but earn no shaping reward."""
    @wraps(func)
    def guarded(env, *args, **kwargs):
        value = func(env, *args, **kwargs)
        manager = getattr(env, "command_manager", None)
        command = manager.get_term("workcell") if manager is not None else None
        mask = getattr(command, "box_safety_failure", None)
        return value if mask is None else torch.where(mask, torch.zeros_like(value), value)
    return guarded
