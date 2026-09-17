"""Pure tensor geometry shared by conveyor success predicates."""

from __future__ import annotations

import torch


def footprint_inside_rectangle(corners_xy: torch.Tensor, half_extents_xy: torch.Tensor) -> torch.Tensor:
    """Whether all four local-frame footprint corners lie inside a rectangle."""
    if corners_xy.ndim != 3 or corners_xy.shape[1:] != (4, 2):
        raise ValueError("corners_xy must have shape [num_envs, 4, 2].")
    n = len(corners_xy)
    if half_extents_xy.shape == (2,):
        half_extents_xy = half_extents_xy.expand(n, -1)
    if half_extents_xy.shape != (n, 2):
        raise ValueError("half_extents_xy must have shape [2] or [num_envs, 2].")
    if corners_xy.device != half_extents_xy.device:
        raise ValueError("Corners and rectangle extents must share one device.")
    if not corners_xy.is_floating_point() or not half_extents_xy.is_floating_point():
        raise TypeError("Rectangle geometry must be floating point.")
    if not bool(torch.isfinite(half_extents_xy).all()) or not bool((half_extents_xy > 0).all()):
        raise ValueError("Rectangle half extents must be finite and positive.")
    finite = torch.isfinite(corners_xy).all(dim=(-1, -2))
    return finite & (corners_xy.abs() <= half_extents_xy[:, None, :]).all(dim=(-1, -2))


def unsigned_axis_angle_error(angle_rad: torch.Tensor) -> torch.Tensor:
    """Smallest error of an unoriented axis, treating 0 and pi as parallel."""
    return 0.5 * torch.atan2(torch.sin(2.0 * angle_rad), torch.cos(2.0 * angle_rad)).abs()
