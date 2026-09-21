"""Conservative fingertip-pad clearance from oriented box collision bounds."""

from __future__ import annotations

import math

import torch

from .pose import quat_apply, quat_conjugate, normalize_quaternion


def pad_to_boxes_clearance_m(
    pad_pose_w: torch.Tensor,
    pad_size_m: tuple[float, float, float],
    box_poses_w: torch.Tensor,
    box_centers: torch.Tensor,
    box_halves: torch.Tensor,
) -> torch.Tensor:
    """Lower bound on each pad's distance to a set of oriented box parts.

    A 3x3x3 point grid samples the pad volume.  Subtracting the grid covering
    radius makes the result conservative: it may require extra separation but
    cannot report 2 cm clear while an unsampled point is closer than 2 cm.
    """
    if pad_pose_w.ndim != 2 or pad_pose_w.shape[1] != 7:
        raise ValueError("pad_pose_w must have shape [num_pads, 7].")
    if box_poses_w.ndim != 2 or box_poses_w.shape[1] != 7:
        raise ValueError("box_poses_w must have shape [num_box_parts, 7].")
    parts = box_poses_w.shape[0]
    if parts < 1 or box_centers.shape != (parts, 3) or box_halves.shape != (parts, 3):
        raise ValueError("Box centers and half-sizes must have shape [num_box_parts, 3].")
    if len(pad_size_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in pad_size_m):
        raise ValueError("Pad size must contain three positive finite values.")
    tensors = (pad_pose_w, box_poses_w, box_centers, box_halves)
    if len({value.device for value in tensors}) != 1:
        raise ValueError("Pad and box geometry must share one device.")
    half = torch.as_tensor(pad_size_m, device=pad_pose_w.device, dtype=pad_pose_w.dtype) / 2
    grid = torch.cartesian_prod(*(pad_pose_w.new_tensor((-1., 0., 1.)) for _ in range(3))) * half
    pad_q = normalize_quaternion(pad_pose_w[:, None, 3:])
    samples = pad_pose_w[:, None, :3] + quat_apply(pad_q, grid[None])
    box_q_inv = quat_conjugate(normalize_quaternion(box_poses_w[None, :, None, 3:]))
    local = quat_apply(
        box_q_inv,
        samples[:, None] - box_poses_w[None, :, None, :3],
    )
    outside = (local - box_centers[None, :, None]).abs() - box_halves[None, :, None]
    sampled_distance = outside.clamp_min(0).norm(dim=-1).amin(dim=(-1, -2))
    covering_radius = half.norm() / 2.0
    return (sampled_distance - covering_radius).clamp_min(0)
