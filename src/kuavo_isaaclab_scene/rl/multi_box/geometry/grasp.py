"""Deployable flap geometry and hand-to-flap pairing for the v2 grasp task."""

from __future__ import annotations

import torch

from ....workcell.rack_box_layout import BOX_FLAP_LENGTH_M
from ..spec import BOX_TYPES
from .pose import quat_apply, relative_pose


GRASP_APPROACH_REWARD_SCALE_M = 1.0 / 12.0


def closest_flap_surface(
    points: torch.Tensor,
    centers: torch.Tensor,
    halves: torch.Tensor,
    normal_axes: torch.Tensor,
) -> torch.Tensor:
    """Closest point on either broad face of each thin rectangular flap."""
    delta = points - centers
    nearest = delta.clamp(-halves, halves)
    axis = normal_axes[..., None]
    side = torch.where(delta.gather(-1, axis) >= 0, 1.0, -1.0)
    return centers + nearest.scatter(-1, axis, side * halves.gather(-1, axis))


def opposing_flap_reach_assignment(
    candidate_distance_m: torch.Tensor, approach_scale_m: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the distinct-flap pairing with the best weaker-hand reach score."""
    if candidate_distance_m.ndim != 3 or candidate_distance_m.shape[1:] != (2, 2):
        raise ValueError("Candidate distances need [env, hand, flap] shape")
    if approach_scale_m <= 0:
        raise ValueError("Approach scale must be positive")
    scores = torch.exp(-candidate_distance_m.clamp_min(0) / approach_scale_m)

    def paired_score(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.25 * (left + right) + 0.5 * torch.minimum(left, right)

    direct = paired_score(scores[:, 0, 0], scores[:, 1, 1])
    swapped = paired_score(scores[:, 0, 1], scores[:, 1, 0])
    assignment = torch.where(
        (direct >= swapped)[:, None],
        torch.tensor((0, 1), device=scores.device),
        torch.tensor((1, 0), device=scores.device),
    )
    return torch.maximum(direct, swapped), assignment


def nominal_flap_geometry(
    box_size_m: torch.Tensor, box_type_id: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right/left neutral flap centers, half extents and normal axes in box frame.

    The stock box wrappers share a 1.01-wide template.  The two flap hinges
    sit at +/- half a template unit on X, with bottoms 0.005 body heights
    above the box top.  This uses known asset geometry, not simulator truth.
    """
    if box_size_m.ndim != 2 or box_size_m.shape[-1] != 3 \
            or box_type_id.shape != (len(box_size_m),):
        raise ValueError("Expected size [env,3] and type [env]")
    flap_lengths = torch.as_tensor(
        [BOX_FLAP_LENGTH_M[name] for name in BOX_TYPES],
        dtype=box_size_m.dtype, device=box_size_m.device,
    )[box_type_id.clamp(0, len(BOX_TYPES) - 1)]
    scale_x = box_size_m[:, 0] / 1.01
    scale_y = box_size_m[:, 1] / 1.01
    zero = torch.zeros_like(scale_x)
    center_z = 1.005 * box_size_m[:, 2] + 0.5 * flap_lengths
    centers = torch.stack((
        torch.stack((0.5 * scale_x, zero, center_z), dim=-1),
        torch.stack((-0.5 * scale_x, zero, center_z), dim=-1),
    ), dim=1)
    half = torch.stack((0.005 * scale_x, 0.49 * scale_y, 0.5 * flap_lengths), dim=-1)
    halves = half[:, None].expand(-1, 2, -1)
    axes = torch.zeros((len(box_size_m), 2), dtype=torch.long, device=box_size_m.device)
    return centers, halves, axes


def estimated_flap_center_poses(
    box_pose_world: torch.Tensor,
    box_size_m: torch.Tensor,
    box_type_id: torch.Tensor,
    tcp_pose_world: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate fixed right/left flap centers and hand-to-surface distances."""
    if box_pose_world.ndim != 2 or box_pose_world.shape[1] != 7 \
            or box_size_m.shape != (len(box_pose_world), 3) \
            or box_type_id.shape != (len(box_pose_world),) \
            or tcp_pose_world.shape != (len(box_pose_world), 2, 7):
        raise ValueError("Expected box [env,7], size [env,3], type [env], TCP [env,2,7]")
    centers, halves, axes = nominal_flap_geometry(box_size_m, box_type_id)

    tcp_in_box = relative_pose(box_pose_world[:, None], tcp_pose_world)[..., :3]
    nearest = closest_flap_surface(
        tcp_in_box[:, :, None], centers[:, None], halves[:, None], axes[:, None])
    distance = (tcp_in_box[:, :, None] - nearest).norm(dim=-1)
    box_quat = box_pose_world[:, None, None, 3:].expand(-1, 2, 2, -1)
    center_world = box_pose_world[:, None, None, :3] + quat_apply(
        box_quat, centers[:, None].expand(-1, 2, -1, -1))
    center_pose_world = torch.cat((center_world, box_quat), dim=-1)
    return center_pose_world, distance
