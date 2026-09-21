"""Clearance of a box bottom from the sloped shelf beneath its rack region."""

from __future__ import annotations

import math

import torch

from ...scenes.layout import RACK_SLOPE_RAD
from ....workcell.rack_box_layout import RACK_RAMP_BACK_DEPTH_RAW
from ....workcell.workcell_layout import RACK_RAW_TIER_RANGES
from .pose import quat_apply, quat_conjugate, normalize_quaternion


def box_shelf_clearance_m(
    box_pose_w: torch.Tensor,
    rack_pose_w: torch.Tensor,
    box_dimensions_m: tuple[float, float, float],
    *,
    shelf: int,
    rack_scale: tuple[float, float, float],
) -> torch.Tensor:
    """Smallest underside-corner gap to the corresponding inclined shelf.

    The box wrapper's root is 0.5% of its height above the bottom, matching
    the v2 spawn planner.  This is a geometry check, not a contact-force check.
    """
    if box_pose_w.ndim != 2 or box_pose_w.shape[1] != 7 or rack_pose_w.shape != box_pose_w.shape:
        raise ValueError("Box and rack poses must have shape [num_envs, 7].")
    if shelf not in (2, 3):
        raise ValueError("V2 rack clearance is defined for shelf 2 or 3.")
    if len(rack_scale) != 3 or any(not math.isfinite(v) or v <= 0 for v in rack_scale):
        raise ValueError("Rack scale must contain three positive finite values.")
    if len(box_dimensions_m) != 3 or any(not math.isfinite(v) or v <= 0 for v in box_dimensions_m):
        raise ValueError("Box dimensions must contain three positive finite values.")
    half_x, half_y = box_dimensions_m[0] / 2, box_dimensions_m[1] / 2
    bottom_offset = -0.005 * box_dimensions_m[2]
    corners = box_pose_w.new_tensor((
        (-half_x, -half_y, bottom_offset),
        (-half_x, +half_y, bottom_offset),
        (+half_x, -half_y, bottom_offset),
        (+half_x, +half_y, bottom_offset),
    ))[None].expand(box_pose_w.shape[0], -1, -1)
    box_q = normalize_quaternion(box_pose_w[:, None, 3:])
    world = box_pose_w[:, None, :3] + quat_apply(box_q, corners)
    rack_q_inv = quat_conjugate(normalize_quaternion(rack_pose_w[:, None, 3:]))
    local = quat_apply(rack_q_inv, world - rack_pose_w[:, None, :3])
    surface_raw = RACK_RAW_TIER_RANGES[shelf - 1][1] - math.tan(RACK_SLOPE_RAD) * (
        RACK_RAMP_BACK_DEPTH_RAW + local[..., 1] / rack_scale[1])
    return (local[..., 2] - surface_raw * rack_scale[2]).amin(dim=-1)
