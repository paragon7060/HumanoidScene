"""Batched frame geometry; never mix world coordinates across cloned environments."""

import torch
from isaaclab.utils.math import quat_apply, quat_apply_inverse, matrix_from_quat


def rotate(q, points):
    return quat_apply(q.reshape(-1, 4), points.reshape(-1, 3)).reshape_as(points)


def unrotate(q, points):
    return quat_apply_inverse(q.reshape(-1, 4), points.reshape(-1, 3)).reshape_as(points)


def yaw(q):
    return torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]),
                       1 - 2 * (q[..., 2].square() + q[..., 3].square()))


def wrap_angle(angle):
    return torch.atan2(angle.sin(), angle.cos())


def projected_half_size(q, half):
    return (matrix_from_quat(q).abs() @ half.unsqueeze(-1)).squeeze(-1)


def slot_offsets(count, pitch, device):
    offsets = torch.zeros(count, 3, device=device)
    offsets[:, 0] = (torch.arange(count, device=device) - (count - 1) / 2) * pitch
    offsets[:, 2] = 0.015  # top of the existing 30 mm conveyor surface
    return offsets
