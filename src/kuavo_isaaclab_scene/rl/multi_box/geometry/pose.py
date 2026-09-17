"""Quaternion and pose transforms for simulator-independent observations."""

from __future__ import annotations

import torch


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    result = quaternion.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quat_multiply(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = first.unbind(-1)
    bw, bx, by, bz = second.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def quat_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quaternion[..., 1:]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * uv + uuv)


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    norm = quaternion.norm(dim=-1, keepdim=True)
    if not bool(torch.isfinite(norm).all()) or bool((norm < 1e-8).any()):
        raise ValueError("Observation poses require finite, nonzero quaternions.")
    return quaternion / norm


def relative_pose(reference_pose: torch.Tensor, target_pose: torch.Tensor) -> torch.Tensor:
    """Express target xyz+wxyz pose in the reference frame."""
    if reference_pose.shape[-1] != 7 or target_pose.shape[-1] != 7:
        raise ValueError("Pose tensors must end in xyz+wxyz (7 values).")
    reference = reference_pose
    while reference.ndim < target_pose.ndim:
        reference = reference.unsqueeze(-2)
    rq = normalize_quaternion(reference[..., 3:])
    tq = normalize_quaternion(target_pose[..., 3:])
    inverse = quat_conjugate(rq)
    position = quat_apply(inverse, target_pose[..., :3] - reference[..., :3])
    quaternion = quat_multiply(inverse, tq)
    # q and -q are the same rotation.  A canonical sign avoids observation
    # discontinuities caused only by pose-estimator quaternion sign choices.
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    return torch.cat((position, quaternion), dim=-1)


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Encode the first two rotation-matrix columns as a continuous 6D vector."""
    q = normalize_quaternion(quaternion)
    w, x, y, z = q.unbind(-1)
    first = torch.stack((
        1 - 2 * (y * y + z * z),
        2 * (x * y + w * z),
        2 * (x * z - w * y),
    ), dim=-1)
    second = torch.stack((
        2 * (x * y - w * z),
        1 - 2 * (x * x + z * z),
        2 * (y * z + w * x),
    ), dim=-1)
    return torch.cat((first, second), dim=-1)


def pose_to_position_rotation_6d(pose: torch.Tensor) -> torch.Tensor:
    if pose.shape[-1] != 7:
        raise ValueError("Pose tensor must end in xyz+wxyz (7 values).")
    return torch.cat((pose[..., :3], quaternion_to_rotation_6d(pose[..., 3:])), dim=-1)
