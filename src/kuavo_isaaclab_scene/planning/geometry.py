"""CPU frame math. Poses are xyz + wxyz, in metres, with column vectors."""

from __future__ import annotations

import math
import numpy as np


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def axis_rotation(axis, angle: float) -> np.ndarray:
    axis = vector(axis, 3, "axis")
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or not math.isfinite(angle):
        raise ValueError("rotation requires nonzero axis and finite angle")
    x, y, z = axis / norm
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * skew @ skew


def origin_matrix(xyz, rpy) -> np.ndarray:
    r, p, y = vector(rpy, 3, "rpy")
    result = np.eye(4)
    result[:3, :3] = (axis_rotation([0, 0, 1], y)
                      @ axis_rotation([0, 1, 0], p) @ axis_rotation([1, 0, 0], r))
    result[:3, 3] = vector(xyz, 3, "xyz")
    return result


def pose_matrix(pose) -> np.ndarray:
    pose = vector(pose, 7, "xyz_wxyz pose")
    q = pose[3:]
    if not np.isclose(np.linalg.norm(q), 1.0, atol=1e-6, rtol=0):
        raise ValueError("pose quaternion must be unit length (wxyz)")
    w, x, y, z = q / np.linalg.norm(q)
    result = np.eye(4)
    result[:3, :3] = [
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
    ]
    result[:3, 3] = pose[:3]
    return result


def rigid_matrix(value) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (4, 4) or not np.isfinite(result).all():
        raise ValueError("transform must be a finite 4x4 matrix")
    r = result[:3, :3]
    if (not np.allclose(result[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0)):
        raise ValueError("transform must be rigid; apply asset scale to local geometry separately")
    return result


def inverse_transform(value) -> np.ndarray:
    value = rigid_matrix(value)
    result = np.eye(4)
    result[:3, :3] = value[:3, :3].T
    result[:3, 3] = -result[:3, :3] @ value[:3, 3]
    return result


def matrix_pose(value) -> list[float]:
    """Rigid matrix to xyz+wxyz, choosing a nonnegative quaternion scalar."""
    value = rigid_matrix(value)
    r = value[:3, :3]
    # Symmetric eigensystem is well-conditioned at 180-degree rotations too.
    k = np.array([
        [r[0, 0]-r[1, 1]-r[2, 2], r[0, 1]+r[1, 0], r[0, 2]+r[2, 0], r[2, 1]-r[1, 2]],
        [r[0, 1]+r[1, 0], r[1, 1]-r[0, 0]-r[2, 2], r[1, 2]+r[2, 1], r[0, 2]-r[2, 0]],
        [r[0, 2]+r[2, 0], r[1, 2]+r[2, 1], r[2, 2]-r[0, 0]-r[1, 1], r[1, 0]-r[0, 1]],
        [r[2, 1]-r[1, 2], r[0, 2]-r[2, 0], r[1, 0]-r[0, 1], np.trace(r)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    x, y, z, w = vectors[:, -1]
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return [*value[:3, 3].tolist(), *q.tolist()]


def pregrasp_transform(reference_world, grasp_reference, approach_axis_reference,
                       distance_m: float) -> np.ndarray:
    """Back off opposite the approach direction expressed in the reference frame.

    The reference is an explicit body/flap link, never an implicit box centre.
    Returns a target only, not an IK solution or a validated grasp.
    """
    if not math.isfinite(distance_m) or distance_m <= 0:
        raise ValueError("pregrasp distance must be finite and positive")
    axis = vector(approach_axis_reference, 3, "approach axis in reference frame")
    if not np.isclose(np.linalg.norm(axis), 1, atol=1e-6, rtol=0):
        raise ValueError("approach axis must be unit length")
    grasp = pose_matrix(grasp_reference)
    grasp[:3, 3] -= axis * distance_m
    return rigid_matrix(reference_world) @ grasp
