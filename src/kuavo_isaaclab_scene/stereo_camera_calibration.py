"""Convert WebXR eye views into robot-head-local stereo camera calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .browser_teleop_bridge import BrowserTrackingSample


@dataclass(frozen=True)
class EyeCameraCalibration:
    eye: str
    local_position: np.ndarray
    local_orientation: np.ndarray
    intrinsic_matrix: np.ndarray
    projection_matrix: np.ndarray


def _normalized_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-8)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = _normalized_quat(quat)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]])


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.array([0.0, *np.asarray(vector, dtype=np.float64)])
    return _quat_multiply(_quat_multiply(quat, pure), _quat_conjugate(quat))[1:]


def projection_to_intrinsics(projection: np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert a column-major WebXR projection matrix into pixel intrinsics."""
    projection = np.asarray(projection, dtype=np.float64)
    if projection.shape != (16,) or not np.all(np.isfinite(projection)):
        raise ValueError("WebXR projection matrix must contain 16 finite values.")
    matrix = projection.reshape((4, 4), order="F")
    fx = abs(float(matrix[0, 0])) * width * 0.5
    fy = abs(float(matrix[1, 1])) * height * 0.5
    cx = width * (1.0 - float(matrix[0, 2])) * 0.5
    cy = height * (1.0 + float(matrix[1, 2])) * 0.5
    if min(fx, fy) <= 1.0 or not 0.0 <= cx <= width or not 0.0 <= cy <= height:
        raise ValueError("WebXR projection matrix produced invalid camera intrinsics.")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)


def calibrations_from_tracking(
    sample: BrowserTrackingSample,
    width: int,
    height: int,
) -> tuple[EyeCameraCalibration, ...]:
    """Express both eye views relative to the WebXR viewer center."""
    if sample.head is None:
        return ()
    head = np.asarray(sample.head, dtype=np.float64)
    head_inverse = _quat_conjugate(head[3:])
    calibrations = []
    for view in sample.views:
        delta_position = np.asarray(view.pose[:3], dtype=np.float64) - head[:3]
        local_position = _quat_rotate(head_inverse, delta_position)
        local_orientation = _quat_multiply(head_inverse, _normalized_quat(view.pose[3:]))
        calibrations.append(
            EyeCameraCalibration(
                eye=view.eye,
                local_position=local_position.astype(np.float32),
                local_orientation=_normalized_quat(local_orientation).astype(np.float32),
                intrinsic_matrix=projection_to_intrinsics(view.projection_matrix, width, height),
                projection_matrix=np.asarray(view.projection_matrix, dtype=np.float32).copy(),
            )
        )
    return tuple(calibrations)


def camera_world_pose(
    center_position: np.ndarray,
    center_orientation: np.ndarray,
    calibration: EyeCameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    """Place a calibrated eye relative to the current robot head camera."""
    center_orientation = _normalized_quat(center_orientation)
    position = np.asarray(center_position, dtype=np.float64) + _quat_rotate(
        center_orientation, calibration.local_position
    )
    orientation = _quat_multiply(center_orientation, calibration.local_orientation)
    return position.astype(np.float32), _normalized_quat(orientation).astype(np.float32)
