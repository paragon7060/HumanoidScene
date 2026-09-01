import numpy as np

from kuavo_isaaclab_scene.browser_teleop_bridge import BrowserEyeView, BrowserTrackingSample
from kuavo_isaaclab_scene.stereo_camera_calibration import (
    calibrations_from_tracking,
    camera_world_pose,
    projection_to_intrinsics,
)


def _projection() -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=np.float32)
    matrix[0, 0] = 1.0
    matrix[1, 1] = 1.0
    matrix[2, 2] = -1.0
    matrix[2, 3] = -1.0
    matrix[3, 2] = -0.2
    return matrix.reshape(-1, order="F")


def test_projection_to_intrinsics_uses_eye_fov():
    intrinsic = projection_to_intrinsics(_projection(), 512, 512)
    np.testing.assert_allclose(intrinsic, [[256, 0, 256], [0, 256, 256], [0, 0, 1]])


def test_eye_views_become_head_local_calibration():
    identity = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    head = np.array([0.0, 0.0, 1.6, *identity], dtype=np.float32)
    left = BrowserEyeView(
        "left", np.array([0.0, 0.032, 1.6, *identity], dtype=np.float32), _projection()
    )
    right = BrowserEyeView(
        "right", np.array([0.0, -0.032, 1.6, *identity], dtype=np.float32), _projection()
    )
    sample = BrowserTrackingSample(1, 10.0, head, None, None, (left, right), 1.0)

    calibrations = calibrations_from_tracking(sample, 512, 512)

    assert [item.eye for item in calibrations] == ["left", "right"]
    assert np.isclose(np.linalg.norm(calibrations[0].local_position - calibrations[1].local_position), 0.064)
    position, orientation = camera_world_pose(
        np.array([1.0, 2.0, 3.0]), identity, calibrations[0]
    )
    np.testing.assert_allclose(position, [1.0, 2.032, 3.0], atol=1.0e-6)
    np.testing.assert_allclose(orientation, identity, atol=1.0e-6)
