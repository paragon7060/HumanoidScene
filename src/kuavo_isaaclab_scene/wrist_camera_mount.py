"""Wrist-camera body-to-optical transforms for the two supported grippers.

The source URDF attaches each camera below ``*_twofinger_base`` through:

``d405_camera_connect -> d405_camera_base -> d405_camera``.

The URDF camera *body* frame points +X toward the two-finger contact region.
It is not an optical frame. Isaac Lab's ``convention="ros"`` instead means
+Z forward, +X image-right and -Y image-up. The correction below maps these
optical axes to body +X, -Y and +Z, respectively; all quaternions are wxyz.

S200062 keeps the physical camera positions and only adds that optical
rotation. Collapsed poses compose the complete source chain and correction.
Rounded URDF angles are represented by exact -pi, pi/3 and +/-pi/2.

S63 uses a different gripper: Robotiq fingers extend along mount +Z, whereas
S200062 and the S56 QiangNao hand extend along -Z. The S63 virtual camera rig is therefore
adapted by Ry(pi), for both position and orientation, in the gripper mount
frame, then moved 30 mm backward along the optical viewing axis so both
fully open contact pads fit the image. This is a simulation adaptation,
not an S63 hardware calibration. S56 uses a centered -Z-facing virtual camera
above its official empty end-effector frame so all five articulated fingers
stay visible through their full curl; this is likewise not hardware calibration.

Source: LejuRobotics/kuavo-ros-opensource, revision
5d60846b092b425a7a3c06479bdfdbc2b100e890.
"""

from __future__ import annotations

from math import cos, pi, sin, sqrt
from typing import NamedTuple


class WristCameraMount(NamedTuple):
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]


# ROS optical +Z -> camera body +X, optical -Y -> camera body +Z.
CAMERA_BODY_TO_ROS_OPTICAL_ROT = (0.5, -0.5, 0.5, -0.5)

_SIN_15 = sin(pi / 12.0)
_COS_15 = cos(pi / 12.0)
_CAMERA_Z = -0.068857 - 0.0192 * sqrt(3.0) / 2.0
_S63_SETBACK = 0.030

S200062_D405_MOUNTS = {
    "left": WristCameraMount(
        pos=(-0.009, 0.046683, _CAMERA_Z),
        rot=(_SIN_15, _COS_15, 0.0, 0.0),
    ),
    "right": WristCameraMount(
        pos=(-0.009, -0.046683, _CAMERA_Z),
        rot=(0.0, 0.0, _COS_15, _SIN_15),
    ),
}

S63_ROBOTIQ_D405_MOUNTS = {
    "left": WristCameraMount(
        pos=(0.009, 0.046683 + _S63_SETBACK / 2.0,
             -_CAMERA_Z - _S63_SETBACK * sqrt(3.0) / 2.0),
        rot=(0.0, 0.0, _SIN_15, -_COS_15),
    ),
    "right": WristCameraMount(
        pos=(0.009, -0.046683 - _S63_SETBACK / 2.0,
             -_CAMERA_Z - _S63_SETBACK * sqrt(3.0) / 2.0),
        rot=(_COS_15, -_SIN_15, 0.0, 0.0),
    ),
}

S56_QIANGNAO_D405_MOUNTS = {
    # The tool frame is 170 mm below the wrist. Move the virtual sensor back
    # above it and look straight down the QiangNao hand's -Z reach axis.
    side: WristCameraMount(
        pos=(0.03, 0.0, 0.22),
        rot=(0.0, 1.0, 0.0, 0.0),
    )
    for side in ("left", "right")
}
