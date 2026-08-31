"""Source-frame and grasp-region checks for both wrist-camera rigs."""

import itertools
import math
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kuavo_isaaclab_scene.paths import ASSET_DIR
from kuavo_isaaclab_scene.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.wrist_camera_mount import (
    CAMERA_BODY_TO_ROS_OPTICAL_ROT,
    S200062_D405_MOUNTS,
    S63_ROBOTIQ_D405_MOUNTS,
)


def _rotation(quat):
    w, x, y, z = np.asarray(quat) / np.linalg.norm(quat)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])


def _axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return _rotation((math.cos(angle/2), *(axis * math.sin(angle/2))))


def _fk(root, base, positions):
    """Read actual URDF chains instead of duplicating their mount constants."""
    frames = {base: np.eye(4)}
    pending = list(root.findall('joint'))
    while pending:
        advanced = False
        for joint in pending[:]:
            parent = joint.find('parent').get('link')
            if parent not in frames:
                continue
            origin = joint.find('origin')
            attrs = {} if origin is None else origin.attrib
            xyz = np.fromstring(attrs.get('xyz', '0 0 0'), sep=' ')
            r, p, y = np.fromstring(attrs.get('rpy', '0 0 0'), sep=' ')
            transform = np.eye(4)
            transform[:3, 3] = xyz
            transform[:3, :3] = (
                _axis_rotation((0, 0, 1), y)
                @ _axis_rotation((0, 1, 0), p)
                @ _axis_rotation((1, 0, 0), r)
            )
            if joint.get('type') in ('revolute', 'continuous'):
                axis = np.fromstring(joint.find('axis').get('xyz'), sep=' ')
                transform[:3, :3] = transform[:3, :3] @ _axis_rotation(
                    axis, positions.get(joint.get('name'), 0.0)
                )
            frames[joint.find('child').get('link')] = frames[parent] @ transform
            pending.remove(joint)
            advanced = True
        if not advanced:
            break  # Other branches of the robot are outside this local rig.
    return frames


def _mount_matrix(mount):
    matrix = np.eye(4)
    matrix[:3, :3] = _rotation(mount.rot)
    matrix[:3, 3] = mount.pos
    return matrix


def test_camera_body_axes_are_not_ros_optical_axes() -> None:
    rotation = _rotation(CAMERA_BODY_TO_ROS_OPTICAL_ROT)
    assert rotation @ [0, 0, 1] == pytest.approx([1, 0, 0])
    assert rotation @ [0, -1, 0] == pytest.approx([0, 0, 1])


def test_head_optical_axis_faces_workcell_instead_of_ceiling():
    root = ET.parse(ASSET_DIR / 'kuavo_s200062/urdf/biped_s200062.urdf').getroot()
    frames = _fk(root, 'waist_yaw_link', {})
    model = resolve_robot_model('s200062')
    body = frames[model.head_camera_body]
    optical = body @ _mount_matrix(model.head_camera_mount)
    assert optical[:3, 3] == pytest.approx(body[:3, 3])
    assert optical[:3, 2] == pytest.approx(body[:3, 0])
    # Source mount is pitched 20.5 degrees downward from body-forward.
    assert optical[0, 2] > .9 and -.5 < optical[2, 2] < -.2


@pytest.mark.parametrize('mounts', [S200062_D405_MOUNTS, S63_ROBOTIQ_D405_MOUNTS])
def test_d405_optical_mounts_are_normalized_mirrors(mounts) -> None:
    left = mounts['left']
    right = mounts['right']

    assert right.pos == pytest.approx((left.pos[0], -left.pos[1], left.pos[2]))
    mirror = np.diag([1, -1, 1])
    assert _rotation(right.rot)[:, 2] == pytest.approx(mirror @ _rotation(left.rot)[:, 2])
    assert math.sqrt(sum(value * value for value in left.rot)) == pytest.approx(1.0)
    assert math.sqrt(sum(value * value for value in right.rot)) == pytest.approx(1.0)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_s200062_collapsed_pose_matches_source_chain_plus_optical_rotation(side):
    root = ET.parse(ASSET_DIR / 'kuavo_s200062/urdf/biped_s200062.urdf').getroot()
    frames = _fk(root, f'{side[0]}_twofinger_base', {})
    actual = frames[f'{side[0]}_d405_camera'] @ _mount_matrix(
        resolve_robot_model('s200062').wrist_camera_mounts[side]
    )
    assert actual == pytest.approx(_mount_matrix(S200062_D405_MOUNTS[side]), abs=1.2e-5)


@pytest.mark.parametrize('side', ['left', 'right'])
def test_s200062_only_changes_sensor_axes_not_physical_position(side):
    mount = resolve_robot_model('s200062').wrist_camera_mounts[side]
    assert mount.pos == (0.0, 0.0, 0.0)
    assert mount.rot == CAMERA_BODY_TO_ROS_OPTICAL_ROT


@pytest.mark.parametrize('side', ['left', 'right'])
def test_s63_rotates_source_rig_and_sets_it_back_for_open_jaws(side):
    source = _mount_matrix(S200062_D405_MOUNTS[side])
    flip = np.diag([-1.0, 1.0, -1.0, 1.0])  # Ry(pi)
    expected = flip @ source
    expected[:3, 3] -= 0.030 * expected[:3, 2]
    assert _mount_matrix(S63_ROBOTIQ_D405_MOUNTS[side]) == pytest.approx(expected)


@pytest.mark.parametrize('model', ['s200062', 's63'])
@pytest.mark.parametrize('side', ['left', 'right'])
@pytest.mark.parametrize('fraction', [0.0, 0.5, 1.0])
def test_finger_contact_regions_are_in_front_and_inside_wrist_fov(model, side, fraction):
    settings = resolve_robot_model(model)
    mount = _mount_matrix(settings.wrist_camera_mounts[side])
    if model == 's200062':
        root = ET.parse(ASSET_DIR / 'kuavo_s200062/urdf/biped_s200062.urdf').getroot()
        prefix = side[0]
        q = 0.55 * fraction
        positions = {f'{prefix}_f_bar_1_joint': q, f'{prefix}_f_bar_3_joint': -q,
                     f'{prefix}_b_bar_1_joint': -q, f'{prefix}_b_bar_3_joint': q}
        frames = _fk(root, f'{prefix}_twofinger_base', positions)
        camera = frames[f'{prefix}_d405_camera'] @ mount
        points = []
        for jaw, x in [('f', -0.02), ('b', 0.02)]:
            for y, z in itertools.product([-0.009, 0.009], [-0.04, -0.06]):
                points.append(frames[f'{prefix}_{jaw}_finger'] @ [x, y, z, 1])
    else:
        root = ET.parse(ASSET_DIR / 'robotiq_2f85/urdf/robotiq_2f85.urdf').getroot()
        q = 0.8 * fraction
        positions = {f'{jaw}_{link}_joint': sign*q for jaw in ['left', 'right']
                     for link, sign in [('driver', 1), ('coupler', -1), ('spring_link', 1), ('follower', -1)]}
        frames = _fk(root, 'base_mount', positions)
        camera = mount
        points = [frames[f'{jaw}_pad'] @ [x, -0.0026, z, 1]
                  for jaw, x, z in itertools.product(['left', 'right'], [-0.011, 0.011], [0.003, 0.034])]
    optical_points = (np.linalg.inv(camera) @ np.asarray(points).T).T[:, :3]
    # Actual wrist camera: 160x120, focal length 12 mm, aperture 24 mm.
    # ROS optical projection uses +Z depth, not +X depth.
    depth = optical_points[:, 2]
    assert np.all(depth > 0.03) and np.all(depth < 2.0)
    u = 80 * optical_points[:, 0] / depth + 80
    v = 80 * optical_points[:, 1] / depth + 60
    assert np.all((u > 4) & (u < 156)), (model, side, fraction, u)
    assert np.all((v > 4) & (v < 116)), (model, side, fraction, v)
