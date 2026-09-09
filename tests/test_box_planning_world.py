from itertools import product

import numpy as np
import pytest

from kuavo_isaaclab_scene.planning.geometry import matrix_pose, pose_matrix, origin_matrix
from kuavo_isaaclab_scene.planning.world import (
    bounded_cuboid,
    cover_cuboid,
    omit_instance_colliders,
    world_config,
)


@pytest.mark.parametrize("angles", [[0, 0, 0], [np.pi, 0, 0], [0, np.pi, 0],
                                  [0, 0, np.pi], [.2, -.5, 1.2]])
def test_matrix_pose_roundtrip(angles):
    original = origin_matrix([1, 2, 3], angles)
    np.testing.assert_allclose(pose_matrix(matrix_pose(original)), original, atol=1e-12)


def test_individual_collider_bounds_apply_scale_once():
    transform = origin_matrix([1, 2, 3], [0, 0, np.pi/2])
    transform[:3, :3] *= [2, 3, 4]
    pose, dims = bounded_cuboid([.1, .2, .3], [.3, .6, .9], transform)
    np.testing.assert_allclose(dims, [.4, 1.2, 2.4])
    np.testing.assert_allclose(pose[:3], [-.2, 2.4, 5.4], atol=1e-12)


def test_sheared_collider_is_not_silently_approximated():
    transform = np.eye(4)
    transform[0, 1] = .2
    with pytest.raises(ValueError, match="rigid"):
        bounded_cuboid([0, 0, 0], [1, 1, 1], transform)


def test_flat_physics_collider_gets_explicit_minimum_thickness():
    pose, dims = bounded_cuboid(
        [0, 0, 0],
        [2, 3, 0],
        np.eye(4),
        minimum_dimension_m=.001,
    )
    np.testing.assert_allclose(pose, [1, 1.5, 0, 1, 0, 0, 0])
    np.testing.assert_allclose(dims, [2, 3, .001])


def test_sphere_cover_contains_corners_and_interior():
    pose = matrix_pose(origin_matrix([1, 2, 3], [.4, .1, .2]))
    dims = np.array([.12, .08, .15])
    spheres = cover_cuboid(pose, dims, .05)
    centers = np.array([s["center"] for s in spheres])
    radii = np.array([s["radius"] for s in spheres])
    for local in product(*(np.linspace(-d/2, d/2, 7) for d in dims)):
        point = (pose_matrix(pose) @ np.r_[local, 1])[:3]
        assert np.any(np.linalg.norm(centers - point, axis=1) <= radii + 1e-12)


def test_world_keeps_rack_parts_and_excludes_robot_only():
    def item(robot, x):
        return {"robot": robot, "pose_w": [x, 0, 0, 1, 0, 0, 0], "dims": [.1, .2, .3]}
    snapshot = {"colliders": [item(True, 0), item(False, 1), item(False, 2)]}
    world = world_config(snapshot, [.5, 0, 0, 1, 0, 0, 0])
    assert len(world["cuboid"]) == 2
    assert world["cuboid"]["obstacle_1"]["pose"][0] == .5
    assert world["cuboid"]["obstacle_2"]["pose"][0] == 1.5


def test_parked_instance_colliders_are_omitted_with_audit_trail():
    snapshot = {
        "colliders": [
            {"path": "/World/MediumBox_0/Body", "robot": False},
            {"path": "/World/LargeBox_0/Body/bottom", "robot": False},
            {"path": "/World/LargeBox_0/flap_left", "robot": False},
            {"path": "/World/Kuavo/link", "robot": True},
        ],
        "continuous_collision_guarantee": False,
    }

    filtered = omit_instance_colliders(snapshot, ("LargeBox_0",))

    assert [item["path"] for item in filtered["colliders"]] == [
        "/World/MediumBox_0/Body",
        "/World/Kuavo/link",
    ]
    assert filtered["pose_editor_omitted_colliders"] == [
        "/World/LargeBox_0/Body/bottom",
        "/World/LargeBox_0/flap_left",
    ]
