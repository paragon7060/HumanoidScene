from itertools import product

import numpy as np
import pytest

from kuavo_isaaclab_scene.planning.geometry import matrix_pose, pose_matrix, origin_matrix
from kuavo_isaaclab_scene.planning.world import bounded_cuboid, cover_cuboid, world_config


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
