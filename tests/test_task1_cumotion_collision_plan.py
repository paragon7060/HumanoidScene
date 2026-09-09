from itertools import product

import numpy as np
import pytest
import yaml

from data_collection.task1_cumotion_collision_plan import (
    axis_alignment_error_deg,
    cover_cuboid,
    xrdf,
)


def test_axis_alignment_error_uses_rotated_local_axis():
    angle = np.deg2rad(15.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert axis_alignment_error_deg(rotation, [1, 0, 0], [1, 0, 0]) == pytest.approx(15.0)


def test_collision_sphere_cover_contains_oriented_cuboid():
    pose = [1.0, 2.0, 3.0, np.cos(.2), 0.0, 0.0, np.sin(.2)]
    dimensions = np.array([.12, .08, .15])
    spheres = cover_cuboid(pose, dimensions, .05)
    centers = np.stack([center for center, _ in spheres])
    radii = np.asarray([radius for _, radius in spheres])

    from data_collection.task1_cumotion_collision_plan import pose_matrix

    transform = pose_matrix(pose)
    for corner in product(*((-dimension / 2, dimension / 2) for dimension in dimensions)):
        point = (transform @ np.r_[corner, 1])[:3]
        assert np.any(np.linalg.norm(centers - point, axis=1) <= radii + 1e-12)


def test_xrdf_keeps_world_and_self_collision_models_separate():
    defaults = {f"zarm_l{index}_joint": 0.0 for index in range(1, 8)}
    world_spheres = {"zarm_l2_link": [{"center": [0, 0, 0], "radius": .06}]}
    self_spheres = {"zarm_l2_link": [{"center": [0, 0, 0], "radius": .0575}]}

    value = yaml.safe_load(
        xrdf(
            side="left",
            defaults=defaults,
            world_spheres=world_spheres,
            self_spheres=self_spheres,
        )
    )

    assert value["cspace"]["joint_names"] == list(defaults)
    assert value["world_collision"]["geometry"] == "kuavo_world_spheres"
    assert value["self_collision"]["geometry"] == "kuavo_self_spheres"
    assert value["geometry"]["kuavo_world_spheres"]["spheres"] == world_spheres
    assert value["geometry"]["kuavo_self_spheres"]["spheres"] == self_spheres
