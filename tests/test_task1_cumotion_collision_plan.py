from itertools import product

import numpy as np
import pytest
import yaml

from data_collection.task1_cumotion_collision_plan import (
    GRIPPER_COLLISION_FRAMES,
    ROBOT_COLLISION_FRAMES,
    SELF_COLLISION_IGNORE,
    axis_alignment_error_deg,
    collision_world_config,
    cover_cuboid,
    robot_spheres,
    xrdf,
)


def test_collision_model_covers_and_internally_ignores_complete_grippers():
    assert len(GRIPPER_COLLISION_FRAMES) == 28
    for side in ("l", "r"):
        gripper_frames = {
            frame for frame in GRIPPER_COLLISION_FRAMES if frame.startswith(f"{side}_")
        }
        ignored_from_wrist = set(SELF_COLLISION_IGNORE[f"zarm_{side}7_link"])
        assert ignored_from_wrist == gripper_frames


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


def test_collision_world_allows_only_selected_target_flaps():
    snapshot = {
        "colliders": [
            {"robot": True, "path": "/robot"},
            {"robot": False, "path": "/rack"},
            {"robot": False, "path": "/MediumBox_0/box/flap_right"},
            {"robot": False, "path": "/MediumBox_0/box/flap_left"},
            {"robot": False, "path": "/MediumBox_0/box/bottom"},
        ]
    }
    world = {"cuboid": {f"obstacle_{index}": index for index in range(1, 5)}}

    filtered, allowed = collision_world_config(
        snapshot, world, allow_target_flap_contact=True
    )

    assert filtered == {"cuboid": {"obstacle_1": 1, "obstacle_4": 4}}
    assert allowed == [
        "/MediumBox_0/box/flap_right",
        "/MediumBox_0/box/flap_left",
    ]


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


def test_robot_spheres_replace_gripper_cuboid_with_mesh_fit():
    snapshot = {
        "colliders": [
            {
                "robot": True,
                "owner": f"/robot/{name}",
                "pose_w": [index, 0, 0, 1, 0, 0, 0],
                "dims": [.1, .1, .1],
            }
            for index, name in enumerate(sorted(ROBOT_COLLISION_FRAMES))
        ]
    }
    body_names = sorted(ROBOT_COLLISION_FRAMES)
    runtime = {
        "body_names": body_names,
        "body_poses_w": [
            [index, 0, 0, 1, 0, 0, 0]
            for index, _ in enumerate(body_names)
        ],
    }
    mesh = {"l_f_finger": [{"center": [.01, .02, .03], "radius": .004}]}

    result = robot_spheres(snapshot, runtime, .05, .002, mesh)

    assert result["l_f_finger"] == [
        {"center": [.01, .02, .03], "radius": pytest.approx(.006)}
    ]
    assert len(result["zarm_l2_link"]) == 8


def test_robot_spheres_use_snapshot_owner_pose_for_local_geometry():
    body_names = sorted(ROBOT_COLLISION_FRAMES)
    snapshot = {
        "colliders": [
            {
                "robot": True,
                "owner": f"/robot/{name}",
                "owner_pose_w": [float(index), 0, 0, 1, 0, 0, 0],
                "pose_w": [float(index) + 0.2, 0, 0, 1, 0, 0, 0],
                "dims": [.02, .02, .02],
            }
            for index, name in enumerate(body_names)
        ]
    }
    runtime = {
        "body_names": body_names,
        "body_poses_w": [
            [float(index), 0, 10, 1, 0, 0, 0]
            for index, _ in enumerate(body_names)
        ],
    }

    result = robot_spheres(snapshot, runtime, .05, 0.0)

    np.testing.assert_allclose(result["zarm_l4_link"][0]["center"], [0.2, 0, 0])


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
