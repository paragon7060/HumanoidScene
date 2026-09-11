from itertools import product

import numpy as np
import pytest
import yaml

from data_collection.task1.collision import (
    DEFAULT_GRIPPER_SPHERE_CONFIG,
    GRIPPER_COLLISION_FRAMES,
    ROBOT_COLLISION_FRAMES,
    SELF_COLLISION_IGNORE,
    axis_alignment_error_deg,
    box_region_goal_points,
    collision_world_config,
    cover_cuboid,
    editor_region_geometry,
    line_goal_points,
    robot_spheres,
    rotation_error_deg,
    runtime_joint_defaults,
    tool_down_angle_deg,
    tool_down_orientation_targets,
    target_flap_line_geometry,
    xrdf,
)


def test_default_gripper_sphere_config_survives_package_move():
    assert DEFAULT_GRIPPER_SPHERE_CONFIG.is_file()


def test_editor_region_geometry_accepts_live_transit_state_key():
    centers, size = editor_region_geometry(
        {
            "transit_center_b_m": [[0.3, 0.2, 1.1], [0.3, -0.1, 1.1]],
            "transit_region_size_b_m": [0.22, 0.28, 0.1],
        },
        "transit",
    )

    np.testing.assert_allclose(centers, [[0.3, 0.2, 1.1], [0.3, -0.1, 1.1]])
    np.testing.assert_allclose(size, [0.22, 0.28, 0.1])


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


def test_tool_down_goalset_keeps_closing_axis_and_spans_requested_angles():
    rotations, angles = tool_down_orientation_targets(
        [0.0, -1.0, 0.0], 30.0, 90.0, 5.0
    )

    np.testing.assert_allclose(angles, np.arange(30.0, 95.0, 5.0))
    for rotation, angle in zip(rotations, angles, strict=True):
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(rotation[:, 0], [0.0, -1.0, 0.0], atol=1e-12)
        assert np.linalg.det(rotation) == pytest.approx(1.0)
        assert tool_down_angle_deg(rotation) == pytest.approx(angle)
        assert rotation_error_deg(rotation, rotation) == pytest.approx(0.0)


def test_tool_down_goalset_includes_non_step_aligned_maximum():
    _, angles = tool_down_orientation_targets([0.0, 1.0, 0.0], 30.0, 88.0, 5.0)

    np.testing.assert_allclose(angles, [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 88])


def test_box_region_goal_points_cover_box_and_prefer_center():
    center = np.array([0.6, 0.2, 1.4])
    points = box_region_goal_points(center, np.full(3, 0.1), 5)

    assert points.shape == (125, 3)
    np.testing.assert_allclose(points[0], center)
    np.testing.assert_allclose(points.min(axis=0), center - 0.05)
    np.testing.assert_allclose(points.max(axis=0), center + 0.05)
    assert len(np.unique(points, axis=0)) == 125


def test_line_goal_points_stay_on_axis_and_prefer_center():
    center = np.array([0.6, 0.2, 1.4])
    axis = np.array([-1.0, 0.0, -0.1])
    points = line_goal_points(center, axis, 0.2, 21)

    assert points.shape == (21, 3)
    np.testing.assert_allclose(points[0], center)
    displacements = points - center
    np.testing.assert_allclose(np.cross(displacements, axis), 0.0, atol=1e-12)
    np.testing.assert_allclose(
        np.sort(np.linalg.norm(displacements, axis=1)),
        np.sort(np.abs(np.linspace(-0.1, 0.1, 21))),
    )


def test_target_flap_line_geometry_uses_long_collider_axis_in_base_frame():
    angle = np.deg2rad(30.0)
    quaternion = [np.cos(angle / 2), 0.0, np.sin(angle / 2), 0.0]
    snapshot = {
        "colliders": [
            {
                "robot": False,
                "path": f"/World/MediumBox_0/MediumBox/{name}",
                "pose_w": [0.0, 0.0, 0.0, *quaternion],
                "dims": [0.003, 0.21, 0.11],
            }
            for name in ("flap_right", "flap_left")
        ]
    }

    axes, lengths = target_flap_line_geometry(
        snapshot, {"root_pose_w": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
    )

    np.testing.assert_allclose(axes, [[0.0, 1.0, 0.0]] * 2, atol=1e-12)
    np.testing.assert_allclose(lengths, [0.21, 0.21])


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

    from data_collection.task1.collision import pose_matrix

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


def test_runtime_joint_defaults_preserve_body_posture_and_apply_editor_overrides():
    runtime = {
        "joint_names": [
            "wheel_left_front_joint",
            "knee_joint",
            "waist_pitch_joint",
            "zhead_1_joint",
            "zarm_l1_joint",
        ],
        "joint_positions": [1.0, 0.25, 0.30, -0.20, 0.10],
        "pose_editor_state": {
            "joints": [
                {"name": "zhead_1_joint", "value": 0.50},
                {"name": "zarm_l1_joint", "value": 0.40},
            ]
        },
    }

    assert runtime_joint_defaults(runtime) == {
        "knee_joint": 0.25,
        "waist_pitch_joint": 0.30,
        "zarm_l1_joint": 0.40,
    }
