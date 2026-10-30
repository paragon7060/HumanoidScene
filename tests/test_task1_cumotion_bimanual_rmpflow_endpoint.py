import numpy as np
import pytest
import yaml

from data_collection.task1.approach import ARM_JOINT_NAMES
from data_collection.task1.endpoint import (
    candidate_angles,
    center_out_offsets,
    endpoint_targets_from_editor_state,
    front_target_candidates,
    parser,
    rmpflow_config,
    rmpflow_xrdf,
    integrate_rmpflow,
    rotation_vector,
    simultaneous_pose_errors,
    solve_simultaneous_jacobian,
)


def test_pair_endpoint_uses_left_tcp_and_pair_midpoint():
    inputs = endpoint_targets_from_editor_state(
        {
            "target_mode": "paired",
            "pair_grasp": {
                "active_arm": "left",
                "grasp_midpoint_b_m": [0.55, -0.15, 1.12],
                "closing_axis_b": [0.0, 1.0, 0.0],
                "selected_flap_paths": ["/a/flap_right", "/b/flap_left"],
            },
        }
    )

    assert inputs["active_arm"] == "left"
    assert inputs["tool_frames"] == ["zarm_l7_endeffector_center"]
    np.testing.assert_allclose(
        inputs["nominal_centers_b_m"], [[0.55, -0.15, 1.12]]
    )
    np.testing.assert_allclose(inputs["inward_normals_b"], [[0.0, 1.0, 0.0]])
    assert inputs["allowed_target_flap_paths"] == [
        "/a/flap_right",
        "/b/flap_left",
    ]


def test_pose_errors_can_evaluate_one_selected_tool_frame():
    class Orientation:
        @staticmethod
        def matrix():
            return np.eye(3)

    class Kinematics:
        @staticmethod
        def position(q, frame):
            assert frame == "left_tcp"
            return np.asarray([1.0, 2.0, 3.0])

        @staticmethod
        def orientation(q, frame):
            assert frame == "left_tcp"
            return Orientation()

        @staticmethod
        def jacobian(q, frame):
            assert frame == "left_tcp"
            return np.zeros((6, 7))

    pos, rot, _, jacobians = simultaneous_pose_errors(
        Kinematics(),
        np.zeros(7),
        np.asarray([[1.1, 2.0, 3.0]]),
        [np.eye(3)],
        tool_frames=["left_tcp"],
    )

    np.testing.assert_allclose(pos, [[0.1, 0.0, 0.0]])
    np.testing.assert_allclose(rot, [[0.0, 0.0, 0.0]])
    assert len(jacobians) == 1


def test_jacobian_line_search_keeps_one_selected_tool_frame():
    class Orientation:
        @staticmethod
        def matrix():
            return np.eye(3)

    class Kinematics:
        @staticmethod
        def position(q, frame):
            assert frame == "left_tcp"
            return q[:3]

        @staticmethod
        def orientation(q, frame):
            assert frame == "left_tcp"
            return Orientation()

        @staticmethod
        def jacobian(q, frame):
            assert frame == "left_tcp"
            value = np.zeros((6, 7))
            value[:3, :3] = np.eye(3)
            return value

    q, _, _ = solve_simultaneous_jacobian(
        Kinematics(),
        np.zeros(7),
        -np.ones(7),
        np.ones(7),
        np.asarray([[0.1, 0.0, 0.0]]),
        [np.eye(3)],
        position_tolerance_m=1e-4,
        orientation_tolerance_deg=1.0,
        tool_frames=["left_tcp"],
    )

    assert q[0] == pytest.approx(0.1, abs=1e-4)


def test_center_out_offsets_cover_line_and_prefer_nearby_points():
    offsets = center_out_offsets(0.2, 21)

    assert offsets[0] == pytest.approx(0.0)
    assert offsets.min() == pytest.approx(-0.1)
    assert offsets.max() == pytest.approx(0.1)
    assert np.all(np.diff(np.abs(offsets)) >= -1e-12)


def test_candidate_angles_include_zero_and_non_step_aligned_maximum():
    np.testing.assert_allclose(candidate_angles(12.0, 5.0), [0.0, 5.0, 10.0, 12.0])
    np.testing.assert_allclose(candidate_angles(28.0, 5.0, 20.0), [20.0, 25.0, 28.0])


def test_rmpflow_xrdf_uses_one_bimanual_cspace():
    names = ["waist_pitch_joint", "waist_yaw_joint", *ARM_JOINT_NAMES]
    value = yaml.safe_load(rmpflow_xrdf(names, dict.fromkeys(names, 0.0), {}, {}))

    assert value["cspace"]["joint_names"] == names
    assert value["tool_frames"] == [
        "zarm_l7_endeffector_center",
        "zarm_r7_endeffector_center",
    ]
    assert len(value["cspace"]["acceleration_limits"]) == 16


def test_rmpflow_config_matches_cspace_size_and_has_required_empty_collision_lists():
    value = yaml.safe_load(rmpflow_config(14))

    assert len(value["joint_limit_buffers"]) == 14
    assert value["body_capsules"] == []
    assert value["body_collision_controllers"] == []


def test_integrate_rmpflow_uses_convergence_callback():
    class ZeroFlow:
        @staticmethod
        def eval_accel(q, velocity, accel):
            accel[:] = 0.0

    checks = []
    q, velocity, steps = integrate_rmpflow(
        ZeroFlow(),
        np.zeros(2),
        -np.ones(2),
        np.ones(2),
        timestep_s=0.01,
        duration_s=10.0,
        convergence_check=lambda value: checks.append(value.copy()) or True,
    )

    assert steps == 121
    assert len(checks) == 1
    np.testing.assert_allclose(q, 0.0)
    np.testing.assert_allclose(velocity, 0.0)


def test_rotation_vector_recovers_axis_angle():
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    np.testing.assert_allclose(rotation_vector(rotation), [0.0, 0.0, angle])


def test_endpoint_defaults_to_arms_only_and_requires_explicit_waist_opt_in():
    args = parser().parse_args(["--snapshot-dir", "/tmp/s", "--urdf", "/tmp/r"])

    assert args.include_waist is False
    assert args.target_x_offsets_m == [0.03, 0.04, 0.05]
    assert args.tool_down_angles_deg == [45.0, 50.0, 56.0]


def test_endpoint_accepts_explicit_seven_dof_active_arm_seed():
    args = parser().parse_args(
        [
            "--snapshot-dir",
            "/tmp/s",
            "--urdf",
            "/tmp/r",
            "--active-arm-seed-rad",
            *[str(index / 10) for index in range(7)],
        ]
    )

    np.testing.assert_allclose(args.active_arm_seed_rad, np.arange(7) / 10)

    waist_args = parser().parse_args(
        ["--snapshot-dir", "/tmp/s", "--urdf", "/tmp/r", "--include-waist"]
    )
    assert waist_args.include_waist is True

    legacy_arm_args = parser().parse_args(
        ["--snapshot-dir", "/tmp/s", "--urdf", "/tmp/r", "--arm-only-baseline"]
    )
    assert legacy_arm_args.include_waist is False


def test_front_target_candidates_preserve_front_first_order():
    values = front_target_candidates([0.03, 0.04, 0.05], 0.015)

    np.testing.assert_allclose(
        values,
        [[0.03, 0.0, 0.015], [0.04, 0.0, 0.015], [0.05, 0.0, 0.015]],
    )
