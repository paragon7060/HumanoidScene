import numpy as np
import pytest
import yaml

from data_collection.task1.approach import ARM_JOINT_NAMES
from data_collection.task1.endpoint import (
    candidate_angles,
    center_out_offsets,
    rmpflow_config,
    rmpflow_xrdf,
    integrate_rmpflow,
    rotation_vector,
)


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
