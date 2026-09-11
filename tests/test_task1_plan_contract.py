import numpy as np
import pytest

from data_collection.task1.contract import (
    ARM_JOINT_NAMES,
    WAIST_ARM_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    compose_waist_arm,
    layout_for_joint_names,
    safe_waist_bounds,
    split_trajectory,
    validate_plan,
)


def test_waist_arm_layout_has_canonical_order():
    layout = layout_for_joint_names(WAIST_ARM_JOINT_NAMES)

    assert layout.joint_names == tuple(WAIST_JOINT_NAMES + ARM_JOINT_NAMES)
    assert layout.waist_indices == (0, 1)
    assert layout.arm_indices == tuple(range(2, 16))


def test_arm_only_layout_requires_explicit_baseline_opt_in():
    with pytest.raises(ValueError, match="arm-only baseline"):
        layout_for_joint_names(ARM_JOINT_NAMES)

    layout = layout_for_joint_names(ARM_JOINT_NAMES, allow_arm_only_baseline=True)
    assert layout.waist_indices == ()
    assert layout.arm_indices == tuple(range(14))


def test_validate_plan_rejects_mismatched_trajectory_width():
    plan = {"joint_names": WAIST_ARM_JOINT_NAMES, "trajectory": [[0.0] * 14]}

    with pytest.raises(ValueError, match="16 columns"):
        validate_plan(plan)


def test_split_and_compose_round_trip():
    waist = np.array([[0.1, -0.2], [0.1, -0.2]])
    arms = np.arange(28, dtype=float).reshape(2, 14)
    full = compose_waist_arm(waist, arms)

    got_waist, got_arms = split_trajectory(WAIST_ARM_JOINT_NAMES, full)
    np.testing.assert_allclose(got_waist, waist)
    np.testing.assert_allclose(got_arms, arms)


def test_compose_accepts_one_constant_waist_posture():
    arms = np.arange(42, dtype=float).reshape(3, 14)

    full = compose_waist_arm([0.15, 0.05], arms)

    np.testing.assert_allclose(full[:, :2], [[0.15, 0.05]] * 3)
    np.testing.assert_allclose(full[:, 2:], arms)


def test_safe_waist_bounds_intersect_joint_limits_with_safety_envelope():
    lower, upper = safe_waist_bounds(
        {"waist_pitch_joint": (-1.0, 2.0), "waist_yaw_joint": (-4.0, 4.0)}
    )

    np.testing.assert_allclose(lower, np.deg2rad([-9.0, -45.0]))
    np.testing.assert_allclose(upper, np.deg2rad([25.0, 45.0]))
