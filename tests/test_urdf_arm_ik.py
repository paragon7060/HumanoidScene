"""Kinematic tests against bundled URDFs; no GUI, GPU, or physics startup."""

import numpy as np
import pytest

from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.teleop.urdf_arm_ik import UrdfArm, box_qp, rotation_error
from kuavo_isaaclab_scene.teleop.teleop_servo import RESPONSIVE


@pytest.mark.parametrize("model", ["s200062", "s63", "s56"])
@pytest.mark.parametrize("side", ["left", "right"])
def test_ready_pose_and_urdf_jacobian(model, side):
    arm = UrdfArm(resolve_robot_model(model).urdf_path, side)
    q = arm.ready_pose()
    assert np.all(q >= arm.lower + .079)
    assert np.all(q <= arm.upper - .079)
    p, r, jac, _ = arm.fk(q)
    assert .29 < p[0] - arm.shoulder[0] < .35
    assert -.35 < p[2] - arm.shoulder[2] < -.29
    for i in range(7):
        delta = np.zeros(7); delta[i] = 1e-6
        p1, r1, _, _ = arm.fk(q + delta)
        np.testing.assert_allclose(jac[:3, i], (p1 - p) / 1e-6, atol=1e-5)
        np.testing.assert_allclose(jac[3:, i], rotation_error(r1, r) / 1e-6, atol=1e-5)


def test_box_qp_resolves_other_joints_when_one_saturates():
    j = np.array([[1., 1.]])
    h = j.T @ j + .001 * np.eye(2)
    b = j.T @ np.array([1.])
    x = box_qp(h, b, np.array([0., -2.]), np.array([0., 2.]))
    np.testing.assert_allclose(x, [0., 1 / 1.001], atol=1e-6)


@pytest.mark.parametrize("bad", ["position", "rotation", "axis", "limits", "nan"])
def test_live_model_mismatch_fails_closed(bad):
    arm = UrdfArm(resolve_robot_model("s200062").urdf_path, "right")
    q = arm.ready_pose()
    p, r, jac, _ = arm.fk(q)
    limits = np.column_stack((arm.lower, arm.upper))
    arm.validate_live(q, p, r, jac, limits)
    if bad == "position":
        p += [.02, 0., 0.]
    elif bad == "rotation":
        r = r @ np.diag([-1., -1., 1.])
    elif bad == "axis":
        jac[:, 2] *= -1
    elif bad == "limits":
        limits[1, 0] -= .1
    else:
        p[0] = np.nan
    with pytest.raises(ValueError):
        arm.validate_live(q, p, r, jac, limits)


def test_box_qp_kkt_conditions():
    rng = np.random.default_rng(3)
    for _ in range(30):
        a = rng.normal(size=(6, 7))
        h = a.T @ a + .01 * np.eye(7)
        b = rng.normal(size=7)
        low, high = -rng.uniform(.01, 1., 7), rng.uniform(.01, 1., 7)
        x = box_qp(h, b, low, high)
        grad = h @ x - b
        assert np.all(x >= low - 1e-9) and np.all(x <= high + 1e-9)
        free = (x > low + 1e-7) & (x < high - 1e-7)
        assert np.all(np.abs(grad[free]) < 1e-6)
        assert np.all(grad[x <= low + 1e-7] >= -1e-6)
        assert np.all(grad[x >= high - 1e-7] <= 1e-6)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("hz", [30, 60])
def test_reach_from_ready_pose_and_unreachable_goal(side, hz):
    arm = UrdfArm(resolve_robot_model("s200062").urdf_path, side)
    q = arm.ready_pose()
    rest = q.copy()
    p, r, _, _ = arm.fk(q)
    target = p + [.10, 0., .03]
    velocity = np.zeros(7)
    for _ in range(2 * hz):
        velocity, effective, status = arm.step(q, target, r, rest, velocity, 1/hz, RESPONSIVE)
        q += velocity / hz
        assert np.all(q >= arm.lower) and np.all(q <= arm.upper)
        assert np.max(np.abs(velocity)) <= RESPONSIVE.max_velocity + 1e-8
    assert np.linalg.norm(target - arm.fk(q)[0]) < .02
    assert status["projection_m"] == pytest.approx(0.)
    target = arm.shoulder + [2., 0., 0.]
    for _ in range(hz):
        velocity, effective, status = arm.step(q, target, r, rest, velocity, 1/hz, RESPONSIVE)
        q += velocity / hz
        assert np.all(q >= arm.lower) and np.all(q <= arm.upper)
    assert status["projection_m"] > 1.
    assert np.linalg.norm(effective - arm.shoulder) <= .95 * arm.reach + 1e-9


def test_joint_stop_damper_and_bad_pose_fail_closed():
    arm = UrdfArm(resolve_robot_model("s200062").urdf_path, "right")
    q = arm.ready_pose()
    q[2] = arm.upper[2] - .01
    p, r, _, _ = arm.fk(q)
    velocity, _, _ = arm.step(q, p + [.2, .2, .1], r, q, np.zeros(7), 1/60, RESPONSIVE)
    assert velocity[2] <= 1e-9
    q[2] = arm.upper[2] + .1
    with pytest.raises(ValueError, match="angles/limits"):
        arm.step(q, p, r, q, np.zeros(7), 1/60, RESPONSIVE)
