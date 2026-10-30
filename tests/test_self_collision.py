"""Real FCL geometry tests without Isaac Sim, GUI, or GPU startup."""
from pathlib import Path

import numpy as np
import pytest

from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.teleop.urdf_arm_ik import UrdfArm
from kuavo_isaaclab_scene.teleop.self_collision import (
    CollisionStop, RobotCollisionModel, SelfCollisionFilter,
)


@pytest.fixture(scope="module")
def robot():
    cfg = resolve_robot_model("s200062")
    policy = Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/configs/self_collision_s200062.json"
    model = RobotCollisionModel(cfg.urdf_path, policy)
    q = np.zeros(len(model.names))
    for side in ("left", "right"):
        arm = UrdfArm(cfg.urdf_path, side)
        for name, value in zip(arm.names, arm.ready_pose()):
            q[model.names.index(name)] = value
    for i, name in enumerate(model.names):
        if "_f_bar_" in name and name.endswith(("1_joint", "3_joint")):
            q[i] = -.25
        if "_b_bar_" in name and name.endswith(("1_joint", "3_joint")):
            q[i] = .25
    return model, q


def test_whole_robot_coverage_and_exclusion_scope(robot):
    model, _ = robot
    assert len(model.shapes) == 58
    assert len(model.pairs) > 1400
    pairs = {frozenset((model.shapes[a].link, model.shapes[b].link)) for a, b in model.pairs}
    for a, b in (("zarm_l4_link", "zarm_r4_link"), ("zarm_l4_link", "waist_yaw_link"),
                 ("l_f_finger", "r_b_finger"), ("l_f_finger", "zhead_2_link"),
                 ("zarm_l7_link", "leg_link"), ("zarm_r7_link", "base_link"),
                 ("zarm_l2_link", "zarm_l7_link")):
        assert frozenset((a, b)) in pairs
    assert frozenset(("l_f_finger", "l_b_finger")) not in pairs  # Intended empty closure.
    assert any(s.link == "l_f_finger" and s.source == "visual fallback" for s in model.shapes)
    assert all(reason for _, _, reason in model.exclusions)


@pytest.mark.parametrize("grip", [0., .1, .25])
def test_ready_pose_and_permitted_gripper_closure(robot, grip):
    model, initial = robot
    q = initial.copy()
    for i, name in enumerate(model.names):
        if "_f_bar_" in name and name.endswith(("1_joint", "3_joint")):
            q[i] = -grip
        if "_b_bar_" in name and name.endswith(("1_joint", "3_joint")):
            q[i] = grip
    guard = SelfCollisionFilter(model)
    np.testing.assert_allclose(guard.filter(q, q, 1/120, np.zeros_like(q)), q)


@pytest.mark.parametrize("link", ["l_f_finger", "zarm_r4_link", "zhead_2_link", "leg_link"])
def test_full_body_point_jacobian_matches_fk(robot, link):
    model, q = robot
    model.forward(q)
    point = model.frames[link][:3, 3].copy()
    jac = model.point_jacobian(link, point)
    for i in range(len(q)):
        changed = q.copy(); changed[i] += 1e-6
        model.forward(changed)
        numerical = (model.frames[link][:3, 3] - point) / 1e-6
        np.testing.assert_allclose(jac[:, i], numerical, atol=2e-5)


def test_distance_gradient_agrees_with_fcl(robot):
    model, q = robot
    d, rows = model.distances(q, .04, gradients=True)
    index, row = next((i, r) for i, r in rows if np.linalg.norm(r) > .02)
    i = np.argmax(np.abs(row))
    # 1e-6 joint perturbations fall below FCL's mesh/GJK distance tolerance.
    plus = q.copy(); plus[i] += 1e-4
    minus = q.copy(); minus[i] -= 1e-4
    d1, _ = model.distances(plus, .04)
    d2, _ = model.distances(minus, .04)
    assert (d1[index] - d2[index]) / 2e-4 == pytest.approx(row[i], abs=2e-3)


@pytest.fixture
def sliders(tmp_path):
    # Two independent branches. Neither pair is an adjacent-link exclusion.
    path = tmp_path / "sliders.urdf"
    path.write_text('''<robot name="sliders"><link name="root"/>
      <link name="left"><collision><geometry><sphere radius="0.06"/></geometry></collision></link>
      <link name="right"><collision><geometry><sphere radius="0.06"/></geometry></collision></link>
      <joint name="left_joint" type="prismatic"><parent link="root"/><child link="left"/>
        <origin xyz="-0.3 0 0"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="5"/></joint>
      <joint name="right_joint" type="prismatic"><parent link="root"/><child link="right"/>
        <origin xyz="0.3 0 0"/><axis xyz="1 0 0"/><limit lower="-1" upper="1" velocity="5"/></joint>
    </robot>''')
    return RobotCollisionModel(path)


def test_simultaneous_collision_targets_are_filtered(sliders):
    guard = SelfCollisionFilter(sliders)
    q = np.zeros(2)
    for _ in range(15):
        q = guard.filter(q, np.array([.3, -.3]), .05)
        d, _ = sliders.distances(q)
        assert d.min() >= guard.clearance - 1e-9


def test_safe_endpoints_do_not_hide_midpath_collision(sliders):
    guard = SelfCollisionFilter(sliders)
    q0, q1 = np.zeros(2), np.array([.6, -.6])
    d0, _ = sliders.distances(q0)
    d1, _ = sliders.distances(q1)
    assert d0.min() > .4 and d1.min() > .4
    assert not guard._swept_safe(q0, q1, d0, d1)


def test_current_collision_and_inertial_prediction_fail_closed(sliders):
    guard = SelfCollisionFilter(sliders)
    with pytest.raises(CollisionStop, match="Current pose"):
        guard.filter(np.array([.3, -.3]), np.zeros(2), .1)
    with pytest.raises(CollisionStop, match="velocity predicts"):
        guard.filter(np.zeros(2), np.zeros(2), .3, np.array([2., -2.]))


def test_nonfinite_commands_fail_closed(sliders):
    with pytest.raises(CollisionStop):
        SelfCollisionFilter(sliders).filter(np.zeros(2), np.array([np.nan, 0.]), .01)


def test_light_guard_checks_at_most_two_endpoints_and_never_optimizes(sliders, monkeypatch):
    guard = SelfCollisionFilter(sliders)
    calls = []
    original = sliders.distances
    def query(q, threshold=.05, gradients=False):
        calls.append(q.copy())
        assert not gradients
        return original(q, threshold, gradients)
    monkeypatch.setattr(sliders, "distances", query)
    monkeypatch.setattr(guard, "_swept_safe", lambda *args: pytest.fail("Heavy path query"))
    q = np.array([.2, -.2])
    np.testing.assert_allclose(guard.filter_light(q, np.array([.3, -.3]), .05), q)
    assert len(calls) == 2 and guard.status["modified"]
    assert guard.status["scale"] == 0
    calls.clear()
    np.testing.assert_allclose(guard.filter_light(q, q, .05), q)
    assert len(calls) == 1


def test_light_guard_intentionally_does_not_certify_between_endpoints(sliders):
    guard = SelfCollisionFilter(sliders)
    q0, q1 = np.zeros(2), np.array([.6, -.6])
    # Demonstrate the documented limitation, not a continuous safety claim.
    np.testing.assert_allclose(guard.filter_light(q0, q1, 1.), q1)
    with pytest.raises(CollisionStop, match="Current pose"):
        guard.filter_light(np.array([.3, -.3]), q0, .1)
    with pytest.raises(CollisionStop, match="Invalid"):
        guard.filter_light(q0, np.array([np.nan, 0.]), .1)


def test_missing_mesh_is_not_silently_skipped(tmp_path):
    path = tmp_path / "missing.urdf"
    path.write_text('<robot name="missing"><link name="root"><visual><geometry>'
                    '<mesh filename="missing.stl"/></geometry></visual></link></robot>')
    with pytest.raises(FileNotFoundError):
        RobotCollisionModel(path)
