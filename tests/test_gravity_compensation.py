"""Gravity feedforward, posture invariants and servo configuration contracts."""

import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS
import xml.etree.ElementTree as ET

import pytest
import torch

from kuavo_isaaclab_scene.robots.gravity_compensation import (
    configure_gravity_compensation, feedforward_joint_ids, gravity_drive_bias, gravity_joint_ids,
    load_s63_servo_gains, wbc_acceleration_profile,
)


def test_bias_adds_gravity_inside_pd_without_compensating_passive_or_locked_joints():
    names = ["l_f_bar_3_joint", "zarm_r1_joint", "knee_joint", "wheel_left_front_joint",
             "waist_yaw_joint", "zarm_l7_joint", "zhead_1_joint"]
    ids = gravity_joint_ids(names)
    assert ids == [1, 2, 4, 5]
    gravity = torch.tensor([[1., 20., -100., 1., 2., 5., 1.], [1., 30., -80., 1., 3., 4., 1.]])
    stiffness = torch.full_like(gravity, 100.)
    stiffness[0, 5] = 0
    limits = torch.tensor([-2., 2.]).expand(2, 7, 2).clone()
    limits[1, 2] = torch.tensor([.1, .1002])  # arms-only body lock
    bias = gravity_drive_bias(gravity, stiffness, limits, ids)
    expected = torch.tensor([[0., .2, -1., 0., .02, 0., 0.], [0., .3, 0., 0., .03, .04, 0.]])
    torch.testing.assert_close(bias, expected)
    q_cmd, q = torch.zeros_like(gravity), torch.full_like(gravity, .01)
    torch.testing.assert_close(stiffness * (q_cmd + bias - q),
                               stiffness * (q_cmd - q) + stiffness * expected)


def test_articulation_write_does_not_accumulate_bias_or_change_logical_targets():
    path = Path(__file__).parents[1] / "src/kuavo_isaaclab_scene/robots/gravity_articulation.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    class Base:
        def _apply_actuator_model(self):
            self._joint_pos_target_sim[:] = self.data.joint_pos_target
            self.data.computed_torque[:] = 0
        def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
            self.data.joint_pos_target[:] = target
        def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
            self.data.joint_vel_target[:] = target
    ns = {"Articulation": Base, "torch": torch, "gravity_drive_bias": gravity_drive_bias}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    robot = ns[cls.name]()
    robot._gravity_joint_ids = [0, 1]
    target = torch.tensor([[.2, -.4, 0.]])
    robot.data = NS(joint_pos_target=target.clone(), joint_vel_target=torch.zeros(1, 3),
                    joint_pos=torch.zeros(1, 3), joint_vel=torch.zeros(1, 3),
                    joint_stiffness=torch.full((1, 3), 100.),
                    joint_pos_limits=torch.tensor([[[-2., 2.]] * 3]),
                    joint_effort_limits=torch.tensor([[20., 20., 20.]]),
                    computed_torque=torch.zeros(1, 3), applied_torque=torch.zeros(1, 3))
    robot.root_physx_view = NS(
        get_gravity_compensation_forces=lambda: torch.tensor([[10., 30., 1.]]),
        get_generalized_mass_matrices=lambda: torch.eye(3).unsqueeze(0),
        get_coriolis_and_centrifugal_compensation_forces=lambda: torch.tensor([[1., 2., 3.]]),
    )
    robot._joint_pos_target_sim = torch.zeros(1, 3)
    robot.gravity_compensation_torque = torch.zeros(1, 3)
    robot.command_feedforward_torque = torch.zeros(1, 3)
    robot.command_feedforward_mask = torch.zeros(1, 3, dtype=torch.bool)
    robot.total_feedforward_torque = torch.zeros(1, 3)
    robot.command_feedforward_mode = "off"
    robot.inverse_dynamics_torque = torch.zeros(1, 3)
    robot._wbc_accel_kp = torch.tensor([[2., 2., 0.]])
    robot._wbc_accel_kd = torch.tensor([[1., 1., 0.]])
    robot._wbc_accel_limit = torch.tensor([[20., 20., 0.]])
    robot._inverse_dynamics_update_pending = True
    robot.gravity_compensation_bias = torch.zeros(1, 3)
    for _ in range(20):
        robot._apply_actuator_model()
        torch.testing.assert_close(robot.data.joint_pos_target, target)
        torch.testing.assert_close(robot._joint_pos_target_sim, torch.tensor([[.3, -.1, 0.]]))
        torch.testing.assert_close(robot.data.computed_torque, torch.tensor([[10., 30., 0.]]))
        torch.testing.assert_close(robot.data.applied_torque, torch.tensor([[10., 20., 0.]]))
    command = torch.tensor([[5., 7., 9.]])
    mask = torch.tensor([[True, False, False]])
    robot.set_command_feedforward_torque(command, "replace_gravity", mask)
    robot._apply_actuator_model()
    torch.testing.assert_close(robot._joint_pos_target_sim, torch.tensor([[.25, -.1, 0.]]))
    torch.testing.assert_close(robot.total_feedforward_torque, torch.tensor([[5., 30., 1.]]))
    robot.set_command_feedforward_torque(command, "add_to_gravity", mask)
    robot._apply_actuator_model()
    torch.testing.assert_close(robot._joint_pos_target_sim, torch.tensor([[.35, -.1, 0.]]))
    torch.testing.assert_close(robot.total_feedforward_torque, torch.tensor([[15., 30., 1.]]))
    robot.set_command_feedforward_torque(command, "inverse_dynamics", mask)
    robot._apply_actuator_model()
    # M=I, C=[1,2,3], G=[10,30,1], qdd=[.4,-.8,0]. Only joint 0 uses ID.
    torch.testing.assert_close(robot.inverse_dynamics_torque, torch.tensor([[11.4, 31.2, 4.]]))
    torch.testing.assert_close(robot.total_feedforward_torque, torch.tensor([[11.4, 30., 1.]]))


def test_floating_inverse_dynamics_uses_joint_block_without_predicting_root_reaction():
    path = Path(__file__).parents[1] / "src/kuavo_isaaclab_scene/robots/gravity_articulation.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))

    class Base:
        def _apply_actuator_model(self):
            self._joint_pos_target_sim[:] = self.data.joint_pos_target
            self.data.computed_torque[:] = 0

    ns = {"Articulation": Base, "torch": torch, "gravity_drive_bias": gravity_drive_bias}
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), ns)
    robot = ns[cls.name]()
    robot._gravity_joint_ids = [0, 1]
    robot.data = NS(
        joint_pos_target=torch.tensor([[1.0, 2.0]]),
        joint_vel_target=torch.zeros(1, 2),
        joint_pos=torch.zeros(1, 2),
        joint_vel=torch.zeros(1, 2),
        joint_stiffness=torch.full((1, 2), 100.0),
        joint_pos_limits=torch.tensor([[[-3.0, 3.0]] * 2]),
        joint_effort_limits=torch.full((1, 2), 100.0),
        computed_torque=torch.zeros(1, 2),
        applied_torque=torch.zeros(1, 2),
    )
    mass = torch.eye(8).unsqueeze(0)
    mass[0, 0, 6] = 2.0
    mass[0, 4, 7] = 3.0
    gravity = torch.tensor([[10., 20., 30., 40., 50., 60., 7., 8.]])
    coriolis = torch.tensor([[1., 2., 3., 4., 5., 6., .1, .2]])
    robot.root_physx_view = NS(
        get_gravity_compensation_forces=lambda: gravity,
        get_generalized_mass_matrices=lambda: mass,
        get_coriolis_and_centrifugal_compensation_forces=lambda: coriolis,
    )
    robot._joint_pos_target_sim = torch.zeros(1, 2)
    robot.gravity_compensation_torque = torch.zeros(1, 2)
    robot.command_feedforward_torque = torch.zeros(1, 2)
    robot.command_feedforward_mask = torch.ones(1, 2, dtype=torch.bool)
    robot.total_feedforward_torque = torch.zeros(1, 2)
    robot.command_feedforward_mode = "inverse_dynamics"
    robot.inverse_dynamics_torque = torch.zeros(1, 2)
    robot._wbc_accel_kp = torch.ones(1, 2)
    robot._wbc_accel_kd = torch.zeros(1, 2)
    robot._wbc_accel_limit = torch.full((1, 2), 20.0)
    robot._inverse_dynamics_update_pending = True
    robot.gravity_compensation_bias = torch.zeros(1, 2)

    robot._apply_actuator_model()

    torch.testing.assert_close(robot.inverse_dynamics_torque, torch.tensor([[8.1, 10.2]]))
    torch.testing.assert_close(robot.total_feedforward_torque, torch.tensor([[8.1, 10.2]]))
    source = path.read_text()
    # Root/joint cross terms are intentionally not exported as an external
    # wrench: desired acceleration is not achieved acceleration under contact
    # and actuator limits, so that prediction can excite the floating base.
    assert "root_dynamic_feedforward_wrench_w" not in source


def test_wbc_acceleration_profile_matches_active_s63_task_and_excludes_other_joints():
    names = ["knee_joint", "waist_yaw_joint", "zarm_l1_joint", "zarm_r4_joint",
             "zarm_l7_joint", "zhead_1_joint", "l_f_bar_1_joint"]
    kp, kd, limit = wbc_acceleration_profile(names)
    torch.testing.assert_close(kp, torch.tensor([30., 30., 300., 300., 70., 0., 0.]))
    torch.testing.assert_close(kd, torch.tensor([6.2, 6.2, 18., 40., 30., 0., 0.]))
    torch.testing.assert_close(limit, torch.tensor([20., 20., 300., 300., 300., 0., 0.]))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_compensation_fails_explicitly(invalid):
    with pytest.raises(ValueError, match="Non-finite"):
        gravity_drive_bias(torch.tensor([[invalid]]), torch.ones(1, 1),
                           torch.tensor([[[-1., 1.]]]), [0])


def test_body_profile_extends_inverse_dynamics_to_the_torso_that_carries_the_arms():
    names = ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint", "wheel_left_front_joint"] + [f"zarm_{s}{i}_joint" for s in "lr" for i in range(1, 8)]
    kp = wbc_acceleration_profile(names)[0].tolist()
    torso = {"knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"}
    body = {names[i] for i in feedforward_joint_ids(names, "s63-body-id", kp)}
    arm_only = {names[i] for i in feedforward_joint_ids(names, "s63-arm-id", kp)}
    assert body - arm_only == torso
    assert len(arm_only) == 14 and "wheel_left_front_joint" not in body
    # Every joint the acceleration task defines gains for must be applied, so
    # torso gains cannot silently become unused configuration again.
    assert body == {name for name, gain in zip(names, kp) if gain > 0}


def test_feedforward_selection_requires_both_complete_arms():
    names = ["knee_joint"] + [f"zarm_l{i}_joint" for i in range(1, 8)]
    with pytest.raises(ValueError, match="seven-joint"):
        feedforward_joint_ids(names, "s63-body-id", wbc_acceleration_profile(names)[0].tolist())


def test_servo_profile_supports_joint_specific_gains_and_rejects_invalid_values(tmp_path, monkeypatch):
    profile = {"actuators": load_s63_servo_gains()}
    profile["actuators"]["arms"]["stiffness"] = {"zarm_[lr]1_joint": 150., "zarm_[lr][2-7]_joint": 100.}
    path = tmp_path / "servo.json"
    path.write_text(json.dumps(profile))
    monkeypatch.setenv("KUAVO_S63_SERVO_CONFIG", str(path))
    assert load_s63_servo_gains()["arms"]["stiffness"]["zarm_[lr]1_joint"] == 150.
    profile["actuators"]["height_axis"]["stiffness"] = 0
    path.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="positive"):
        load_s63_servo_gains()


@pytest.mark.parametrize("model,gripper,count", [
    ("s200062", None, 18), ("s63", "none", 18), ("s63", "leju-twofinger", 18),
    ("s56", None, 27), ("s56", "none", 27), ("s56", "s56_twofinger", 27),
])
def test_all_model_variants_select_their_body_and_both_arms(model, gripper, count):
    from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
    settings = resolve_robot_model(model, gripper)
    joints = [joint.attrib["name"] for joint in ET.parse(settings.urdf_path).getroot().findall("joint")
              if joint.attrib["type"] != "fixed"]
    selected = [joints[i] for i in gravity_joint_ids(joints)]
    assert len(selected) == count
    assert set(settings.teleop_body_joint_names).issubset(selected)
    assert len([name for name in selected if name.startswith("zarm_")]) == 14
    assert not any("bar_" in name or "head" in name or "wheel" in name for name in selected)


@pytest.mark.parametrize("model", ["s200062", "s63", "s56"])
def test_common_writer_preserves_model_gains_except_explicit_s63_profile(monkeypatch, model):
    module = "kuavo_isaaclab_scene.robots.gravity_articulation"
    writer = type("GravityCompensatedArticulation", (), {})
    monkeypatch.setitem(sys.modules, module, NS(GravityCompensatedArticulation=writer))
    monkeypatch.setitem(sys.modules, "isaaclab.sim", NS(JointDrivePropertiesCfg=lambda: NS()))
    cfg = NS(class_type=None, spawn=NS(joint_drive_props=None),
             actuators={name: NS(stiffness=123., damping=4.)
                                       for name in ("height_axis", "arms", "upper_body")})
    configure_gravity_compensation(cfg, NS(name=model))
    assert cfg.class_type is writer
    assert cfg.spawn.joint_drive_props.drive_type == "force"
    assert cfg.actuators["arms"].stiffness == (220. if model == "s63" else 123.)
    assert cfg.actuators["arms"].damping == (22. if model == "s63" else 4.)
