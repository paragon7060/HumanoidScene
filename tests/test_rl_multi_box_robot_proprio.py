"""CPU checks for the deployable robot telemetry boundary."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.state.robot_proprio import (
    ACTUATED_BODY_JOINTS,
    closure_fraction,
    kinematic_base_twist_world,
    robot_state_from_sensors,
)


def _pose(*shape):
    result = torch.zeros(*shape, 7)
    result[..., 3] = 1
    return result


def test_body_joint_order_excludes_passive_linkage_and_gripper_drivers():
    assert len(ACTUATED_BODY_JOINTS) == 20
    assert ACTUATED_BODY_JOINTS[:4] == (
        "knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint")
    assert all("bar" not in name and "finger" not in name for name in ACTUATED_BODY_JOINTS)


def test_gripper_closure_projects_two_driver_angles_and_clamps():
    opened = torch.tensor([-0.25, 0.25])
    closed = torch.tensor([0.0, 0.0])
    measured = torch.tensor([[-0.25, 0.25], [-0.125, 0.125], [0.0, 0.0],
                             [0.1, -0.1]])
    torch.testing.assert_close(
        closure_fraction(measured, opened, closed),
        torch.tensor([0.0, 0.5, 1.0, 1.0]))
    with pytest.raises(ValueError, match="must differ"):
        closure_fraction(measured, opened, opened)


def test_robot_telemetry_accepts_only_fixed_body_width():
    def build(joints):
        return robot_state_from_sensors(
            joint_pos=torch.zeros(2, joints), joint_vel=torch.zeros(2, joints),
            base_pose_world=_pose(2), base_twist_world=torch.zeros(2, 6),
            tcp_pose_world=_pose(2, 2), gripper_position=torch.zeros(2, 2),
            gripper_command=torch.zeros(2, 2))

    state = build(20)
    assert state.joint_pos.shape == (2, 20)
    with pytest.raises(ValueError, match="fixed 20-joint"):
        build(25)


def test_planar_drive_twist_is_rotated_into_world_frame():
    angle = torch.tensor(torch.pi / 4)
    yaw_ninety = torch.tensor([[angle.cos(), 0.0, 0.0, angle.sin()]])
    twist = kinematic_base_twist_world(yaw_ninety, torch.tensor([[0.2, 0.0, 0.3]]))
    torch.testing.assert_close(twist, torch.tensor([[0.0, 0.2, 0.0, 0.0, 0.0, 0.3]]),
                               atol=1e-6, rtol=0)
