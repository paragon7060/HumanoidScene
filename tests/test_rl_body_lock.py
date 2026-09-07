"""Test body constraint/reset indexing with CPU tensors, no simulator or rendering."""

from types import SimpleNamespace
from pathlib import Path
import runpy

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.body_lock import FixedBody, ARM_JOINT_NAMES
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


class Robot:
    is_fixed_base = True
    joint_names = ["zarm_l1_joint", "waist_yaw_joint", "zhead_1_joint", "zhead_2_joint",
                   "knee_joint", "wheel_left_front_joint", "l_b_bar_1_joint", "l_b_bar_3_joint"]

    def __init__(self):
        q = torch.tensor([[0.6, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8],
                          [0.7, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9]])
        self.data = SimpleNamespace(joint_pos=q, joint_pos_limits=torch.tensor([[[-2., 2.]]*8]*2),
                                    default_joint_pos=torch.zeros_like(q))
        self.position_targets = torch.zeros_like(q)
        self.velocity_targets = torch.ones_like(q)
        self.state_writes = 0

    def write_joint_position_limit_to_sim(self, limits, *, joint_ids, env_ids, warn_limit_violation):
        self.data.joint_pos_limits[env_ids[:, None], joint_ids] = limits
        # Match Isaac Lab's documented/default-position-clamping side effect.
        self.data.default_joint_pos[env_ids[:, None], joint_ids] = limits[..., 0]

    def write_joint_state_to_sim(self, q, dq, *, joint_ids, env_ids):
        self.data.joint_pos[env_ids[:, None], joint_ids] = q
        assert not dq.any()
        self.state_writes += 1

    def set_joint_position_target(self, q, *, joint_ids, env_ids):
        self.position_targets[env_ids[:, None], joint_ids] = q

    def set_joint_velocity_target(self, dq, *, joint_ids, env_ids):
        self.velocity_targets[env_ids[:, None], joint_ids] = dq


def test_locks_only_body_and_preserves_policy_reference():
    robot = Robot(); lock = FixedBody(robot)
    lock.reset()
    assert lock.joint_names == ["waist_yaw_joint", "zhead_1_joint", "zhead_2_joint",
                                "knee_joint", "wheel_left_front_joint"]
    limits = robot.data.joint_pos_limits[:, lock.joint_ids]
    torch.testing.assert_close(limits.mean(-1), lock.targets)
    torch.testing.assert_close(limits[..., 1] - limits[..., 0], torch.full_like(lock.targets, 2e-4))
    # All arm and both driven/passive hand joints remain free.
    assert (robot.data.joint_pos_limits[:, [0, 6, 7], 0] == -2).all()
    assert not robot.data.default_joint_pos.any()


def test_partial_reset_latches_new_pose_and_keeps_other_env_locked():
    robot = Robot(); lock = FixedBody(robot)
    lock.reset()
    first_targets = lock.targets[0].clone()
    first_limits = robot.data.joint_pos_limits[0].clone()
    # New reset state may be far outside the previous narrow lock; original
    # physical limits, rather than the last episode's lock, must validate it.
    robot.data.joint_pos[1, 1:6] = -0.4
    lock.reset(torch.tensor([1]))
    torch.testing.assert_close(lock.targets[0], first_targets)
    torch.testing.assert_close(robot.data.joint_pos_limits[0], first_limits)
    assert (lock.targets[1] == -0.4).all()
    before = robot.state_writes
    robot.position_targets.zero_()
    lock.apply()
    torch.testing.assert_close(robot.position_targets[:, lock.joint_ids], lock.targets)
    assert not robot.velocity_targets[:, lock.joint_ids].any()
    assert robot.state_writes == before  # no joint teleporting during a physics step


def test_free_root_and_invalid_positions_rejected():
    robot = Robot(); robot.is_fixed_base = False
    with pytest.raises(ValueError, match="fixed"):
        FixedBody(robot)
    robot.is_fixed_base = True
    lock = FixedBody(robot)
    robot.data.joint_pos[0, 1] = 3
    with pytest.raises(ValueError, match="physical"):
        lock.reset([0])


def test_arm_order_and_mode_validation():
    assert len(ARM_JOINT_NAMES) == 14
    assert ARM_JOINT_NAMES[0] == "zarm_l1_joint" and ARM_JOINT_NAMES[7] == "zarm_r1_joint"
    task_spec("pick", control_mode="arms-only").validate()
    with pytest.raises(ValueError, match="control_mode"):
        task_spec("pick", control_mode="typo").validate()
    with pytest.raises(ValueError, match="navigation"):
        task_spec("full", control_mode="arms-only").validate()


def test_stationary_pick_example_configures_requested_thresholds():
    path = Path(__file__).resolve().parents[1] / "configs/rl_pick_arms_only.py"
    custom = runpy.run_path(str(path))
    spec = custom["configure_task"](task_spec("pick"))
    spec.validate()
    assert spec.control_mode == "arms-only" and spec.required_grasp_hands == 1
    assert spec.grasp_hand == "right" and spec.grasp_hand_indices == (1,)
    assert spec.lift_height == 0.06 and spec.cargo_per_box == 0
    actions = SimpleNamespace(upper_body=SimpleNamespace(), left_gripper=SimpleNamespace(),
                              right_gripper=SimpleNamespace())
    agent = SimpleNamespace(policy=SimpleNamespace())
    custom["configure"](SimpleNamespace(actions=actions), agent)
    assert actions.upper_body.scale == 0.02 and agent.policy.init_noise_std == 0.15
