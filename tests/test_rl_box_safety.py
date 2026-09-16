"""Numerical box failure must be shared, bounded and independent of collisions."""

from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.box_safety import box_safety_checks, failed, guard_box_reward
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
from kuavo_isaaclab_scene.rl.algorithms.sac import soft_target
from kuavo_isaaclab_scene.rl.algorithms.common import generalized_advantage
from kuavo_isaaclab_scene.rl.mdp.robot_safety import robot_motion_unsafe


def state():
    origins = torch.tensor([[0., 0., 0.], [80., 50., 10.]])
    baseline = torch.tensor([[1., 1.5], [1., 1.5]])
    centers = origins[:, None].repeat(1, 2, 1)
    centers[..., 2] += baseline
    poses = torch.cat((centers, torch.zeros(2, 2, 4)), -1)
    poses[..., 3] = 1
    return centers, poses, torch.zeros(2, 2, 6), baseline, origins


@pytest.mark.parametrize("spec", [task_spec("pick", collision_constraints_enabled=False), MultiBoxSpec()])
def test_normal_lift_is_allowed_but_any_box_at_ceiling_fails(spec):
    centers, poses, velocities, baseline, origins = state()
    centers[0, 0, 2] += .06
    centers[1, 1, 2] += .50
    checks = box_safety_checks(centers, poses, velocities, baseline, origins, spec)
    assert failed(checks).tolist() == [False, True]
    assert checks["box_over_lift"].tolist() == [False, True]


@pytest.mark.parametrize("field,value", [("position", float("nan")), ("pose", float("inf")),
                                         ("velocity", float("nan"))])
def test_nonfinite_state_fails_even_without_collision_termination(field, value):
    centers, poses, velocities, baseline, origins = state()
    tensor = {"position": centers, "pose": poses, "velocity": velocities}[field]
    tensor[1, 0, 0] = value
    checks = box_safety_checks(centers, poses, velocities, baseline, origins, task_spec("pick"))
    assert checks["box_non_finite"].tolist() == [False, True]
    assert failed(checks).tolist() == [False, True]


def test_speed_guard_catches_blowup_before_position_has_moved():
    centers, poses, velocities, baseline, origins = state()
    velocities[0, 0, 0] = 1e30
    velocities[1, 1, 5] = 101
    checks = box_safety_checks(centers, poses, velocities, baseline, origins, task_spec("pick"))
    assert checks["box_excessive_speed"].all()
    assert not checks["box_over_lift"].any()


def test_robot_motion_guard_catches_finite_blowup_and_nonfinite_state():
    robot = NS(data=NS(joint_pos=torch.zeros(3, 2), joint_vel=torch.zeros(3, 2)))
    robot.data.joint_vel[1, 0] = 101.0
    robot.data.joint_pos[2, 1] = float("nan")
    assert robot_motion_unsafe(robot).tolist() == [False, True, True]


def test_invalid_terminal_step_gets_no_nan_or_extreme_shaping():
    command = NS(box_safety_failure=torch.tensor([False, True, True]))
    env = NS(command_manager=NS(get_term=lambda _: command))
    @guard_box_reward
    def shaping(env, scale=1.):
        return torch.tensor([.25, float("nan"), 1e30]) * scale
    torch.testing.assert_close(shaping(env, scale=2.), torch.tensor([.5, 0., 0.]))


def test_terminal_bootstrap_does_not_multiply_zero_by_nan():
    reward = torch.tensor([-60., 1.])
    done = torch.tensor([True, False])
    result = soft_target(reward, done, torch.tensor([float("nan"), 2.]),
                         torch.tensor([float("nan"), 0.]), .1, .99)
    torch.testing.assert_close(result, torch.tensor([-60., 2.98]))
    advantages, _ = generalized_advantage(reward[None], torch.zeros(1, 2),
        torch.tensor([[float("nan"), 2.]]), done[None], done[None])
    torch.testing.assert_close(advantages[0], result)


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), 0., .05])
def test_invalid_or_unreachable_safety_ceiling_is_rejected(limit):
    for spec in (task_spec("pick"), MultiBoxSpec()):
        with pytest.raises(ValueError, match="Box safety"):
            replace(spec, max_box_lift_height=limit).validate()
