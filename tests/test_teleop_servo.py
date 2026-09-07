"""Small CPU-only servo checks; these are not robot/physics benchmarks."""

import pytest

torch = pytest.importorskip("torch")
from kuavo_isaaclab_scene.teleop.teleop_servo import (
    SMOOTH, RESPONSIVE, arm_response_profile, joint_servo_step,
)


def test_auto_uses_responsive_for_scaled_and_absolute_controllers():
    for mapping in ("scaled", "absolute"):
        assert arm_response_profile("auto", mapping, "controllers") is RESPONSIVE
        assert arm_response_profile("smooth", mapping, "controllers") is SMOOTH
    for mapping, mode in (("relative", "controllers"), ("scaled", "hands"), ("absolute", "hands")):
        assert arm_response_profile("auto", mapping, mode) is SMOOTH
        assert arm_response_profile("responsive", mapping, mode) is RESPONSIVE


def moving_target_error(response, hz):
    dt = 1. / hz
    joints = torch.zeros(1, 6)
    rest = joints.clone()
    velocity = joints.clone()
    command = joints.clone()
    filtered = joints.clone()
    jac = torch.eye(6).unsqueeze(0)
    limits = torch.tensor([-3., 3.]).expand(1, 6, 2)
    for step in range(2 * hz):
        target = torch.zeros_like(joints)
        target[0, 0] = .2 * (step + 1) * dt
        filtered.lerp_(target, dt / (response.smoothing_s + dt))
        previous = velocity.clone()
        velocity, command = joint_servo_step(
            jac, filtered - joints, joints, rest, limits, velocity, command, dt, response)
        assert velocity.abs().max() <= response.max_velocity + 1e-5
        assert (velocity - previous).abs().max() <= response.max_acceleration * dt + 1e-5
        joints = command.clone()  # Ideal plant; no claim about real robot dynamics.
    return float((target - joints).norm())


@pytest.mark.parametrize("hz", [30, 60])
def test_responsive_reduces_moving_target_lag(hz):
    assert moving_target_error(RESPONSIVE, hz) < .4 * moving_target_error(SMOOTH, hz)


@pytest.mark.parametrize("response", [SMOOTH, RESPONSIVE])
@pytest.mark.parametrize("singular", [False, True])
def test_unreachable_target_remains_finite_and_bounded(response, singular):
    joints = torch.zeros(1, 7)
    rest, command, velocity = joints.clone(), joints.clone(), joints.clone()
    jac = torch.zeros(1, 6, 7) if singular else torch.eye(6, 7).unsqueeze(0)
    limits = torch.tensor([-.07, .07]).expand(1, 7, 2)
    for _ in range(120):
        previous = velocity.clone()
        velocity, command = joint_servo_step(
            jac, torch.full((1, 6), 10.), joints, rest, limits, velocity, command, 1/60, response)
        assert torch.isfinite(command).all()
        assert velocity.abs().max() <= response.max_velocity + 1e-5
        assert (velocity - previous).abs().max() <= response.max_acceleration / 60 + 1e-5
        assert command.abs().max() <= .07 + 1e-5
        assert (command - joints).abs().max() <= .10 + 1e-5
