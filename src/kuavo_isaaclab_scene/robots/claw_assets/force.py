"""Normal-force control for the donor closed four-bar claw.

Force is the sum of the two jaw normal forces, not force per jaw or wrist
wrench. RL uses the lightweight geometric feedforward below; the optional VR
diagnostic path retains measured contact-force feedback.
"""

from functools import lru_cache
import math

import numpy as np
import torch

from .package import load_claw_config


def force_control_settings():
    """Return package-owned force-servo tuning shared by RL and VR."""
    return load_claw_config()["force_control"]


@lru_cache(maxsize=2)
def jaw_leverage_table(side):
    from .geometry import TwoFingerGeometry
    from .linkage import DRIVER_OPEN_MIN

    geometry = TwoFingerGeometry(side)
    angles = np.linspace(DRIVER_OPEN_MIN, 0., 257)
    gaps = np.array([geometry.gap_m(float(q)) for q in angles])
    # Both symmetric jaws contribute to gap change; each driver moves one jaw.
    lever = -np.gradient(gaps, angles) * .5
    if not np.isfinite(lever).all() or (lever <= 0).any():
        raise ValueError("Invalid closed-linkage jaw leverage")
    return angles, lever


def _jaw_leverage(q, directions, angles, levers):
    """Interpolate each jaw's linear leverage at its current driver angle."""
    driver = (q * directions).clamp(angles[0], angles[-1])
    index = torch.searchsorted(angles, driver.contiguous()).clamp(1, len(angles) - 1)
    weight = (driver - angles[index - 1]) / (angles[index] - angles[index - 1])
    return torch.lerp(levers[index - 1], levers[index], weight)


class JawForceFeedforward:
    """Convert a requested total squeeze into per-jaw closing torque.

    This is deliberately sensor-free. It adds the torque corresponding to
    ``force_n / 2`` at each jaw while the regular position PD remains active.
    The result is a commanded force equivalent; measured contact force can
    still vary with contact geometry, friction and solver compliance.
    """

    def __init__(self, side, num_envs, device, force_n, directions):
        if not math.isfinite(force_n) or force_n <= 0:
            raise ValueError("Gripper close force must be positive and finite")
        angles, leverage = jaw_leverage_table(side)
        self.angles = torch.as_tensor(angles, device=device, dtype=torch.float32)
        self.levers = torch.as_tensor(leverage, device=device, dtype=torch.float32)
        self.directions = directions
        self.force_n = float(force_n)
        self.per_jaw_n = self.force_n * .5
        self.torque_request = torch.zeros((num_envs, 2), device=device)

    def reset(self, env_ids=None):
        self.torque_request[slice(None) if env_ids is None else env_ids] = 0

    def advance(self, q, closing, effort_limits):
        leverage = _jaw_leverage(q, self.directions, self.angles, self.levers)
        torque = self.per_jaw_n * leverage * self.directions
        self.torque_request[:] = torch.where(closing, torque, 0.).clamp(
            -effort_limits, effort_limits)
        # With no object between the jaws there is no squeeze force to hold.
        # Avoid continuously loading the authored mechanical stop.
        at_closed_stop = (q * self.directions) >= -1e-4
        self.torque_request.masked_fill_(closing & at_closed_stop, 0)
        return self.torque_request


class JawForceServo:
    """PI trim around geometric force feedforward, without free-space windup."""

    def __init__(self, side, num_envs, device, force_n, directions, max_velocity=None):
        if not math.isfinite(force_n) or force_n <= 0:
            raise ValueError("Gripper close force must be positive and finite")
        angles, lever = jaw_leverage_table(side)
        self.angles = torch.as_tensor(angles, device=device, dtype=torch.float32)
        self.levers = torch.as_tensor(lever, device=device, dtype=torch.float32)
        self.directions = directions
        self.force_n = float(force_n)
        self.per_jaw_n = self.force_n * .5
        tuning = force_control_settings()
        self.max_velocity = float(
            tuning["max_velocity_rad_s"] if max_velocity is None else max_velocity)
        self.filter_time_constant = float(tuning["force_filter_time_constant_s"])
        self.ramp_time = float(tuning["force_ramp_time_s"])
        self.proportional_gain = float(tuning["proportional_gain"])
        self.integral_gain = float(tuning["integral_gain_per_s"])
        self.integral_limit = float(tuning["integral_limit_multiplier"])
        self.max_force = float(tuning["max_force_multiplier"])
        if min(self.max_velocity, self.filter_time_constant, self.ramp_time,
               self.integral_limit, self.max_force) <= 0:
            raise ValueError("Force-control package tuning must be positive")
        # Implicit viscosity limits free closing speed; stiffness is zero.
        # Two times nominal torque leaves room for contact-feedback correction.
        self.damping = 2 * float(lever.max()) * self.per_jaw_n / self.max_velocity
        shape = (num_envs, 2)
        self.filtered_force = torch.zeros(shape, device=device)
        self.integral_n = torch.zeros(shape, device=device)
        self.elapsed = torch.zeros(num_envs, 1, device=device)
        self.active = torch.zeros(num_envs, 1, dtype=torch.bool, device=device)
        self.torque_request = torch.zeros(shape, device=device)
        self.valid = torch.ones(num_envs, 1, dtype=torch.bool, device=device)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        for field in (self.filtered_force, self.integral_n, self.elapsed, self.torque_request):
            field[ids] = 0
        self.active[ids] = False
        self.valid[ids] = True

    def advance(self, q, measured_force, closing, dt, effort_limits, open_command=None):
        changed = closing != self.active
        self.integral_n.masked_fill_(changed.expand_as(q), 0)
        self.elapsed.masked_fill_(changed, 0)
        self.filtered_force.masked_fill_(changed.expand_as(q), 0)
        self.active[:] = closing
        self.valid[:] = torch.isfinite(measured_force).all(-1, keepdim=True)
        force = torch.nan_to_num(measured_force, nan=0., posinf=0., neginf=0.).clamp_min(0)
        alpha = -math.expm1(-dt / self.filter_time_constant)
        self.filtered_force.lerp_(force, alpha)
        self.elapsed.add_(dt)
        target = self.per_jaw_n * (self.elapsed / self.ramp_time).clamp(0, 1)
        error = target - self.filtered_force
        contact = force > .1
        # Absence of contact is not a force error that should accumulate.
        self.integral_n[:] = torch.where(contact & closing & self.valid,
            (self.integral_n + self.integral_gain * error * dt).clamp(
                -self.integral_limit * self.per_jaw_n,
                self.integral_limit * self.per_jaw_n), 0.)
        requested_force = torch.where(closing,
            (target + self.proportional_gain * error * contact + self.integral_n).clamp(
                0, self.max_force * self.per_jaw_n), target)
        leverage = _jaw_leverage(q, self.directions, self.angles, self.levers)
        sign = torch.where(closing, 1., -1.)
        self.torque_request[:] = (requested_force * leverage * self.directions * sign).clamp(-effort_limits, effort_limits)
        self.torque_request *= self.valid
        # The logical closed pose is also the mechanism stop. Do not keep
        # driving into it when no object can receive the requested force.
        signed_q = q * self.directions
        open_limit = self.angles[0] if open_command is None else open_command * self.directions
        at_stop = torch.where(closing, signed_q >= -.001, signed_q <= open_limit + .001)
        self.torque_request.masked_fill_(at_stop, 0)
        self.integral_n.masked_fill_(at_stop, 0)
        velocity = (self.torque_request / self.damping).clamp(-self.max_velocity, self.max_velocity)
        return velocity
