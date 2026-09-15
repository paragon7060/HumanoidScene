"""Bounded normal-force feedback for the donor closed four-bar claw.

Force is the sum of the two jaw normal forces, not force per jaw or wrist
wrench. Free closing and opening use that same force in opposite directions.
"""

from functools import lru_cache
import math

import numpy as np
import torch


@lru_cache(maxsize=2)
def jaw_leverage_table(side):
    from .twofinger_geometry import TwoFingerGeometry
    from .twofinger_linkage import DRIVER_OPEN_MIN

    geometry = TwoFingerGeometry(side)
    angles = np.linspace(DRIVER_OPEN_MIN, 0., 257)
    gaps = np.array([geometry.gap_m(float(q)) for q in angles])
    # Both symmetric jaws contribute to gap change; each driver moves one jaw.
    lever = -np.gradient(gaps, angles) * .5
    if not np.isfinite(lever).all() or (lever <= 0).any():
        raise ValueError("Invalid closed-linkage jaw leverage")
    return angles, lever


class JawForceServo:
    """PI trim around geometric force feedforward, without free-space windup."""

    def __init__(self, side, num_envs, device, force_n, directions, max_velocity=.5):
        if not math.isfinite(force_n) or force_n <= 0:
            raise ValueError("Gripper close force must be positive and finite")
        angles, lever = jaw_leverage_table(side)
        self.angles = torch.as_tensor(angles, device=device, dtype=torch.float32)
        self.levers = torch.as_tensor(lever, device=device, dtype=torch.float32)
        self.directions = directions
        self.force_n = float(force_n)
        self.per_jaw_n = self.force_n * .5
        self.max_velocity = float(max_velocity)
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
        alpha = -math.expm1(-dt / .02)
        self.filtered_force.lerp_(force, alpha)
        self.elapsed.add_(dt)
        target = self.per_jaw_n * (self.elapsed / .3).clamp(0, 1)
        error = target - self.filtered_force
        contact = force > .1
        # Absence of contact is not a force error that should accumulate.
        self.integral_n[:] = torch.where(contact & closing & self.valid,
            (self.integral_n + 4. * error * dt).clamp(-4 * self.per_jaw_n, 4 * self.per_jaw_n), 0.)
        requested_force = torch.where(closing,
            (target + .25 * error * contact + self.integral_n).clamp(0, 5 * self.per_jaw_n), target)
        driver = (q * self.directions).clamp(self.angles[0], self.angles[-1])
        idx = torch.searchsorted(self.angles, driver.contiguous()).clamp(1, len(self.angles) - 1)
        weight = (driver - self.angles[idx - 1]) / (self.angles[idx] - self.angles[idx - 1])
        leverage = torch.lerp(self.levers[idx - 1], self.levers[idx], weight)
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
