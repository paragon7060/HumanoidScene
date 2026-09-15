"""Simulator-independent scalar gripper action conversions."""

from __future__ import annotations

import math
import torch


class GripperTargetFilter:
    """Physics-time one/two-stage target filter with direction-specific taus."""

    def __init__(self, initial_targets, close_direction, settings):
        self.current = initial_targets.clone()
        self.intermediate = initial_targets.clone()
        self.stages = settings.get("stages", 1)
        if type(self.stages) is not int or self.stages not in (1, 2):
            raise ValueError("Gripper target filter stages must be 1 or 2")
        self.direction = close_direction
        self.closing_tau = float(settings["closing_time_constant_s"])
        self.opening_tau = float(settings["opening_time_constant_s"])
        if any(not math.isfinite(tau) or tau <= 0 for tau in (self.closing_tau, self.opening_tau)):
            raise ValueError("Gripper target filter time constants must be finite and positive")
        self._last_dt = None

    def advance(self, desired, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("Physics dt must be finite and positive")
        if dt != self._last_dt:
            self._closing_alpha = self.current.new_tensor(-math.expm1(-dt / self.closing_tau))
            self._opening_alpha = self.current.new_tensor(-math.expm1(-dt / self.opening_tau))
            self._closing_gain = self.current.new_tensor(dt / self.closing_tau * math.exp(-dt / self.closing_tau))
            self._opening_gain = self.current.new_tensor(dt / self.opening_tau * math.exp(-dt / self.opening_tau))
            self._last_dt = dt
        delta = desired - self.current
        closing = (delta * self.direction).sum(-1, keepdim=True) >= 0
        alpha = torch.where(closing, self._closing_alpha, self._opening_alpha)
        if self.stages == 1:
            self.current.add_(alpha * delta)
        else:
            # On reversal, discard an intermediate target that would keep
            # driving away from the new request. Reset uses the same rule.
            stage = self.intermediate.clamp(torch.minimum(self.current, desired),
                                             torch.maximum(self.current, desired))
            gain = torch.where(closing, self._closing_gain, self._opening_gain)
            # Exact zero-order-hold solution of two equal first-order stages.
            # It starts gently without a transport-delay queue and is invariant
            # to control decimation for a held desired target.
            self.current.lerp_(desired, alpha)
            self.current.addcmul_(stage-desired, gain)
            torch.lerp(stage, desired, alpha, out=self.intermediate)
        return self.current

    def reset(self, targets, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.current[ids] = targets.to(self.current)
        self.intermediate[ids] = targets.to(self.current)


class DirectionalGripperMapping:
    """Measured command envelopes with continuous holds at direction reversals.

    Fractions describe joint travel, not physical width. A reversal holds the
    last target until the opposite envelope reaches it. Minor loops have not
    been fully validated; this rule avoids jumps opposite to requested motion.
    """

    def __init__(self, mapping, num_envs, device):
        self.percent = torch.tensor(mapping["command_percent"], device=device)
        self.closing = torch.tensor(mapping["closing_fractions"], device=device)
        self.opening = torch.tensor(mapping["opening_fractions"], device=device)
        self.previous_percent = torch.zeros(num_envs, 1, device=device)
        self.fraction = torch.zeros_like(self.previous_percent)

    @staticmethod
    def _interpolate(value, x, y):
        index = torch.searchsorted(x, value.contiguous(), right=True).clamp(1, len(x) - 1)
        weight = ((value - x[index - 1]) / (x[index] - x[index - 1]).clamp_min(1e-8)).clamp(0, 1)
        return y[index - 1] + weight * (y[index] - y[index - 1])

    def process(self, signed_action):
        percent = ((1 - signed_action.float()) * 50).clamp(0, 100)
        closing = self._interpolate(percent, self.percent, self.closing)
        opening = self._interpolate(percent, self.percent, self.opening)
        self.fraction[:] = torch.where(percent > self.previous_percent,
            torch.maximum(self.fraction, closing), torch.where(percent < self.previous_percent,
            torch.minimum(self.fraction, opening), self.fraction))
        self.previous_percent[:] = percent
        return self.fraction

    def reset(self, env_ids=None, fraction=None):
        ids = slice(None) if env_ids is None else env_ids
        if fraction is None:
            self.fraction[ids] = 0
            self.previous_percent[ids] = 0
        else:
            self.fraction[ids] = fraction.reshape(-1, 1).clamp(0, 1)
            self.previous_percent[ids] = self._interpolate(
                self.fraction[ids], self.closing, self.percent)


def interpolate_signed_gripper_action(
    actions: torch.Tensor,
    open_command: torch.Tensor,
    close_command: torch.Tensor,
) -> torch.Tensor:
    """Interpolate hand targets while retaining the legacy signed convention.

    ``+1`` is fully open, ``-1`` is fully closed, and values in between retain
    the continuous claw fraction emitted by LeRobot policies.
    """
    close_fraction = ((1.0 - actions.float()) * 0.5).clamp(0.0, 1.0)
    return open_command + close_fraction * (close_command - open_command)
