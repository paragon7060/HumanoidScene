"""Privileged grasp-success predicate for the v2 two-arm task.

This module consumes contact/geometry measurements.  It does not calculate a
reward and does not import any legacy reward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


OPPOSING_FLAPS = ("flap_right", "flap_left")


@dataclass(frozen=True)
class GraspSuccessConfig:
    """Approved proof-lift criterion in SI units."""

    flap_names: tuple[str, str] = OPPOSING_FLAPS
    proof_lift_m: float = 0.008
    hold_seconds: float = 0.25

    def validate(self) -> None:
        if self.flap_names != OPPOSING_FLAPS:
            raise ValueError(
                "V2 dual-arm grasp requires the opposing flap_right/flap_left pair."
            )
        if not math.isfinite(self.proof_lift_m) or not 0.005 <= self.proof_lift_m <= 0.010:
            raise ValueError("Proof lift must remain inside the approved 5-10 mm range.")
        if not math.isfinite(self.hold_seconds) or not 0.20 <= self.hold_seconds <= 0.30:
            raise ValueError("Grasp hold must remain inside the approved 0.2-0.3 s range.")


@dataclass(frozen=True)
class GraspSuccessInput:
    """Privileged measurements, one row per vectorized environment.

    ``hand_pinching`` means each hand already passed the physical two-finger
    checks: both pads on one flap, opposed jaws, valid contact region, and the
    configured minimum contact force.  ``relative_pose_stable`` is produced by
    the future state provider; its numerical tolerance remains a state-design
    decision and is not hidden in this predicate.
    """

    hand_pinching: torch.Tensor
    hand_flap_index: torch.Tensor
    relative_pose_stable: torch.Tensor
    rack_clearance_m: torch.Tensor

    def validate(self) -> None:
        if self.hand_pinching.ndim != 2 or self.hand_pinching.shape[1] != 2:
            raise ValueError("hand_pinching must have shape [num_envs, 2].")
        expected = self.hand_pinching.shape
        if self.hand_pinching.dtype != torch.bool:
            raise TypeError("hand_pinching must be boolean.")
        if self.hand_flap_index.shape != expected or self.hand_flap_index.dtype != torch.long:
            raise ValueError("hand_flap_index must be torch.long [num_envs, 2].")
        if self.relative_pose_stable.shape != expected or self.relative_pose_stable.dtype != torch.bool:
            raise ValueError("relative_pose_stable must be boolean [num_envs, 2].")
        if self.rack_clearance_m.shape != expected[:1] or not self.rack_clearance_m.is_floating_point():
            raise ValueError("rack_clearance_m must be floating point [num_envs].")
        devices = {
            self.hand_pinching.device,
            self.hand_flap_index.device,
            self.relative_pose_stable.device,
            self.rack_clearance_m.device,
        }
        if len(devices) != 1:
            raise ValueError("All grasp-success measurements must share one device.")


@dataclass(frozen=True)
class GraspSuccessResult:
    bilateral_pinch: torch.Tensor
    opposing_flaps: torch.Tensor
    stable: torch.Tensor
    proof_lift: torch.Tensor
    instantaneous: torch.Tensor
    hold_time_s: torch.Tensor
    success: torch.Tensor


class GraspSuccessTracker:
    """Accumulates continuous proof-lift time and resets immediately on loss."""

    def __init__(self, num_envs: int, device: str | torch.device, config=None):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.config = config or GraspSuccessConfig()
        self.config.validate()
        self.hold_time_s = torch.zeros(num_envs, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.hold_time_s.zero_()
        else:
            self.hold_time_s[env_ids] = 0.0

    def update(self, measurements: GraspSuccessInput, dt: float) -> GraspSuccessResult:
        measurements.validate()
        if len(measurements.rack_clearance_m) != len(self.hold_time_s):
            raise ValueError("Grasp-success batch size does not match the tracker.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")

        bilateral = measurements.hand_pinching.all(dim=-1)
        # The candidate list is exactly (right, left), so distinct valid IDs
        # mean that the two hands hold the mutually opposing flap pair.
        valid_ids = ((measurements.hand_flap_index >= 0)
                     & (measurements.hand_flap_index < len(self.config.flap_names))).all(dim=-1)
        opposing = valid_ids & (
            measurements.hand_flap_index[:, 0] != measurements.hand_flap_index[:, 1])
        stable = measurements.relative_pose_stable.all(dim=-1)
        proof_lift = measurements.rack_clearance_m >= self.config.proof_lift_m
        instantaneous = bilateral & opposing & stable & proof_lift
        self.hold_time_s = torch.where(
            instantaneous, self.hold_time_s + dt, torch.zeros_like(self.hold_time_s))
        success = self.hold_time_s >= self.config.hold_seconds
        return GraspSuccessResult(
            bilateral, opposing, stable, proof_lift, instantaneous,
            self.hold_time_s.clone(), success,
        )
