"""Initial normalization from SI-unit task metrics to [0, 1] potentials.

The scales are measurement starting points, not reward weights.  VR shadow
logs should report both raw values and these potentials so the scales can be
revised from real distributions without changing task-success predicates.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import torch

from ..geometry.grasp import GRASP_APPROACH_REWARD_SCALE_M, opposing_flap_reach_assignment


@dataclass(frozen=True)
class MetricScaleConfig:
    grasp_approach_m: float = 0.15
    grasp_alignment_rad: float = math.radians(20.0)
    grasp_capture_m: float = 0.015
    proof_lift_m: float = 0.008
    extraction_remaining_m: float = 0.10
    belt_distance_m: float = 0.30
    free_space_m: float = 0.10
    pre_place_min_m: float = 0.05
    pre_place_max_m: float = 0.15
    pre_place_falloff_m: float = 0.05
    footprint_outside_m: float = 0.05
    place_alignment_rad: float = math.radians(10.0)
    descent_height_m: float = 0.05
    stable_linear_mps: float = 0.05
    stable_angular_radps: float = 0.20

    def validate(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Metric scale {field.name} must be finite and positive.")
        if self.pre_place_min_m >= self.pre_place_max_m:
            raise ValueError("Pre-place height interval must be ordered.")


@dataclass(frozen=True)
class GraspRawMetrics:
    matched_flap_distance_m: torch.Tensor
    jaw_alignment_error_rad: torch.Tensor
    capture_error_m: torch.Tensor
    proof_lift_m: torch.Tensor


@dataclass(frozen=True)
class CarryRawMetrics:
    extraction_remaining_m: torch.Tensor
    footprint_distance_to_belt_m: torch.Tensor
    free_space_clearance_m: torch.Tensor
    box_bottom_height_m: torch.Tensor


@dataclass(frozen=True)
class PlaceRawMetrics:
    footprint_outside_m: torch.Tensor
    long_axis_error_rad: torch.Tensor
    free_space_clearance_m: torch.Tensor
    box_bottom_height_m: torch.Tensor
    linear_speed_mps: torch.Tensor
    angular_speed_radps: torch.Tensor


def _same_shape_finite(values) -> None:
    tensors = list(values)
    if not tensors or any(not isinstance(value, torch.Tensor) or not value.is_floating_point()
                          for value in tensors):
        raise TypeError("Raw metrics must be floating point tensors.")
    if len({value.shape for value in tensors}) != 1:
        raise ValueError("Raw metrics in one skill must share a batch shape.")
    if len({value.device for value in tensors}) != 1:
        raise ValueError("Raw metrics in one skill must share a device.")
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise ValueError("Raw metrics must be finite.")


def _exp_error(error: torch.Tensor, scale: float) -> torch.Tensor:
    return torch.exp(-error.abs() / scale)


def _interval(value: torch.Tensor, low: float, high: float, falloff: float) -> torch.Tensor:
    outside = (low - value).clamp_min(0) + (value - high).clamp_min(0)
    return torch.exp(-outside / falloff)


def grasp_potentials(raw: GraspRawMetrics, config=None) -> dict[str, torch.Tensor]:
    config = config or MetricScaleConfig()
    config.validate()
    _same_shape_finite(getattr(raw, field.name) for field in fields(raw))
    return {
        "approach": _exp_error(raw.matched_flap_distance_m, config.grasp_approach_m),
        "alignment": _exp_error(raw.jaw_alignment_error_rad, config.grasp_alignment_rad),
        "capture": _exp_error(raw.capture_error_m, config.grasp_capture_m),
        "proof_lift": (raw.proof_lift_m / config.proof_lift_m).clamp(0, 1),
    }


def grasp_reward_potentials(
    candidate_distance_m: torch.Tensor,
    matched_distance_m: torch.Tensor,
    jaw_alignment_cos: torch.Tensor,
    capture_error_m: torch.Tensor,
    jaw_gap_m: torch.Tensor,
    flap_thickness_m: torch.Tensor,
    proof_lift_m: torch.Tensor,
    *,
    approach_scale_m: float = GRASP_APPROACH_REWARD_SCALE_M,
    capture_scale_m: float = 0.10,
) -> dict[str, torch.Tensor]:
    """V2 grasp shaping: reward both hands reaching different flaps.

    The gap score uses calibrated finger tips.  Its preparation factor keeps
    closing useful only after a hand approaches and straddles its flap.
    """
    n = proof_lift_m.shape[0]
    if candidate_distance_m.shape != (n, 2, 2) or any(
        value.shape != (n, 2) for value in (
            matched_distance_m, jaw_alignment_cos, capture_error_m,
            jaw_gap_m, flap_thickness_m,
        )
    ):
        raise ValueError("Grasp reward geometry needs [env, hand, flap] and [env, hand] tensors")
    if approach_scale_m <= 0 or capture_scale_m <= 0:
        raise ValueError("Grasp reward distance scales must be positive")
    approach, _ = opposing_flap_reach_assignment(candidate_distance_m, approach_scale_m)
    alignment = jaw_alignment_cos.clamp(0, 1).square()
    near = (1.0 - matched_distance_m.clamp_min(0) / 0.10).clamp(0, 1)
    capture = torch.exp(-capture_error_m.clamp_min(0) / capture_scale_m)
    preparation = (
        torch.exp(-matched_distance_m.clamp_min(0) / 0.10)
        * alignment * torch.exp(-capture_error_m.clamp_min(0) / 0.02)
    )
    open_clearance = 0.04
    desired_gap = flap_thickness_m + (1.0 - preparation) * open_clearance
    gap_score = preparation * torch.exp(-(jaw_gap_m - desired_gap).abs() / 0.02)
    premature_close = near * (1.0 - preparation) * (
        (flap_thickness_m + open_clearance - jaw_gap_m) / open_clearance
    ).clamp(0, 1)
    return {
        "approach": approach,
        "alignment": alignment.mean(-1),
        "alignment_proximity": near.mean(-1),
        "capture": capture.mean(-1),
        "jaw_gap": gap_score.mean(-1),
        "premature_close": premature_close.mean(-1),
        "proof_lift": (proof_lift_m / 0.008).clamp(0, 1),
    }


def grasp_gated_lift_inputs(
    current: torch.Tensor,
    previous: torch.Tensor,
    eligible: torch.Tensor,
    was_eligible: torch.Tensor,
    discount: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Only continuous opposing-flap pinches can earn box-lift progress."""
    if (current.shape != previous.shape or current.shape != eligible.shape
            or current.shape != was_eligible.shape or eligible.dtype != torch.bool
            or was_eligible.dtype != torch.bool):
        raise ValueError("Lift inputs need matching scalar batches and boolean pinch masks")
    active = torch.where(eligible, current, torch.zeros_like(current))
    baseline = torch.where(
        eligible & was_eligible, previous, discount * active)
    return baseline, active


def carry_potentials(raw: CarryRawMetrics, config=None) -> dict[str, torch.Tensor]:
    config = config or MetricScaleConfig()
    config.validate()
    _same_shape_finite(getattr(raw, field.name) for field in fields(raw))
    return {
        "extraction": _exp_error(raw.extraction_remaining_m.clamp_min(0),
                                 config.extraction_remaining_m),
        "belt": _exp_error(raw.footprint_distance_to_belt_m.clamp_min(0), config.belt_distance_m),
        "free_space": (raw.free_space_clearance_m / config.free_space_m).clamp(0, 1),
        "pre_place_height": _interval(raw.box_bottom_height_m,
                                      config.pre_place_min_m, config.pre_place_max_m,
                                      config.pre_place_falloff_m),
    }


def place_potentials(raw: PlaceRawMetrics, config=None) -> dict[str, torch.Tensor]:
    config = config or MetricScaleConfig()
    config.validate()
    _same_shape_finite(getattr(raw, field.name) for field in fields(raw))
    footprint = _exp_error(raw.footprint_outside_m.clamp_min(0), config.footprint_outside_m)
    alignment = _exp_error(raw.long_axis_error_rad, config.place_alignment_rad)
    free_space = (raw.free_space_clearance_m / config.free_space_m).clamp(0, 1)
    descent = _exp_error(raw.box_bottom_height_m.clamp_min(0), config.descent_height_m)
    # Lowering outside the belt or on top of another box must not look like
    # useful descent, so gate it continuously rather than with a hard phase bit.
    descent = descent * footprint * alignment * free_space
    stability = _exp_error(raw.linear_speed_mps.clamp_min(0), config.stable_linear_mps) \
        * _exp_error(raw.angular_speed_radps.clamp_min(0), config.stable_angular_radps)
    return {
        "footprint": footprint,
        "alignment": alignment,
        "free_space": free_space,
        "descent": descent,
        "stability": stability,
    }
