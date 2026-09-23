"""Pure reward composition from normalized progress and one-step events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .weights import MultiBoxRewardWeights


@dataclass(frozen=True)
class RewardBreakdown:
    terms: Mapping[str, torch.Tensor]
    total: torch.Tensor


@dataclass(frozen=True)
class CommonRewardInput:
    robot_rack_collision_event: torch.Tensor
    self_collision_event: torch.Tensor
    box_drop_event: torch.Tensor
    obstacle_collision_event: torch.Tensor
    workspace_limit_event: torch.Tensor
    normalized_base_motion: torch.Tensor
    normalized_action_rate: torch.Tensor
    normalized_joint_limit: torch.Tensor


@dataclass(frozen=True)
class GraspRewardInput:
    previous_approach: torch.Tensor
    approach: torch.Tensor
    previous_alignment: torch.Tensor
    alignment: torch.Tensor
    previous_capture: torch.Tensor
    capture: torch.Tensor
    previous_proof_lift: torch.Tensor
    proof_lift: torch.Tensor
    one_hand_pinch_event: torch.Tensor
    bilateral_pinch_event: torch.Tensor
    success_event: torch.Tensor
    common: CommonRewardInput


@dataclass(frozen=True)
class CarryRewardInput:
    previous_extraction: torch.Tensor
    extraction: torch.Tensor
    previous_belt: torch.Tensor
    belt: torch.Tensor
    previous_free_space: torch.Tensor
    free_space: torch.Tensor
    previous_pre_place_height: torch.Tensor
    pre_place_height: torch.Tensor
    grasp_loss_event: torch.Tensor
    placed_box_disturbance_event: torch.Tensor
    success_event: torch.Tensor
    common: CommonRewardInput


@dataclass(frozen=True)
class PlaceRewardInput:
    previous_footprint: torch.Tensor
    footprint: torch.Tensor
    previous_alignment: torch.Tensor
    alignment: torch.Tensor
    previous_free_space: torch.Tensor
    free_space: torch.Tensor
    previous_descent: torch.Tensor
    descent: torch.Tensor
    previous_stability: torch.Tensor
    stability: torch.Tensor
    support_event: torch.Tensor
    correct_release_event: torch.Tensor
    premature_release_event: torch.Tensor
    success_event: torch.Tensor
    common: CommonRewardInput


@dataclass(frozen=True)
class HighLevelRewardInput:
    first_placement_count: torch.Tensor
    displacement_count: torch.Tensor
    full_success_event: torch.Tensor
    failure_event: torch.Tensor
    elapsed_seconds: torch.Tensor


def potential_progress(previous: torch.Tensor, current: torch.Tensor, discount: float) -> torch.Tensor:
    """Potential-based shaping: gamma * Phi(s_next) - Phi(s)."""
    if previous.shape != current.shape or not previous.is_floating_point() \
            or not current.is_floating_point():
        raise ValueError("Potential pairs must be floating point tensors with equal shape.")
    return discount * current - previous


def _event(value: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.bool:
        raise TypeError("Reward events must be boolean one-step pulses.")
    return value.to(torch.float32)


def _sum(terms: dict[str, torch.Tensor]) -> RewardBreakdown:
    values = list(terms.values())
    if not values:
        raise ValueError("Reward needs at least one term.")
    total = torch.stack(values, dim=0).sum(dim=0)
    return RewardBreakdown(terms, total)


class MultiBoxRewardModel:
    def __init__(self, weights=None):
        self.weights = weights or MultiBoxRewardWeights()
        self.weights.validate()

    def _common(self, value: CommonRewardInput) -> dict[str, torch.Tensor]:
        w = self.weights.common
        return {
            "robot_rack_collision": -w.robot_rack_collision * _event(value.robot_rack_collision_event),
            "self_collision": -w.self_collision * _event(value.self_collision_event),
            "box_drop": -w.box_drop * _event(value.box_drop_event),
            "obstacle_collision": -w.obstacle_collision * _event(value.obstacle_collision_event),
            "workspace_limit": -w.workspace_limit * _event(value.workspace_limit_event),
            "base_motion": -w.base_motion * value.normalized_base_motion,
            "action_rate": -w.action_rate * value.normalized_action_rate,
            "joint_limit": -w.joint_limit * value.normalized_joint_limit,
        }

    def grasp(self, value: GraspRewardInput) -> RewardBreakdown:
        w, gamma = self.weights.grasp, self.weights.discount
        terms = self._common(value.common)
        terms["base_motion"] = -w.base_motion * value.common.normalized_base_motion
        terms["action_rate"] = -w.action_rate * value.common.normalized_action_rate
        terms.update({
            "approach_progress": w.approach_progress * potential_progress(
                value.previous_approach, value.approach, gamma),
            "alignment_progress": w.alignment_progress * potential_progress(
                value.previous_alignment, value.alignment, gamma),
            "capture_progress": w.capture_progress * potential_progress(
                value.previous_capture, value.capture, gamma),
            "proof_lift_progress": w.proof_lift_progress * potential_progress(
                value.previous_proof_lift, value.proof_lift, gamma),
            "one_hand_pinch_event": w.one_hand_pinch_event * _event(value.one_hand_pinch_event),
            "bilateral_pinch_event": w.bilateral_pinch_event * _event(value.bilateral_pinch_event),
            "success_event": w.success_event * _event(value.success_event),
        })
        return _sum(terms)

    def carry(self, value: CarryRewardInput) -> RewardBreakdown:
        w, gamma = self.weights.carry, self.weights.discount
        terms = self._common(value.common)
        terms.update({
            "extraction_progress": w.extraction_progress * potential_progress(
                value.previous_extraction, value.extraction, gamma),
            "belt_progress": w.belt_progress * potential_progress(
                value.previous_belt, value.belt, gamma),
            "free_space_progress": w.free_space_progress * potential_progress(
                value.previous_free_space, value.free_space, gamma),
            "pre_place_height_progress": w.pre_place_height_progress * potential_progress(
                value.previous_pre_place_height, value.pre_place_height, gamma),
            "grasp_loss_event": -w.grasp_loss_event * _event(value.grasp_loss_event),
            "placed_box_disturbance_event": -w.placed_box_disturbance_event * _event(
                value.placed_box_disturbance_event),
            "success_event": w.success_event * _event(value.success_event),
        })
        return _sum(terms)

    def place(self, value: PlaceRewardInput) -> RewardBreakdown:
        w, gamma = self.weights.place, self.weights.discount
        terms = self._common(value.common)
        terms.update({
            "footprint_progress": w.footprint_progress * potential_progress(
                value.previous_footprint, value.footprint, gamma),
            "alignment_progress": w.alignment_progress * potential_progress(
                value.previous_alignment, value.alignment, gamma),
            "free_space_progress": w.free_space_progress * potential_progress(
                value.previous_free_space, value.free_space, gamma),
            "descent_progress": w.descent_progress * potential_progress(
                value.previous_descent, value.descent, gamma),
            "stability_progress": w.stability_progress * potential_progress(
                value.previous_stability, value.stability, gamma),
            "support_event": w.support_event * _event(value.support_event),
            "correct_release_event": w.correct_release_event * _event(value.correct_release_event),
            "premature_release_event": -w.premature_release_event * _event(
                value.premature_release_event),
            "success_event": w.success_event * _event(value.success_event),
        })
        return _sum(terms)

    def high_level(self, value: HighLevelRewardInput) -> RewardBreakdown:
        w = self.weights.high_level
        return _sum({
            "first_placement": w.first_placement * value.first_placement_count,
            "displacement": -w.displacement * value.displacement_count,
            "full_success_event": w.full_success_event * _event(value.full_success_event),
            "failure_event": -w.failure_event * _event(value.failure_event),
            "elapsed_time": -w.elapsed_second * value.elapsed_seconds,
        })
