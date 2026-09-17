"""Initial v2 reward ratios.

Potential inputs are normalized to [0, 1].  Event inputs are one-step pulses.
The ratios make terminal skill success larger than all positive shaping terms
in that skill, while destructive terminal events cost more than one success.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import math


@dataclass(frozen=True)
class CommonWeights:
    robot_rack_collision: float = 6.0
    self_collision: float = 4.0
    box_drop: float = 8.0
    obstacle_collision: float = 4.0
    base_motion: float = 0.002
    action_rate: float = 0.001
    joint_limit: float = 0.002


@dataclass(frozen=True)
class GraspWeights:
    approach_progress: float = 0.50
    alignment_progress: float = 0.30
    capture_progress: float = 0.30
    proof_lift_progress: float = 0.40
    bilateral_pinch_event: float = 1.00
    success_event: float = 3.00


@dataclass(frozen=True)
class CarryWeights:
    extraction_progress: float = 0.60
    belt_progress: float = 1.00
    free_space_progress: float = 0.50
    pre_place_height_progress: float = 0.40
    grasp_loss_event: float = 3.00
    placed_box_disturbance_event: float = 1.00
    success_event: float = 4.00


@dataclass(frozen=True)
class PlaceWeights:
    footprint_progress: float = 0.80
    alignment_progress: float = 0.80
    free_space_progress: float = 0.60
    descent_progress: float = 0.50
    stability_progress: float = 0.30
    support_event: float = 0.50
    correct_release_event: float = 1.00
    premature_release_event: float = 2.50
    success_event: float = 5.00


@dataclass(frozen=True)
class HighLevelWeights:
    first_placement: float = 5.00
    displacement: float = 3.00
    full_success_event: float = 12.00
    failure_event: float = 8.00
    elapsed_second: float = 0.01


@dataclass(frozen=True)
class MultiBoxRewardWeights:
    discount: float = 0.99
    common: CommonWeights = CommonWeights()
    grasp: GraspWeights = GraspWeights()
    carry: CarryWeights = CarryWeights()
    place: PlaceWeights = PlaceWeights()
    high_level: HighLevelWeights = HighLevelWeights()

    def validate(self) -> None:
        if not math.isfinite(self.discount) or not 0 < self.discount <= 1:
            raise ValueError("Reward discount must be in (0, 1].")
        for group in (self.common, self.grasp, self.carry, self.place, self.high_level):
            for field in fields(group):
                value = getattr(group, field.name)
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Reward weight {field.name} must be finite and nonnegative.")
        grasp_dense = sum((self.grasp.approach_progress, self.grasp.alignment_progress,
                           self.grasp.capture_progress, self.grasp.proof_lift_progress))
        carry_dense = sum((self.carry.extraction_progress, self.carry.belt_progress,
                           self.carry.free_space_progress, self.carry.pre_place_height_progress))
        place_dense = sum((self.place.footprint_progress, self.place.alignment_progress,
                           self.place.free_space_progress, self.place.descent_progress,
                           self.place.stability_progress))
        if self.grasp.success_event <= grasp_dense \
                or self.carry.success_event <= carry_dense \
                or self.place.success_event <= place_dense:
            raise ValueError("Each skill success weight must exceed its total dense shaping weight.")
        largest_success = max(self.grasp.success_event, self.carry.success_event,
                              self.place.success_event)
        if self.common.box_drop <= largest_success or self.high_level.failure_event <= largest_success:
            raise ValueError("Destructive terminal penalties must exceed one low-level success.")
