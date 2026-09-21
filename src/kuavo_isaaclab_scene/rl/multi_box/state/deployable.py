"""Assemble policy state exclusively from replaceable deployable inputs."""

from __future__ import annotations

import torch

from ..hierarchy import SkillControlState
from .perception import PerceptionFrame
from .placement import DeployablePlacementEstimateState
from .schema import DeployableRobotState, DeployableTaskState


def assemble_deployable_task_state(
    *, perception: PerceptionFrame, robot: DeployableRobotState,
    placement: DeployablePlacementEstimateState, control: SkillControlState,
    transition_confidence: torch.Tensor,
) -> DeployableTaskState:
    """Join perception, telemetry, and skill state without privileged truth."""
    if not isinstance(placement, DeployablePlacementEstimateState):
        raise TypeError("Actor state requires a pose-based placement estimate.")
    state = DeployableTaskState(
        boxes=perception.boxes, robot=robot,
        rack_pose_world=perception.rack_pose_world,
        conveyor_pose_world=perception.conveyor_pose_world,
        placement=placement, control=control,
        transition_confidence=transition_confidence,
    )
    state.validate(len(perception.boxes.active))
    return state
