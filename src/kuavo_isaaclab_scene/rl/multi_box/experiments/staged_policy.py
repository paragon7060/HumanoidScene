"""Shared checkpoint dispatch for staged training and frozen video replay."""

from .staged_goal_sac import StagedGoalSACPilot
from .staged_hybrid_goal_sac import StagedHybridGoalSACPilot
from .physical_body_sac import PhysicalBodySACPilot


def staged_policy_class(artifact_type):
    return {cls.artifact_type: cls for cls in (
        StagedGoalSACPilot, StagedHybridGoalSACPilot, PhysicalBodySACPilot,
    )}.get(artifact_type)


def staged_policy_metadata(pilot):
    key = ('physical_body_contract' if isinstance(pilot, PhysicalBodySACPilot)
           else 'goal_contract')
    return {key: pilot.contract}
