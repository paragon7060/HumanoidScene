"""Single-step v2 orchestration for replaceable sensing and box selection.

This coordinates deployable state only. Privileged success, rewards, collision
termination, and policy training remain separate providers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .hierarchy.state_machine import NO_TARGET, SkillStateMachine
from .hierarchy.transition_estimator import (
    DeployableTransitionConfig, DeployableTransitionEstimate,
    DeployableTransitionEstimator,
)
from .hierarchy.types import HighLevelSelection
from .spec import MAX_BOXES
from .observations import build_actor_observation
from .observations.schema import ActorObservation
from .state.deployable import assemble_deployable_task_state
from .state.perception import PerceptionFrame
from .state.pose_placement import (
    PosePlacementEstimateTracker, PosePlacementEvidence,
)
from .state.schema import DeployableRobotState, DeployableTaskState


class PerceptionSource(Protocol):
    def read(self) -> PerceptionFrame: ...


class RobotTelemetrySource(Protocol):
    def read(self) -> DeployableRobotState: ...


class HighLevelBoxSelector(Protocol):
    def select(
        self, observation: ActorObservation, selectable_boxes: torch.Tensor,
        waiting_env_ids: torch.Tensor,
    ) -> HighLevelSelection: ...


class FirstSelectableBoxSelector:
    """Deterministic plumbing check; replace with the learned high-level policy."""

    def select(self, observation: ActorObservation, selectable_boxes: torch.Tensor,
               waiting_env_ids: torch.Tensor) -> HighLevelSelection:
        if selectable_boxes.ndim != 2 or selectable_boxes.shape[1] != MAX_BOXES \
                or selectable_boxes.dtype != torch.bool \
                or not bool(selectable_boxes.any(-1).all()):
            raise ValueError("Each waiting environment needs a selectable box.")
        return HighLevelSelection(selectable_boxes.to(torch.long).argmax(-1))


@dataclass(frozen=True)
class DeployableRuntimeStep:
    state: DeployableTaskState
    actor_observation: ActorObservation
    transition: DeployableTransitionEstimate
    placement_evidence: PosePlacementEvidence
    selected_box: torch.Tensor
    waiting_without_box: torch.Tensor


class MultiBoxDeployableRuntime:
    """Read sensors once, select a target, advance one skill, expose actor state."""

    def __init__(
        self, *, num_envs: int, device: str | torch.device,
        perception_source: PerceptionSource,
        robot_source: RobotTelemetrySource,
        selector: HighLevelBoxSelector | None = None,
    ):
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.perception_source = perception_source
        self.robot_source = robot_source
        self.selector = selector or FirstSelectableBoxSelector()
        self.machine = SkillStateMachine(num_envs, self.device)
        self.placement = PosePlacementEstimateTracker(num_envs, self.device)
        self.transition = DeployableTransitionEstimator(
            num_envs, self.device,
            config=DeployableTransitionConfig(
                place_hold_seconds=self.placement.config.hold_seconds),
        )
        self.previous_progress = torch.zeros(num_envs, 3, device=self.device)

    def reset(self, env_ids=None) -> None:
        self.machine.reset(env_ids)
        self.placement.reset(env_ids)
        self.transition.reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self.previous_progress[ids] = 0.0

    def step(self, *, previous_action: torch.Tensor, dt: float) -> DeployableRuntimeStep:
        perception = self.perception_source.read()
        robot = self.robot_source.read()
        placement, placement_evidence = self.placement.update(perception, robot, dt)
        selectable = placement.selectable
        waiting = self.machine.needs_target
        can_select = waiting & selectable.any(-1)
        waiting_ids = can_select.nonzero(as_tuple=False).flatten()
        selected_box = torch.full((self.num_envs,), NO_TARGET,
                                  dtype=torch.long, device=self.device)
        if len(waiting_ids):
            # The selector sees the same actor contract that a learned policy
            # will see; no privileged state enters this decision.
            before = assemble_deployable_task_state(
                perception=perception, robot=robot, placement=placement,
                control=self.machine.state(),
                transition_confidence=self.previous_progress)
            observation = build_actor_observation(before, previous_action)
            selection = self.selector.select(
                observation, selectable[waiting_ids], waiting_ids)
            self.machine.assign_subset(waiting_ids, selection, selectable)
            selected_box[waiting_ids] = selection.box_index

        transition = self.transition.update(
            perception=perception, robot=robot, placement=placement,
            placement_evidence=placement_evidence,
            control=self.machine.state(), dt=dt)
        control = self.machine.advance(transition.evidence)
        self.previous_progress = transition.progress.clone()
        self.previous_progress[control.needs_target] = 0.0
        state = assemble_deployable_task_state(
            perception=perception, robot=robot, placement=placement,
            control=control, transition_confidence=self.previous_progress)
        return DeployableRuntimeStep(
            state=state,
            actor_observation=build_actor_observation(state, previous_action),
            transition=transition,
            placement_evidence=placement_evidence,
            selected_box=selected_box,
            waiting_without_box=control.needs_target & ~placement.selectable.any(-1),
        )
