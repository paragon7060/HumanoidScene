"""Deployable fixed-order skill orchestration with a locked target box."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch

from .types import HighLevelSelection, SKILL_IDS, SkillDispatch
from ..spec import MAX_BOXES, SKILLS


NO_TARGET = -1


@dataclass(frozen=True)
class DeployableTransitionEvidence:
    """Real-sensor/perception predicates used to advance one skill at a time."""

    grasp_complete: torch.Tensor
    carry_complete: torch.Tensor
    place_complete: torch.Tensor

    def validate(self, num_envs: int, device: torch.device) -> None:
        for name, value in vars(self).items():
            if value.shape != (num_envs,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean [num_envs].")
            if value.device != device:
                raise ValueError("Transition evidence and state machine must share one device.")


class DeployableTransitionProvider(Protocol):
    """Implemented by simulator estimates or the real perception stack."""

    def evaluate(
        self,
        context: Any,
        target_box: torch.Tensor,
        current_skill: torch.Tensor,
    ) -> DeployableTransitionEvidence: ...


@dataclass(frozen=True)
class SkillControlState:
    target_box: torch.Tensor
    current_skill: torch.Tensor
    current_skill_one_hot: torch.Tensor
    needs_target: torch.Tensor
    completed_box: torch.Tensor


class SkillStateMachine:
    """Locks the selected box and advances grasp -> carry -> place."""

    def __init__(self, num_envs: int, device: str | torch.device):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        self.device = torch.device(device)
        self.target_box = torch.full((num_envs,), NO_TARGET, dtype=torch.long, device=self.device)
        self.current_skill = torch.full(
            (num_envs,), SKILL_IDS["grasp"], dtype=torch.long, device=self.device)

    @property
    def num_envs(self) -> int:
        return len(self.target_box)

    @property
    def needs_target(self) -> torch.Tensor:
        return self.target_box == NO_TARGET

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self.target_box[ids] = NO_TARGET
        self.current_skill[ids] = SKILL_IDS["grasp"]

    def assign(self, selection: HighLevelSelection, selectable_boxes: torch.Tensor) -> None:
        """Assign proposals only to environments currently waiting for a target."""
        if selectable_boxes.shape != (self.num_envs, MAX_BOXES):
            raise ValueError(f"selectable_boxes must have shape [num_envs, {MAX_BOXES}].")
        waiting_ids = self.needs_target.nonzero(as_tuple=False).flatten()
        self.assign_subset(waiting_ids, selection, selectable_boxes)

    def assign_subset(self, env_ids: torch.Tensor, selection: HighLevelSelection,
                      selectable_boxes: torch.Tensor) -> None:
        """Assign only waiting rows that have at least one selectable box."""
        if selectable_boxes.shape != (self.num_envs, MAX_BOXES) \
                or selectable_boxes.dtype != torch.bool:
            raise ValueError(f"selectable_boxes must be boolean [num_envs, {MAX_BOXES}].")
        if selectable_boxes.device != self.device:
            raise ValueError("selectable_boxes and state machine must share one device.")
        if env_ids.ndim != 1 or env_ids.dtype != torch.long or env_ids.device != self.device:
            raise ValueError("env_ids must be torch.long [selected_envs] on the tracker device.")
        if bool(((env_ids < 0) | (env_ids >= self.num_envs)).any()) \
                or len(torch.unique(env_ids)) != len(env_ids):
            raise ValueError("env_ids must contain distinct valid environments.")
        if selection.box_index.shape != (len(env_ids),):
            raise ValueError("Selection batch must match selected waiting environments.")
        if not bool(self.needs_target[env_ids].all()):
            raise ValueError("Cannot replace a locked target before place completion.")
        selection.validate(selectable_boxes[env_ids])
        self.target_box[env_ids] = selection.box_index
        self.current_skill[env_ids] = SKILL_IDS["grasp"]

    def dispatch(self, active_boxes: torch.Tensor) -> tuple[torch.Tensor, SkillDispatch]:
        """Return controller rows and deterministic skill assignments."""
        env_ids = (~self.needs_target).nonzero(as_tuple=False).flatten()
        result = SkillDispatch(self.target_box[env_ids], self.current_skill[env_ids])
        result.validate(active_boxes[env_ids])
        return env_ids, result

    def advance(self, evidence: DeployableTransitionEvidence) -> SkillControlState:
        """Advance at most one phase; place completion releases the target lock."""
        evidence.validate(self.num_envs, self.device)
        old_skill = self.current_skill.clone()
        has_target = ~self.needs_target
        grasp_done = has_target & (old_skill == SKILL_IDS["grasp"]) & evidence.grasp_complete
        carry_done = has_target & (old_skill == SKILL_IDS["carry"]) & evidence.carry_complete
        place_done = has_target & (old_skill == SKILL_IDS["place"]) & evidence.place_complete
        self.current_skill[grasp_done] = SKILL_IDS["carry"]
        self.current_skill[carry_done] = SKILL_IDS["place"]
        completed = torch.full_like(self.target_box, NO_TARGET)
        completed[place_done] = self.target_box[place_done]
        self.target_box[place_done] = NO_TARGET
        self.current_skill[place_done] = SKILL_IDS["grasp"]
        return self.state(completed)

    def state(self, completed_box=None) -> SkillControlState:
        if completed_box is None:
            completed_box = torch.full_like(self.target_box, NO_TARGET)
        one_hot = torch.nn.functional.one_hot(
            self.current_skill, num_classes=len(SKILLS)).to(torch.float32)
        return SkillControlState(
            target_box=self.target_box.clone(),
            current_skill=self.current_skill.clone(),
            current_skill_one_hot=one_hot,
            needs_target=self.needs_target.clone(),
            completed_box=completed_box.clone(),
        )
