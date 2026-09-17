"""High-level target selection and deterministic low-level dispatch types."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..spec import MAX_BOXES, SKILLS


SKILL_IDS = {name: index for index, name in enumerate(SKILLS)}


def _validate_box_indices(box_index: torch.Tensor, boxes: torch.Tensor, label: str) -> None:
    if box_index.dtype != torch.long or box_index.ndim != 1:
        raise TypeError("Box indices must be torch.long [num_envs].")
    if boxes.shape != (len(box_index), MAX_BOXES) or boxes.dtype != torch.bool:
        raise ValueError(f"{label} must be boolean [num_envs, {MAX_BOXES}].")
    if box_index.device != boxes.device:
        raise ValueError("Box indices and masks must share one device.")
    if not bool(((0 <= box_index) & (box_index < MAX_BOXES)).all()):
        raise ValueError("Selected box index is outside the logical box set.")
    env_ids = torch.arange(len(box_index), device=box_index.device)
    if not bool(boxes[env_ids, box_index].all()):
        raise ValueError(f"Selected box is inactive or unavailable in {label}.")


@dataclass(frozen=True)
class HighLevelSelection:
    """The high-level policy chooses only a target box."""

    box_index: torch.Tensor

    def validate(self, selectable_boxes: torch.Tensor) -> None:
        _validate_box_indices(self.box_index, selectable_boxes, "selectable_boxes")


@dataclass(frozen=True)
class SkillDispatch:
    """Locked target and current skill produced by the state machine."""

    box_index: torch.Tensor
    skill_index: torch.Tensor

    def validate(self, active_boxes: torch.Tensor) -> None:
        _validate_box_indices(self.box_index, active_boxes, "active_boxes")
        if self.skill_index.dtype != torch.long or self.skill_index.shape != self.box_index.shape:
            raise TypeError("Skill indices must be torch.long [num_envs].")
        if self.skill_index.device != self.box_index.device:
            raise ValueError("Box and skill indices must share one device.")
        if not bool(((0 <= self.skill_index) & (self.skill_index < len(SKILLS))).all()):
            raise ValueError("Skill index is outside the grasp/carry/place sequence.")
