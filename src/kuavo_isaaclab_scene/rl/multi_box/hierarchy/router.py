"""Policy dispatch without implicit phase transitions or target reselection."""

from __future__ import annotations

from typing import Any

import torch

from .types import SkillDispatch
from ..skills.registry import SkillRegistry


class SkillRouter:
    def __init__(self, skills: SkillRegistry):
        skills.validate_complete()
        self.skills = skills

    def act(
        self,
        observation: Any,
        dispatch: SkillDispatch,
        active_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Dispatch subsets to skills and restore actions to environment order."""
        dispatch.validate(active_boxes)
        result = None
        for skill_index, skill_name in enumerate(self.skills.names):
            env_ids = (dispatch.skill_index == skill_index).nonzero(as_tuple=False).flatten()
            if not len(env_ids):
                continue
            actions = self.skills[skill_name].act(observation, dispatch.box_index, env_ids)
            if actions.ndim != 2 or len(actions) != len(env_ids):
                raise ValueError(f"Skill {skill_name!r} must return [selected_envs, action_dim].")
            if result is None:
                result = actions.new_zeros((len(dispatch.box_index), actions.shape[1]))
            elif result.shape[1] != actions.shape[1]:
                raise ValueError("All low-level skills must share one action dimension.")
            result[env_ids] = actions
        if result is None:
            raise ValueError("Cannot route an empty high-level action batch.")
        return result
