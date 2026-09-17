"""Task-semantic call order without any built-in task definition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .contracts import TaskSemantics


@dataclass(frozen=True)
class SemanticStep:
    task_state: Any
    success: Any
    observation: Any
    reward: Any


class SemanticPipeline:
    """Runs approved providers in a stable order for managers and debug tools."""

    def __init__(self, semantics: TaskSemantics):
        semantics.require_complete()
        self.semantics = semantics

    def reset(self, context: Any, env_ids: Any) -> None:
        self.semantics.state.reset(context, env_ids)

    def step(self, context: Any) -> SemanticStep:
        task_state = self.semantics.state.update(context)
        success = self.semantics.success.evaluate(context)
        observation = self.semantics.observation.build(context, task_state)
        reward = self.semantics.reward.compute(context, task_state, success)
        return SemanticStep(task_state, success, observation, reward)
