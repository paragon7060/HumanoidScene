"""Explicit design boundary for the v2 task.

The randomized scene can be built before task semantics are agreed.  Runtime
training cannot: success, internal state, policy observations, and rewards must
all be supplied deliberately instead of inheriting the legacy four-box task.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class DesignArea(str, Enum):
    SUCCESS = "success"
    STATE = "state"
    OBSERVATION = "observation"
    REWARD = "reward"


class PendingTaskDesignError(RuntimeError):
    """Raised when code tries to assemble training before design is approved."""


@runtime_checkable
class SuccessEvaluator(Protocol):
    def evaluate(self, context: Any) -> Any:
        """Return per-environment success/failure results."""


@runtime_checkable
class StateProvider(Protocol):
    def reset(self, context: Any, env_ids: Any) -> None: ...
    def update(self, context: Any) -> Any: ...


@runtime_checkable
class ObservationProvider(Protocol):
    def build(self, context: Any, task_state: Any) -> Any: ...


@runtime_checkable
class RewardProvider(Protocol):
    def compute(self, context: Any, task_state: Any, success: Any) -> Any: ...


@dataclass(frozen=True)
class TaskSemantics:
    """Dependency bundle filled only after the four task-design decisions."""

    success: SuccessEvaluator | None = None
    state: StateProvider | None = None
    observation: ObservationProvider | None = None
    reward: RewardProvider | None = None

    @property
    def pending(self) -> tuple[DesignArea, ...]:
        return tuple(
            area for area, value in (
                (DesignArea.SUCCESS, self.success),
                (DesignArea.STATE, self.state),
                (DesignArea.OBSERVATION, self.observation),
                (DesignArea.REWARD, self.reward),
            ) if value is None
        )

    def require_complete(self) -> None:
        if self.pending:
            names = ", ".join(area.value for area in self.pending)
            raise PendingTaskDesignError(
                f"Multi-box v2 task design is pending: {names}. "
                "Approve these semantics before assembling a training environment."
            )
