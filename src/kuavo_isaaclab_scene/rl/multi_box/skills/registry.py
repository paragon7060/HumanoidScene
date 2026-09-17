"""Injectable low-level policy registry."""

from __future__ import annotations

from typing import Any, Protocol

from ..spec import SKILLS


class LowLevelPolicy(Protocol):
    def act(self, observation: Any, selected_box: Any, env_ids: Any) -> Any: ...


class SkillRegistry:
    """Holds grasp/carry/place policies without defining their objectives."""

    names = SKILLS

    def __init__(self, **policies: LowLevelPolicy):
        unknown = set(policies) - set(self.names)
        if unknown:
            raise KeyError(f"Unknown low-level skills: {sorted(unknown)}")
        self._policies = dict(policies)

    def __getitem__(self, name: str) -> LowLevelPolicy:
        return self._policies[name]

    def validate_complete(self) -> None:
        missing = tuple(name for name in self.names if name not in self._policies)
        if missing:
            raise ValueError(f"Low-level policy skeleton is incomplete: {', '.join(missing)}")
