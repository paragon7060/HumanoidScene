"""Simulator-independent assembly skeleton for multi-box v2."""

from __future__ import annotations

from dataclasses import dataclass, field

from .contracts import TaskSemantics
from .skills import SkillRegistry
from .spec import MultiBoxSpec


@dataclass
class MultiBoxTaskSkeleton:
    spec: MultiBoxSpec = field(default_factory=MultiBoxSpec)
    semantics: TaskSemantics = field(default_factory=TaskSemantics)
    skills: SkillRegistry = field(default_factory=SkillRegistry)

    def validate_scene(self) -> None:
        """Validate decisions that are already fixed and safe to build."""
        self.spec.validate()

    def validate_training(self) -> None:
        """Reject training until semantic providers and all policies exist."""
        self.validate_scene()
        self.semantics.require_complete()
        self.skills.validate_complete()
