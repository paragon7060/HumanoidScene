"""Format-neutral reset snapshot container.

Task-state fields are deliberately not prescribed here.  The payload stores
named simulator tensors and metadata so a later state design can version its
own fields without coupling them to the file format.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch

from ..spec import SCHEMA_VERSION, SKILLS


@dataclass
class ResetSnapshot:
    source_skill: str
    tensors: Mapping[str, torch.Tensor]
    metadata: Mapping[str, str | int | float | bool] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Reset snapshot schema must be {SCHEMA_VERSION}.")
        if self.source_skill not in SKILLS:
            raise ValueError(f"Unknown reset snapshot source skill: {self.source_skill}")
        if not self.tensors:
            raise ValueError("Reset snapshot must contain named simulator tensors.")
        for name, value in self.tensors.items():
            if not name or not isinstance(value, torch.Tensor):
                raise TypeError("Reset snapshot payload must map non-empty names to tensors.")
            if value.requires_grad:
                raise ValueError(f"Reset snapshot tensor {name!r} must be detached.")

    def cpu_copy(self) -> "ResetSnapshot":
        self.validate()
        return ResetSnapshot(
            source_skill=self.source_skill,
            tensors={name: value.detach().cpu().clone() for name, value in self.tensors.items()},
            metadata=dict(self.metadata),
            schema_version=self.schema_version,
        )
