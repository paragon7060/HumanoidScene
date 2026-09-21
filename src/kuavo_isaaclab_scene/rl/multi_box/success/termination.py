"""Shared low-level terminal contract for staged multi-box skills."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SkillTerminationInput:
    """One-step predicates produced by privileged training adapters."""

    success: torch.Tensor
    unsafe: torch.Tensor
    phase_armed: torch.Tensor
    grasp_maintained: torch.Tensor
    box_tilt_ok: torch.Tensor
    released: torch.Tensor
    placement_region: torch.Tensor

    def validate(self) -> None:
        shape = self.success.shape
        for name, value in vars(self).items():
            if value.shape != shape or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean {list(shape)}")
        if len({value.device for value in vars(self).values()}) != 1:
            raise ValueError("Termination predicates must share one device")


@dataclass(frozen=True)
class SkillTerminationResult:
    success: torch.Tensor
    failure: torch.Tensor
    terminated: torch.Tensor
    grasp_loss: torch.Tensor
    tilt_violation: torch.Tensor
    premature_release: torch.Tensor


def low_level_termination(
    skill: str,
    value: SkillTerminationInput,
) -> SkillTerminationResult:
    """Apply terminal rules without allowing a later success to erase failure."""
    if skill not in {"grasp", "carry", "place"}:
        raise ValueError(f"Unsupported low-level skill: {skill}")
    value.validate()
    false = torch.zeros_like(value.success)
    grasp_loss = (
        value.phase_armed & ~value.grasp_maintained
        if skill == "carry" else false
    )
    tilt_violation = (
        value.phase_armed & ~value.box_tilt_ok
        if skill == "carry" else false
    )
    premature_release = (
        value.phase_armed & value.released & ~value.placement_region
        if skill == "place" else false
    )
    failure = value.unsafe | grasp_loss | tilt_violation | premature_release
    # A simultaneous failure wins; it cannot be offset by the success bonus.
    success = value.success & ~failure
    return SkillTerminationResult(
        success=success,
        failure=failure,
        terminated=success | failure,
        grasp_loss=grasp_loss,
        tilt_violation=tilt_violation,
        premature_release=premature_release,
    )
