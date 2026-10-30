"""Gripper observations independent of the owning articulation and robot model.

Integrated hands are views of robot joints; mounted hands are views of their
own articulation. Neither view changes the USD hierarchy or joint ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping, Sequence

import torch

from .gripper_config import GripperSettings


def claw_fraction(joint_pos: torch.Tensor, *, open_pos: torch.Tensor,
                  close_pos: torch.Tensor) -> torch.Tensor:
    """Compress a hand pose into one measured scalar: 0=open, 1=closed."""
    if (
        joint_pos.ndim < 1 or open_pos.ndim < 1 or close_pos.ndim < 1
        or joint_pos.shape[-1] != open_pos.shape[-1]
        or joint_pos.shape[-1] != close_pos.shape[-1]
    ):
        raise ValueError("Hand position/open/close tensors must have the same joint dimension.")
    span = close_pos - open_pos
    movable = torch.abs(span) > 1.0e-6
    if not bool(torch.all(movable.any(dim=-1))):
        raise ValueError("At least one hand joint must differ between open and close poses.")
    safe_span = torch.where(movable, span, torch.ones_like(span))
    fraction = (joint_pos - open_pos) / safe_span
    fraction = torch.where(movable, fraction, torch.zeros_like(fraction)).clamp(0.0, 1.0)
    return fraction.sum(dim=-1, keepdim=True) / movable.sum(dim=-1, keepdim=True)


def _command_tensor(names: Sequence[str], expressions: Mapping[str, float],
                    *, device: Any, dtype: Any) -> torch.Tensor:
    values = []
    for name in names:
        matches = [float(value) for pattern, value in expressions.items() if re.fullmatch(pattern, name)]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one hand command for {name!r}; got {len(matches)}.")
        values.append(matches[0])
    return torch.tensor(values, device=device, dtype=dtype).unsqueeze(0)


def _find_joints(asset: Any, patterns: Sequence[str]) -> tuple[list[int], tuple[str, ...]]:
    ids, names = asset.find_joints(list(patterns), preserve_order=True)
    if not ids or len(set(ids)) != len(ids):
        raise ValueError(f"Gripper joint patterns must resolve to distinct joints: {patterns}")
    return list(ids), tuple(names)


@dataclass
class GripperView:
    side: str
    asset_name: str
    asset: Any
    joint_ids: list[int]
    joint_names: tuple[str, ...]
    command_joint_ids: list[int]
    open_pos: torch.Tensor | None = None
    close_pos: torch.Tensor | None = None

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(f"{self.side}_{name}" for name in self.joint_names)

    def joint_state(self, *, relative: bool = False) -> torch.Tensor:
        values = self.asset.data.joint_pos[:, self.joint_ids]
        if relative:
            values = values - self.asset.data.default_joint_pos[:, self.joint_ids]
        return values.clone()

    def claw_state(self) -> torch.Tensor:
        if self.open_pos is None or self.close_pos is None:
            raise ValueError(f"{self.side} gripper needs configured open/close commands for claw state.")
        return claw_fraction(
            self.asset.data.joint_pos[:, self.command_joint_ids],
            open_pos=self.open_pos, close_pos=self.close_pos,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "joint_names": list(self.joint_names),
            "state_names": list(self.state_names),
            "command_joint_names": [self.asset.data.joint_names[i] for i in self.command_joint_ids],
        }


def resolve_gripper_views(env: Any, settings: GripperSettings | None) -> dict[str, GripperView]:
    """Resolve both hand types through the preset, failing on missing enabled hands.

    Without a preset, retain external-articulation discovery for legacy callers.
    Generic external joint observations retain their existing all-joint layout.
    Integrated observations use the preset's controlled hand joints (not bars
    that passively follow the closed linkage).
    """
    result = {}
    sides = settings.active_sides if settings is not None else ("left", "right")
    for side in sides:
        asset_name = settings.asset_name_for(side) if settings else f"{side}_gripper"
        try:
            asset = env.scene[asset_name]
        except KeyError as exc:
            if settings is None:
                continue
            raise ValueError(f"Enabled {side} gripper asset {asset_name!r} is missing.") from exc
        command_ids, command_names = _find_joints(
            asset, settings.joint_names_for(side) if settings else (".*",)
        )
        if settings is not None and settings.integrated:
            ids, names = command_ids, command_names
        else:
            ids, names = _find_joints(asset, (".*",))
        view = GripperView(side, asset_name, asset, ids, names, command_ids)
        if settings is not None:
            view.open_pos = _command_tensor(
                command_names, settings.command_for(side, settings.open_command),
                device=asset.device, dtype=asset.data.joint_pos.dtype,
            )
            view.close_pos = _command_tensor(
                command_names, settings.command_for(side, settings.close_command),
                device=asset.device, dtype=asset.data.joint_pos.dtype,
            )
            view.claw_state()  # Validate the calibration before the first rollout.
        result[side] = view
    return result
