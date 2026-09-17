"""Live per-box placement state for the full multi-box task."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from ..spec import MAX_BOXES


@dataclass(frozen=True)
class PlacementSetState:
    active: torch.Tensor
    placed: torch.Tensor
    selectable: torch.Tensor
    hold_time_s: torch.Tensor
    full_success: torch.Tensor


class PlacementSetTracker:
    """Tracks live placement; disturbed boxes immediately become selectable."""

    def __init__(self, num_envs: int, device: str | torch.device, *, hold_seconds=0.50):
        if num_envs < 1:
            raise ValueError("num_envs must be positive.")
        if not math.isfinite(hold_seconds) or hold_seconds <= 0:
            raise ValueError("hold_seconds must be finite and positive.")
        self.hold_seconds = hold_seconds
        self.hold_time_s = torch.zeros(num_envs, MAX_BOXES, device=device)
        self.placed = torch.zeros(num_envs, MAX_BOXES, dtype=torch.bool, device=device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self.hold_time_s.zero_()
            self.placed.zero_()
        else:
            self.hold_time_s[env_ids] = 0.0
            self.placed[env_ids] = False

    def update(self, active: torch.Tensor, valid_place: torch.Tensor, dt: float) -> PlacementSetState:
        expected = self.placed.shape
        if active.shape != expected or valid_place.shape != expected:
            raise ValueError(f"active and valid_place must have shape {list(expected)}.")
        if active.dtype != torch.bool or valid_place.dtype != torch.bool:
            raise TypeError("active and valid_place must be boolean.")
        if active.device != self.placed.device or valid_place.device != self.placed.device:
            raise ValueError("Placement masks and tracker must share one device.")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive.")

        live = active & valid_place
        self.hold_time_s = torch.where(
            live, self.hold_time_s + dt, torch.zeros_like(self.hold_time_s))
        # This assignment deliberately revokes placement immediately when a
        # previously completed box is displaced, tilted, overlapped, or moving.
        self.placed = live & (self.hold_time_s >= self.hold_seconds)
        selectable = active & ~self.placed
        full_success = active.any(dim=-1) & ((~active) | self.placed).all(dim=-1)
        return PlacementSetState(
            active.clone(),
            self.placed.clone(),
            selectable,
            self.hold_time_s.clone(),
            full_success,
        )
