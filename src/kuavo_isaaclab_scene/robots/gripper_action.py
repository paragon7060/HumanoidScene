"""Simulator-independent scalar gripper action conversions."""

from __future__ import annotations

import torch


def interpolate_signed_gripper_action(
    actions: torch.Tensor,
    open_command: torch.Tensor,
    close_command: torch.Tensor,
) -> torch.Tensor:
    """Interpolate hand targets while retaining the legacy signed convention.

    ``+1`` is fully open, ``-1`` is fully closed, and values in between retain
    the continuous claw fraction emitted by LeRobot policies.
    """
    close_fraction = ((1.0 - actions.float()) * 0.5).clamp(0.0, 1.0)
    return open_command + close_fraction * (close_command - open_command)
