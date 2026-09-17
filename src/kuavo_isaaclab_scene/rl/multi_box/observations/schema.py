"""Structured observation tensors for set-based policies."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ActorObservation:
    robot_proprio: torch.Tensor
    anchor_poses: torch.Tensor
    box_tokens: torch.Tensor
    box_mask: torch.Tensor
    target_box: torch.Tensor
    target_one_hot: torch.Tensor
    current_skill_one_hot: torch.Tensor
    needs_target: torch.Tensor
    previous_action: torch.Tensor


@dataclass(frozen=True)
class CriticObservation:
    actor: ActorObservation
    privileged_box_tokens: torch.Tensor
    privileged_global: torch.Tensor


@dataclass(frozen=True)
class ObservationBundle:
    actor: ActorObservation
    critic: CriticObservation
