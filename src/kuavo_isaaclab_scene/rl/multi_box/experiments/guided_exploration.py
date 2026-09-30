"""Deployable-observation-only, temporally coherent SAC demonstration warmup."""

from __future__ import annotations

import math

import torch

from ..observations.builder import (
    BOX_TOKEN_DIM, HAND_FLAP_RELATION_DIM, ROBOT_PROPRIO_DIM,
)
from ..spec import MAX_BOXES


RELATION_START = ROBOT_PROPRIO_DIM + 2 * 9 + MAX_BOXES * BOX_TOKEN_DIM
ASSIGNMENT_START = RELATION_START + HAND_FLAP_RELATION_DIM


def assigned_flap_center_distance(actor_obs: torch.Tensor) -> torch.Tensor:
    """Read the same assigned relative flap centers seen by the deployed actor."""
    if actor_obs.ndim != 2 or actor_obs.shape[1] < ASSIGNMENT_START + 2:
        raise ValueError("V2 actor observation lacks assigned flap relations")
    relations = actor_obs[:, RELATION_START:ASSIGNMENT_START].reshape(-1, 2, 2, 9)
    assignment = actor_obs[:, ASSIGNMENT_START:ASSIGNMENT_START + 2]
    valid = assignment.sum(-1) > 0.5
    left = assignment.argmax(-1)
    hand = torch.arange(2, device=actor_obs.device)[None]
    flap = torch.stack((left, 1 - left), dim=-1)
    rows = torch.arange(len(actor_obs), device=actor_obs.device)[:, None]
    distance = relations[rows, hand, flap, :3].norm(dim=-1)
    return torch.where(valid[:, None], distance, torch.full_like(distance, torch.inf))


def critical_teacher_rows(actor_obs, action):
    """Give precise bilateral approach/closure labels their own imitation stratum."""
    near_both = assigned_flap_center_distance(actor_obs).amax(-1) <= 0.25
    return near_both | (action[:, 20:22] > 0).any(-1)


class GuidedDemoWarmup:
    """Follow the pretrained actor with correlated continuous perturbations.

    This is an exploration aid, not a collision-free motion planner. The
    environment's existing contact termination remains the safety authority.
    """

    def __init__(self, env, *, noise_scale: float = 0.12,
                 correlation: float = 0.90, close_distance_m: float = 0.12):
        if not 0 <= noise_scale <= 1 or not 0 <= correlation < 1 \
                or not 0 < close_distance_m <= 0.2:
            raise ValueError("Invalid guided warmup parameters")
        self.noise_scale = noise_scale
        self.correlation = correlation
        self.close_distance_m = close_distance_m
        self.noise = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device)
        self.gripper_columns = {}
        offset = 0
        for name in env.action_manager.active_terms:
            width = env.action_manager.get_term(name).action_dim
            if name in ("left_gripper", "right_gripper"):
                if width != 1:
                    raise ValueError("Guided warmup expects binary one-dimensional grippers")
                self.gripper_columns[name] = offset
            offset += width
        if set(self.gripper_columns) != {"left_gripper", "right_gripper"}:
            raise ValueError("Guided warmup requires both gripper action terms")

    def act(self, agent, actor_obs: torch.Tensor) -> torch.Tensor:
        action = agent.act(actor_obs, deterministic=True).clone()
        self.noise.mul_(self.correlation).add_(
            torch.randn_like(self.noise),
            alpha=self.noise_scale * math.sqrt(1 - self.correlation**2),
        )
        for column in self.gripper_columns.values():
            self.noise[:, column] = 0.0
        action.add_(self.noise).clamp_(-1, 1)
        distance = assigned_flap_center_distance(actor_obs)
        for hand, name in enumerate(("left_gripper", "right_gripper")):
            column = self.gripper_columns[name]
            near = distance[:, hand] <= self.close_distance_m
            action[:, column] = torch.where(
                near & (action[:, column] > 0), 1.0, -1.0)
        return action

    def reset(self, done: torch.Tensor) -> None:
        if done.shape != (len(self.noise),) or done.dtype != torch.bool:
            raise ValueError("Done mask must be one boolean per environment")
        self.noise[done] = 0.0


class GraspActionProjector:
    """Keep both SAC grippers open until their assigned flap is close enough.

    The same projection must be used for collection, actor optimization,
    Bellman target actions, and deterministic checkpoint playback.
    """

    name = "v2_assigned_flap_close_gate_0p12m"

    def __init__(self, action_terms, close_distance_m: float = 0.12):
        if not 0 < close_distance_m <= 0.2:
            raise ValueError("Close distance must be in (0, 0.2] metres")
        offset = 0
        columns = {}
        for name, width in action_terms:
            if name in ("left_gripper", "right_gripper"):
                if width != 1:
                    raise ValueError("Gripper actions must be one-dimensional")
                columns[name] = offset
            offset += width
        if set(columns) != {"left_gripper", "right_gripper"}:
            raise ValueError("Both gripper actions are required")
        self.action_dim = offset
        self.columns = (columns["left_gripper"], columns["right_gripper"])
        self.close_distance_m = close_distance_m

    def __call__(self, actor_obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.shape != (len(actor_obs), self.action_dim):
            raise ValueError("Projected action shape differs from the environment")
        near = assigned_flap_center_distance(actor_obs) <= self.close_distance_m
        projected = action.clone()
        for hand, column in enumerate(self.columns):
            projected[:, column] = torch.where(
                near[:, hand], action[:, column], -torch.ones_like(action[:, column]))
        return projected

    def entropy_mask(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Excluded far-away gripper outputs must not earn entropy credit."""
        near = assigned_flap_center_distance(actor_obs) <= self.close_distance_m
        mask = torch.ones(len(actor_obs), self.action_dim, device=actor_obs.device)
        for hand, column in enumerate(self.columns):
            mask[:, column] = near[:, hand].to(mask.dtype)
        return mask
