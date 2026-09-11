"""Signed consecutive-step approach progress, suspended during each hand's grasp."""

import torch


class ReachProgress:
    def __init__(self, num_envs, device):
        self.previous = torch.zeros(num_envs, 2, device=device)
        self.delta = torch.zeros_like(self.previous)
        self.initialized = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.target = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.previous[ids] = 0
        self.delta[ids] = 0
        self.initialized[ids] = False
        self.target[ids] = -1

    def advance(self, distances, target, enabled, update, held=None):
        """Called exactly once per physics CONTROL step, never on rendering/observation reads.

        Rebase without reward on reset, target-box switch, initial waiting, or
        grasp loss. Acquisition and held steps pay zero. Flap candidate switches
        do NOT reset history. Positive/negative changes have equal scale.
        """
        score = torch.exp(-12 * distances.clamp_min(0))
        valid = enabled[:, None] & torch.isfinite(distances)
        if held is not None:
            valid = valid & ~held
        fresh = ~self.initialized | (self.target != target)[:, None]
        rewardable = update[:, None] & valid & ~fresh
        self.delta[update] = 0
        self.delta[rewardable] = (score - self.previous)[rewardable]
        self.previous[update] = torch.where(valid, score, 0.)[update]
        self.target[update] = target[update]
        self.initialized[update] = valid[update]
