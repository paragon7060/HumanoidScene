"""Per-environment high-water-mark approach progress; no repeated dwell/cycle reward."""

import torch


class ReachProgress:
    def __init__(self, num_envs, device):
        self.best = torch.zeros(num_envs, 2, device=device)
        self.delta = torch.zeros_like(self.best)
        self.initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.target = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.best[ids] = 0
        self.delta[ids] = 0
        self.initialized[ids] = False
        self.target[ids] = -1

    def advance(self, distances, target, enabled, update):
        """Called exactly once per physics CONTROL step, never on rendering/observation reads.

        Rebase without reward on reset, target-box switch, or the first enabled
        sample after initial waiting. Flap candidate switches do NOT reset history.
        """
        score = torch.exp(-12 * distances.clamp_min(0))
        valid = enabled & torch.isfinite(distances).all(-1)
        fresh = ~self.initialized | (self.target != target)
        rewardable = update & valid & ~fresh
        self.delta[update] = 0
        self.delta[rewardable] = (score - self.best).clamp_min(0)[rewardable]
        rebased = torch.where(fresh[:, None], score, torch.maximum(self.best, score))
        active = update & valid
        self.best[active] = rebased[active]
        self.target[active] = target[active]
        self.initialized[update] = valid[update]
        # Disabled history must not leak an earlier episode/phase into observations.
        self.best[update & ~valid] = 0
