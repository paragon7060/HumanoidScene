"""Episode-stable expert collection and honest SAC success attribution."""

import torch


class EpisodicIKGuidance:
    """Sample assistance at episode boundaries; never switch an active episode."""

    def __init__(self, num_envs, device, initial_fraction, decay_updates):
        if not 0 <= initial_fraction < 1 or decay_updates < 1:
            raise ValueError("Invalid episodic IK guidance schedule")
        self.initial_fraction, self.decay_updates = initial_fraction, decay_updates
        self.mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.used_teacher = torch.zeros_like(self.mask)
        self.initialized = False

    def fraction(self, actor_updates):
        return self.initial_fraction * max(0.0, 1.0 - actor_updates / self.decay_updates)

    def select(self, *, warming_up, ready, actor_updates):
        if warming_up:
            self.used_teacher |= ready
            return torch.zeros_like(self.mask)
        if not self.initialized:
            self.mask = torch.rand_like(self.mask, dtype=torch.float32) < self.fraction(actor_updates)
            self.initialized = True
        self.used_teacher |= self.mask & ready
        return self.mask.clone()

    def successes(self, success, *, warming_up):
        if warming_up:
            return int(success.sum()), 0, 0, 0
        ik = success & self.mask
        sac = success & ~self.mask
        pure = sac & ~self.used_teacher
        return 0, int(ik.sum()), int(sac.sum()), int(pure.sum())

    def reset(self, done, actor_updates):
        self.used_teacher[done] = False
        if self.initialized and bool(done.any()):
            sampled = torch.rand_like(self.mask, dtype=torch.float32) < self.fraction(actor_updates)
            self.mask[done] = sampled[done]
