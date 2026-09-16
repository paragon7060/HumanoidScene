"""Batched transfer geometry and phase-specific, signed progress caches."""
import torch


def rack_clearance(local_center, local_half, rack_center, rack_half, margin):
    """Whole box footprint must leave the rack, in its scaled local frame."""
    gap = (local_center[..., :2] - rack_center[:2]).abs() - local_half[..., :2] - rack_half[:2]
    remaining = (margin - gap.amax(-1)).clamp_min(0)
    return remaining == 0, remaining


class TransferProgress:
    def __init__(self, num_envs, device):
        self.phase = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.previous = torch.zeros(num_envs, 2, device=device)
        self.delta = torch.zeros_like(self.previous)
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def reset(self, ids):
        self.phase[ids] = -1
        self.previous[ids] = 0
        self.delta[ids] = 0
        self.ready[ids] = False

    def advance(self, extract_remaining, target_distance, phase, enabled, update):
        # Carry pays extraction + approach; place pays approach/lowering only.
        distance = torch.stack((extract_remaining, target_distance), -1)
        active = enabled & torch.isfinite(distance).all(-1) & ((phase == 2) | (phase == 3))
        continuous = update & active & self.ready & (self.phase == phase)
        self.delta[update] = 0
        self.delta[continuous] = (self.previous - distance)[continuous]
        self.previous[update] = distance[update]
        self.phase[update] = phase[update]
        self.ready[update] = active[update]

    def observation(self):
        return torch.cat((self.previous, self.ready[:, None].float()), -1)
