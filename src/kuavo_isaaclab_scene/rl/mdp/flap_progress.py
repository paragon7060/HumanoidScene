"""Cached pre-grasp alignment, once-per-episode grasp and held lift progress."""

import torch


class FlapProgress:
    def __init__(self, num_envs, device):
        self.alignment = torch.zeros(num_envs, 2, device=device)
        self.distance = torch.zeros_like(self.alignment)
        self.candidate = torch.full((num_envs, 2), -1, dtype=torch.long, device=device)
        self.orientation_ready = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.orientation_delta = torch.zeros_like(self.alignment)
        self.orientation_distance = torch.zeros_like(self.alignment)
        self.height = torch.zeros(num_envs, device=device)
        self.lift_ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.lift_delta = torch.zeros_like(self.height)
        self.grasp_seen = torch.zeros_like(self.lift_ready)
        self.grasp_bonus = torch.zeros_like(self.height)
        self.target = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        for tensor in (self.alignment, self.distance, self.orientation_ready,
                       self.orientation_delta, self.orientation_distance, self.height,
                       self.lift_ready, self.lift_delta, self.grasp_seen, self.grasp_bonus):
            tensor[ids] = 0
        self.target[ids] = -1
        self.candidate[ids] = -1

    def advance(self, alignment, distance, candidate, hand_held, grasped, height,
                target, enabled, update, *, prime=False):
        """One call per control step; rewards/observations only read these caches.

        height is normalized to the success height. Boundary steps only rebase:
        wait/reset/target switch, hand acquisition/loss, and flap switches.
        Grasp credit never re-arms on loss, target changes or phase changes.
        """
        same_target = self.target == target
        valid = enabled[:, None] & ~hand_held & torch.isfinite(alignment) & torch.isfinite(distance)
        score = alignment.clamp(0, 1).square()
        continuous = valid & self.orientation_ready & same_target[:, None] & (self.candidate == candidate)
        self.orientation_delta[update] = 0
        if not prime:
            mask = update[:, None] & continuous
            self.orientation_delta[mask] = (score - self.alignment)[mask]
        # Using the farther of the two distances prevents pure approach from
        # earning alignment reward, and gives symmetric weighting on a reversal.
        self.orientation_distance[update] = torch.where(continuous,
            torch.maximum(distance, self.distance), 0.)[update]
        self.alignment[update] = torch.where(valid, score, 0.)[update]
        self.distance[update] = torch.where(valid, distance, 0.)[update]
        self.orientation_ready[update] = valid[update]
        self.candidate[update] = candidate[update]

        active = enabled & torch.isfinite(height)
        held = active & grasped
        normalized = height.clamp(0, 1)
        self.lift_delta[update] = 0
        self.grasp_bonus[update] = 0
        if not prime:
            lift_mask = update & held & self.lift_ready & same_target
            self.lift_delta[lift_mask] = (normalized - self.height)[lift_mask]
            # A held reset/baseline is not an acquisition event.
            acquired = update & held & same_target & ~self.grasp_seen
            self.grasp_bonus[acquired] = 1.
        self.grasp_seen |= update & held
        self.height[update] = torch.where(held, normalized, 0.)[update]
        self.lift_ready[update] = held[update]
        # Disabled waiting must not make its first active held sample payable.
        self.target[update] = torch.where(active, target, -1)[update]

    def observation(self):
        """Eleven history values, including the non-renewable grasp credit flag."""
        return torch.cat((self.alignment, self.distance, self.candidate.float(),
                          self.orientation_ready.float(), self.height[:, None],
                          self.lift_ready[:, None].float(), self.grasp_seen[:, None].float()), -1)
