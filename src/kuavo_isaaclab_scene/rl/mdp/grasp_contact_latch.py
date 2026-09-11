"""Bounded contact hysteresis; never attach a box or restore a lost grasp by height."""

import torch


def opposed_jaws(points, centers, normal_axes):
    signed = (points - centers[:, :, None]).gather(
        -1, normal_axes[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
    return torch.isfinite(signed).all(-1) & (signed[..., 0] * signed[..., 1] < 0)


class GraspContactLatch:
    def __init__(self, num_envs, device, spec, channels=2):
        self.spec = spec
        self.active = torch.zeros(num_envs, channels, dtype=torch.bool, device=device)
        self.missing_s = torch.zeros(num_envs, channels, device=device)
        self.reference_midpoint = torch.zeros(num_envs, channels, 3, device=device)
        self.reference_gap = torch.zeros(num_envs, channels, device=device)
        self.last_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)

    def reset(self, ids):
        self.active[ids] = False
        self.missing_s[ids] = 0
        self.reference_midpoint[ids] = 0
        self.reference_gap[ids] = 0
        self.last_step[ids] = -1

    def update(self, strict, sustained_contact, jaw_local, step, dt):
        # _measure() is called by several managers during one control step.
        # Only new steps may change memory or consume the contact grace budget.
        update = self.last_step != step
        midpoint = jaw_local.mean(-2)
        gap = (jaw_local[:, :, 0] - jaw_local[:, :, 1]).norm(dim=-1)
        finite = torch.isfinite(jaw_local).all((-1, -2))
        follows = ((midpoint - self.reference_midpoint).norm(dim=-1) <= self.spec.grasp_hold_slip_m)
        follows &= gap <= self.reference_gap + self.spec.grasp_open_tolerance_m
        was_active = self.active.clone()
        acquire = strict & finite & ~was_active
        # Verified contacts outrank link-origin drift. Geometry only constrains
        # the fallback when contacts disappear; reference the LAST verified grasp.
        live = sustained_contact & finite
        missing = torch.where(live | acquire, 0., self.missing_s + dt)
        active = acquire | (was_active & finite
                            & (live | (follows & (missing < self.spec.grasp_contact_grace_s))))
        verified = (acquire | (was_active & live)) & update[:, None]
        self.reference_midpoint = torch.where(verified[..., None], midpoint, self.reference_midpoint)
        self.reference_gap = torch.where(verified, gap, self.reference_gap)
        self.active[update] = active[update]
        self.missing_s[update] = torch.where(active, missing, 0.)[update]
        self.last_step[update] = step
        return self.active.clone()
