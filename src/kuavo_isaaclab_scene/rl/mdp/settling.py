"""Per-environment settling; never advance other environments inside a reset."""

import torch


class ResetSettling:
    def __init__(self, num_envs, device, spec):
        self.spec = spec
        self.ready = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.stable_time = torch.zeros_like(self.elapsed)

    def reset(self, ids):
        self.ready[ids] = False
        self.elapsed[ids] = 0
        self.stable_time[ids] = 0

    def advance(self, velocities, update, dt):
        pending = ~self.ready & update
        self.elapsed += pending * dt
        stable = ((velocities[..., :3].norm(dim=-1) < .01)
                  & (velocities[..., 3:].norm(dim=-1) < .05)).all(-1)
        stable &= self.elapsed >= self.spec.reset_settle_seconds
        self.stable_time[pending] = torch.where(stable[pending], self.stable_time[pending] + dt, 0.)
        completed = pending & (self.stable_time >= self.spec.reset_settle_hold_seconds)
        self.ready |= completed
        return completed

    @property
    def failed(self):
        return ~self.ready & (self.elapsed >= self.spec.reset_settle_timeout)


def ready(command):
    return command.settling.ready if command.settling is not None else 1.0


def gate_actions(env, actions):
    if env.cfg.task.reset_settle_seconds <= 0 or env.cfg.task.reset_bank:
        return actions
    command = env.command_manager.get_term("workcell")
    return actions * command.settling.ready[:, None]
