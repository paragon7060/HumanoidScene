"""Numerical utilities shared by SAC and state-conditioned diffusion policies."""

import math
import torch
from torch import nn


def mlp(inputs, outputs, hidden=256):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(),
                         nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ObservationNormalizer(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(0.0))

    @torch.no_grad()
    def update(self, x):
        if not len(x):
            return
        var, mean = torch.var_mean(x, dim=0, unbiased=False)
        total = self.count + len(x)
        delta = mean - self.mean
        merged = (self.var * self.count + var * len(x)
                  + delta.square() * self.count * len(x) / total)
        self.mean.add_(delta * len(x) / total)
        self.var.copy_(merged / total)
        self.count.copy_(total)

    def forward(self, x):
        return ((x - self.mean) / self.var.clamp_min(1e-4).sqrt()).clamp(-10, 10)


def gaussian_log_prob(value, mean, std):
    return (-0.5 * ((value - mean) / std).square()
            - std.log() - 0.5 * math.log(2 * math.pi))


def generalized_advantage(rewards, values, next_values, terminated, done, gamma=.99, lam=.95):
    """Bootstrap timeouts from terminal observations; never carry GAE across resets."""
    advantages = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * (~terminated[t]) * next_values[t] - values[t]
        carry = delta + gamma * lam * (~done[t]) * carry
        advantages[t] = carry
    return advantages, advantages + values


def optimize(optimizer, loss, parameters, max_norm=1.0):
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite training loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
    optimizer.step()
