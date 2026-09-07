"""Continuous SAC: squashed Gaussian, twin Q, Polyak targets and learned entropy."""

from copy import deepcopy
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import mlp, ObservationNormalizer, gaussian_log_prob, optimize


@dataclass
class SACConfig:
    hidden: int = 256
    lr: float = 3e-4
    gamma: float = .99
    tau: float = .005
    initial_alpha: float = .1


class ReplayBuffer:
    """Capacity counts transitions, not vector steps. Never saved in checkpoints."""
    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.data = {key: torch.empty((capacity, *shape), device=device, dtype=dtype)
                     for key, shape, dtype in (
                         ("obs", (obs_dim,), torch.float32), ("action", (action_dim,), torch.float32),
                         ("reward", (), torch.float32), ("next_obs", (obs_dim,), torch.float32),
                         ("terminated", (), torch.bool))}

    @property
    def bytes(self):
        return sum(x.numel() * x.element_size() for x in self.data.values())

    @torch.no_grad()
    def add(self, **batch):
        count = min(len(batch["obs"]), self.capacity)
        indices = (torch.arange(count, device=self.data["obs"].device) + self.cursor) % self.capacity
        for key, storage in self.data.items():
            storage[indices] = batch[key][-count:].to(storage.device)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count, device):
        if not self.size:
            raise ValueError("Cannot sample an empty replay buffer")
        ids = torch.randint(self.size, (count,), device=self.data["obs"].device)
        return {key: value[ids].to(device) for key, value in self.data.items()}


class SquashedActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden):
        super().__init__()
        self.network = mlp(obs_dim, 2 * action_dim, hidden)

    def forward(self, obs, deterministic=False):
        mean, log_std = self.network(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        latent = mean if deterministic else mean + std * torch.randn_like(mean)
        # Stable log(1-tanh(x)^2), including highly saturated actions.
        correction = 2 * (math.log(2) - latent - F.softplus(-2 * latent))
        logp = (gaussian_log_prob(latent, mean, std) - correction).sum(-1)
        return latent.tanh(), logp


def soft_target(reward, terminated, next_q, next_logp, alpha, gamma):
    return reward + gamma * (~terminated) * (next_q - alpha * next_logp)


class SAC(nn.Module):
    def __init__(self, obs_dim, action_dim, config=None, device="cpu"):
        super().__init__()
        self.config = config or SACConfig()
        self.obs_dim, self.action_dim = obs_dim, action_dim
        cfg = self.config
        self.normalizer = ObservationNormalizer(obs_dim)
        self.actor = SquashedActor(obs_dim, action_dim, cfg.hidden)
        self.q1 = mlp(obs_dim + action_dim, 1, cfg.hidden)
        self.q2 = mlp(obs_dim + action_dim, 1, cfg.hidden)
        self.target1, self.target2 = deepcopy(self.q1), deepcopy(self.q2)
        self.target1.requires_grad_(False)
        self.target2.requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(cfg.initial_alpha)))
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_optimizer = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        return self.actor(self.normalizer(obs), deterministic)[0]

    def update(self, batch):
        cfg = self.config
        obs, next_obs = self.normalizer(batch["obs"]), self.normalizer(batch["next_obs"])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            action, logp = self.actor(next_obs)
            features = torch.cat((next_obs, action), -1)
            q = torch.minimum(self.target1(features), self.target2(features)).squeeze(-1)
            target = soft_target(batch["reward"], batch["terminated"], q, logp, alpha, cfg.gamma)
        features = torch.cat((obs, batch["action"]), -1)
        q_loss = F.mse_loss(self.q1(features).squeeze(-1), target) + F.mse_loss(self.q2(features).squeeze(-1), target)
        optimize(self.q_optimizer, q_loss, list(self.q1.parameters()) + list(self.q2.parameters()))
        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        try:
            action, logp = self.actor(obs)
            features = torch.cat((obs, action), -1)
            q = torch.minimum(self.q1(features), self.q2(features)).squeeze(-1)
            actor_loss = (alpha * logp - q).mean()
            optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
        finally:
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)
        # Target entropy = -action_dim.
        alpha_loss = -(self.log_alpha * (logp.detach() - self.action_dim)).mean()
        optimize(self.alpha_optimizer, alpha_loss, [self.log_alpha])
        with torch.no_grad():
            for source, target_net in ((self.q1, self.target1), (self.q2, self.target2)):
                for param, target_param in zip(source.parameters(), target_net.parameters()):
                    target_param.lerp_(param, cfg.tau)
        return {"q_loss": q_loss.item(), "actor_loss": actor_loss.item(), "alpha": self.log_alpha.exp().item()}

    def checkpoint(self):
        return dict(algorithm="sac", config=asdict(self.config), obs_dim=self.obs_dim,
                    action_dim=self.action_dim, model=self.state_dict(),
                    optimizers=[o.state_dict() for o in self.optimizers])

    @property
    def optimizers(self):
        return (self.actor_optimizer, self.q_optimizer, self.alpha_optimizer)

    def restore(self, state, training=True):
        self.load_state_dict(state["model"])
        if training:
            for optimizer, saved in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(saved)
