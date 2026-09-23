"""Asymmetric SAC: deployable actor observations and privileged critic state."""

from __future__ import annotations

from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ObservationNormalizer, mlp, optimize
from .sac import SACConfig, SquashedActor, soft_target


class AsymmetricReplayBuffer:
    """Replay transitions with separate actor and critic observations."""

    def __init__(
        self,
        capacity: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        device: str = "cpu",
    ):
        if capacity < 1:
            raise ValueError("Replay capacity must be positive")
        self.capacity = capacity
        self.size = 0
        self.cursor = 0
        self.data = {
            key: torch.empty((capacity, *shape), device=device, dtype=dtype)
            for key, shape, dtype in (
                ("actor_obs", (actor_obs_dim,), torch.float32),
                ("critic_obs", (critic_obs_dim,), torch.float32),
                ("action", (action_dim,), torch.float32),
                ("reward", (), torch.float32),
                ("next_actor_obs", (actor_obs_dim,), torch.float32),
                ("next_critic_obs", (critic_obs_dim,), torch.float32),
                ("terminated", (), torch.bool),
            )
        }

    @property
    def bytes(self) -> int:
        return sum(value.numel() * value.element_size() for value in self.data.values())

    @torch.no_grad()
    def add(self, **batch) -> None:
        count = min(len(batch["actor_obs"]), self.capacity)
        if count == 0:
            return
        indices = (
            torch.arange(count, device=self.data["actor_obs"].device) + self.cursor
        ) % self.capacity
        for key, storage in self.data.items():
            storage[indices] = batch[key][-count:].to(storage.device)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count: int, device: str) -> dict[str, torch.Tensor]:
        if not self.size:
            raise ValueError("Cannot sample an empty replay buffer")
        ids = torch.randint(self.size, (count,), device=self.data["actor_obs"].device)
        return {key: value[ids].to(device) for key, value in self.data.items()}


class AsymmetricSAC(nn.Module):
    """SAC whose actor never receives simulator-only critic features."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        config: SACConfig | None = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.config = config or SACConfig()
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        cfg = self.config
        if not 0 <= cfg.min_alpha <= cfg.initial_alpha:
            raise ValueError("min_alpha must be between zero and initial_alpha")
        self.actor_normalizer = ObservationNormalizer(actor_obs_dim)
        self.critic_normalizer = ObservationNormalizer(critic_obs_dim)
        self.actor = SquashedActor(actor_obs_dim, action_dim, cfg.hidden)
        self.q1 = mlp(critic_obs_dim + action_dim, 1, cfg.hidden)
        self.q2 = mlp(critic_obs_dim + action_dim, 1, cfg.hidden)
        from copy import deepcopy
        self.target1 = deepcopy(self.q1).requires_grad_(False)
        self.target2 = deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(float(cfg.initial_alpha)).log())
        self.to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @torch.no_grad()
    def update_normalizers(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> None:
        self.actor_normalizer.update(actor_obs)
        self.critic_normalizer.update(critic_obs)

    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        return self.actor(self.actor_normalizer(actor_obs), deterministic)[0]

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        cfg = self.config
        actor_obs = self.actor_normalizer(batch["actor_obs"])
        critic_obs = self.critic_normalizer(batch["critic_obs"])
        next_actor_obs = self.actor_normalizer(batch["next_actor_obs"])
        next_critic_obs = self.critic_normalizer(batch["next_critic_obs"])
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_logp = self.actor(next_actor_obs)
            target_features = torch.cat((next_critic_obs, next_action), dim=-1)
            next_q = torch.minimum(
                self.target1(target_features), self.target2(target_features)
            ).squeeze(-1)
            target = soft_target(
                batch["reward"], batch["terminated"], next_q, next_logp,
                alpha, cfg.gamma)

        replay_features = torch.cat((critic_obs, batch["action"]), dim=-1)
        q1 = self.q1(replay_features).squeeze(-1)
        q2 = self.q2(replay_features).squeeze(-1)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        optimize(
            self.q_optimizer, q_loss,
            list(self.q1.parameters()) + list(self.q2.parameters()))

        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        try:
            action, logp = self.actor(actor_obs)
            policy_features = torch.cat((critic_obs, action), dim=-1)
            q = torch.minimum(
                self.q1(policy_features), self.q2(policy_features)).squeeze(-1)
            actor_loss = (alpha * logp - q).mean()
            optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
        finally:
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)

        alpha_loss = -(self.log_alpha * (logp.detach() - self.action_dim)).mean()
        optimize(self.alpha_optimizer, alpha_loss, [self.log_alpha])
        with torch.no_grad():
            if cfg.min_alpha > 0:
                self.log_alpha.clamp_(min=math.log(cfg.min_alpha))
            for source, target_network in (
                (self.q1, self.target1), (self.q2, self.target2)
            ):
                for parameter, target_parameter in zip(
                    source.parameters(), target_network.parameters()
                ):
                    target_parameter.lerp_(parameter, cfg.tau)
        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": self.log_alpha.exp().item(),
            "policy_logp_mean": logp.detach().mean().item(),
            "policy_action_std_mean": action.detach().std(dim=0, unbiased=False).mean().item(),
        }

    @property
    def optimizers(self):
        return self.actor_optimizer, self.q_optimizer, self.alpha_optimizer

    def checkpoint(self) -> dict:
        return {
            "algorithm": "asymmetric_sac",
            "config": asdict(self.config),
            "actor_obs_dim": self.actor_obs_dim,
            "critic_obs_dim": self.critic_obs_dim,
            "action_dim": self.action_dim,
            "model": self.state_dict(),
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def restore(self, state: dict, training: bool = True) -> None:
        expected = (self.actor_obs_dim, self.critic_obs_dim, self.action_dim)
        actual = (
            state.get("actor_obs_dim"), state.get("critic_obs_dim"),
            state.get("action_dim"),
        )
        if actual != expected:
            raise ValueError(
                f"Asymmetric SAC dimensions differ: expected {expected}, got {actual}")
        self.load_state_dict(state["model"])
        if training:
            for optimizer, saved in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(saved)
