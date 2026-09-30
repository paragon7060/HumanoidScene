"""Asymmetric SAC: deployable actor observations and privileged critic state."""

from __future__ import annotations

from dataclasses import asdict
import math

import torch
from torch import nn
from torch.nn import functional as F

from .common import ObservationNormalizer, mlp, optimize
from .sac import SACConfig, SquashedActor, soft_target


class ActorFeatures(nn.Module):
    """Route the selected box into one slot; preserve deployable inputs only."""

    def __init__(self, input_dim: int, mode: str):
        super().__init__()
        self.mode = mode
        if mode == "flat":
            self.output_dim = input_dim
        elif mode == "grasp_target":
            from ..multi_box.observations.builder import BOX_TOKEN_DIM, ROBOT_PROPRIO_DIM
            from ..multi_box.spec import MAX_BOXES
            self.token_start = ROBOT_PROPRIO_DIM + 18
            self.token_width = BOX_TOKEN_DIM
            self.boxes = MAX_BOXES
            self.relation_start = self.token_start + MAX_BOXES * BOX_TOKEN_DIM
            self.mask_start = self.relation_start + 38
            self.target_start = self.mask_start + MAX_BOXES
            self.tail_start = self.target_start + MAX_BOXES
            if input_dim < self.tail_start + 5:
                raise ValueError("grasp_target needs the deployable v2 observation contract")
            self.output_dim = self.token_start + BOX_TOKEN_DIM + 38 + input_dim - self.tail_start
        else:
            raise ValueError(f"Unknown actor feature mode: {mode}")

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.mode == "flat":
            return observation
        target = observation[:, self.target_start:self.tail_start]
        selected = target.argmax(-1)
        tokens = observation[:, self.token_start:self.relation_start].reshape(
            -1, self.boxes, self.token_width)
        rows = torch.arange(len(observation), device=observation.device)
        mask = observation[:, self.mask_start:self.target_start]
        valid = (target.sum(-1) > 0.5) & (mask[rows, selected] > 0.5)
        token = tokens[rows, selected] * valid[:, None]
        return torch.cat((observation[:, :self.token_start], token,
                          observation[:, self.relation_start:self.mask_start],
                          observation[:, self.tail_start:]), -1)


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


class ActorImitationBuffer:
    """Controller labels only: never represent hypothetical actions as Q transitions."""

    def __init__(self, capacity: int, obs_dim: int, action_dim: int,
                 priority_fn=None, priority_fraction=0.0, priority_capacity=0):
        if capacity < 1:
            raise ValueError("Imitation capacity must be positive")
        if not 0 <= priority_fraction < 1 or (priority_fraction and priority_fn is None):
            raise ValueError("Imitation priority fraction requires a predicate")
        if priority_capacity < 0 or (priority_capacity and priority_fn is None):
            raise ValueError("Persistent imitation priority requires a predicate")
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.priority_fn, self.priority_fraction = priority_fn, priority_fraction
        self.priority = torch.zeros(capacity, dtype=torch.bool)
        self._priority_indices = None
        self.data = {
            "actor_obs": torch.empty(capacity, obs_dim),
            "action": torch.empty(capacity, action_dim),
        }
        # Sampling priority alone cannot keep rare labels alive when a large
        # batch of off-target states overwrites the ordinary FIFO.
        self.priority_capacity = priority_capacity
        self.priority_size = self.priority_cursor = 0
        self.priority_data = {key: torch.empty(priority_capacity, *value.shape[1:])
                              for key, value in self.data.items()}

    @torch.no_grad()
    def add(self, *, actor_obs, action):
        actor_obs, action = actor_obs.detach().cpu(), action.detach().cpu()
        selected = self.priority_fn(actor_obs, action) if self.priority_fn is not None else None
        count = min(len(action), self.capacity)
        if not count:
            return
        ids = (torch.arange(count) + self.cursor) % self.capacity
        self.data["actor_obs"][ids] = actor_obs[-count:]
        self.data["action"][ids] = action[-count:]
        if self.priority_fn is not None:
            self.priority[ids] = selected[-count:]
            self._priority_indices = None
        if self.priority_capacity:
            critical = selected.nonzero(as_tuple=False).flatten()[-self.priority_capacity:]
            n = len(critical)
            if n:
                slots = (torch.arange(n) + self.priority_cursor) % self.priority_capacity
                self.priority_data["actor_obs"][slots] = actor_obs[critical]
                self.priority_data["action"][slots] = action[critical]
                self.priority_cursor = (self.priority_cursor + n) % self.priority_capacity
                self.priority_size = min(self.priority_size + n, self.priority_capacity)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.size + count, self.capacity)

    def sample(self, count, device):
        if not self.size:
            raise ValueError("Cannot sample empty controller labels")
        ids = torch.randint(self.size, (count,))
        if self.priority_size:
            prioritized = round(count * self.priority_fraction)
            critical_ids = torch.randint(self.priority_size, (prioritized,))
            return {key: torch.cat((self.priority_data[key][critical_ids], value[ids[prioritized:]])).to(device)
                    for key, value in self.data.items()}
        if self._priority_indices is None:
            self._priority_indices = torch.where(self.priority[:self.size])[0]
        selected = self._priority_indices
        prioritized = round(count * self.priority_fraction) if len(selected) else 0
        if prioritized:
            ids[:prioritized] = selected[torch.randint(len(selected), (prioritized,))]
        return {key: value[ids].to(device) for key, value in self.data.items()}

    def snapshot(self, max_rows=100_000):
        if self.size <= max_rows and not self.priority_size:
            return {key: value[:self.size].clone() for key, value in self.data.items()}
        return self.sample(min(max_rows, max(self.size, self.priority_size)), "cpu")


class SuccessfulTransitionHistory:
    """Retain genuine contiguous pre-success transitions separately per environment."""

    def __init__(self, num_envs, horizon, actor_dim, critic_dim, action_dim):
        if min(num_envs, horizon) < 1:
            raise ValueError("Success history dimensions must be positive")
        self.horizon = horizon
        self.data = AsymmetricReplayBuffer(
            num_envs * horizon, actor_dim, critic_dim, action_dim, "cpu").data
        self.cursor = torch.zeros(num_envs, dtype=torch.long)
        self.count = torch.zeros_like(self.cursor)

    def add(self, env_ids, **batch):
        env_ids = env_ids.detach().cpu()
        ids = env_ids * self.horizon + self.cursor[env_ids]
        for key, storage in self.data.items():
            storage[ids] = batch[key].detach().cpu()
        self.cursor[env_ids] = (self.cursor[env_ids] + 1) % self.horizon
        self.count[env_ids] = (self.count[env_ids] + 1).clamp_max(self.horizon)

    def tails(self, env_ids):
        env_ids = env_ids.detach().cpu()
        step = torch.arange(self.horizon)[None]
        count = self.count[env_ids, None]
        index = (self.cursor[env_ids, None] - count + step) % self.horizon
        ids = (env_ids[:, None] * self.horizon + index)[step < count]
        return {key: value[ids] for key, value in self.data.items()}

    def reset(self, mask):
        mask = mask.detach().cpu()
        self.cursor[mask] = 0
        self.count[mask] = 0


class AsymmetricSAC(nn.Module):
    """SAC whose actor never receives simulator-only critic features."""

    def __init__(
        self,
        actor_obs_dim: int,
        critic_obs_dim: int,
        action_dim: int,
        config: SACConfig | None = None,
        device: str = "cpu",
        action_projector=None,
    ):
        super().__init__()
        self.config = config or SACConfig()
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.action_dim = action_dim
        self.action_projector = action_projector
        cfg = self.config
        if not math.isfinite(cfg.reward_scale) or cfg.reward_scale <= 0:
            raise ValueError("reward_scale must be positive and finite")
        if not math.isfinite(cfg.initial_policy_std) or not 0 < cfg.initial_policy_std <= 1:
            raise ValueError("initial_policy_std must be in (0, 1]")
        if not math.isfinite(cfg.max_policy_std) or cfg.max_policy_std < cfg.initial_policy_std:
            raise ValueError("max_policy_std must be finite and at least initial_policy_std")
        if not 0 <= cfg.min_alpha <= cfg.initial_alpha:
            raise ValueError("min_alpha must be between zero and initial_alpha")
        self.actor_features = actor_features = ActorFeatures(actor_obs_dim, cfg.actor_feature_mode)
        self.actor_normalizer = ObservationNormalizer(actor_features.output_dim)
        self.critic_normalizer = ObservationNormalizer(critic_obs_dim)
        self.actor = SquashedActor(actor_features.output_dim, action_dim, cfg.hidden, cfg.max_policy_std)
        # H(tanh(N)) <= H(N). A fixed -1/dim target is unattainable
        # under tight std caps (e.g. 0.02 => at most -2.493 nats/dim).
        # Leave 0.5 nats/dim below that bound, preserving the usual target
        # whenever the cap already permits it.
        self.target_entropy_per_dim = min(
            -1.0, self.actor.log_std_max + 0.5 * math.log(2 * math.pi * math.e) - 0.5)
        if cfg.initial_policy_std != 1.0:
            with torch.no_grad():
                output = self.actor.network[-1]
                output.weight[action_dim:].zero_()
                output.bias[action_dim:].fill_(math.log(cfg.initial_policy_std))
        self.q1 = mlp(critic_obs_dim + action_dim, 1, cfg.hidden)
        self.q2 = mlp(critic_obs_dim + action_dim, 1, cfg.hidden)
        from copy import deepcopy
        self.target1 = deepcopy(self.q1).requires_grad_(False)
        self.target2 = deepcopy(self.q2).requires_grad_(False)
        self.log_alpha = nn.Parameter(torch.tensor(float(cfg.initial_alpha)).log())
        self.to(device)
        if cfg.actor_lr is not None and (not math.isfinite(cfg.actor_lr) or cfg.actor_lr <= 0):
            raise ValueError("actor_lr must be positive and finite")
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=cfg.actor_lr if cfg.actor_lr is not None else cfg.lr)
        self.q_optimizer = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=cfg.lr)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=cfg.lr)

    @torch.no_grad()
    def update_normalizers(
        self, actor_obs: torch.Tensor, critic_obs: torch.Tensor
    ) -> None:
        if not self.config.freeze_actor_normalizer:
            self.actor_normalizer.update(self.actor_features(actor_obs))
        self.critic_normalizer.update(critic_obs)

    @torch.no_grad()
    def act(self, actor_obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action = self.actor(self.actor_normalizer(self.actor_features(actor_obs)), deterministic)[0]
        return self.action_projector(actor_obs, action) if self.action_projector else action

    def _sample_projected_policy(self, raw_obs: torch.Tensor, normalized_obs: torch.Tensor):
        if self.action_projector is None:
            action, logp = self.actor(normalized_obs)
            active_dims = torch.full_like(logp, self.action_dim)
            return action, logp, active_dims
        action, logp_by_dim = self.actor(normalized_obs, return_per_dim=True)
        mask = self.action_projector.entropy_mask(raw_obs)
        return (self.action_projector(raw_obs, action),
                (logp_by_dim * mask).sum(-1), mask.sum(-1))

    def pretrain_actor(
        self, actor_obs: torch.Tensor, action: torch.Tensor, *,
        steps: int, batch_size: int, update_normalizer: bool = True,
    ) -> dict[str, float]:
        """Fit the deployable actor before online rollout; never train Q on old rewards."""
        if steps < 0 or batch_size < 1 or actor_obs.shape != (len(action), self.actor_obs_dim) \
                or action.shape != (len(action), self.action_dim) or not len(action):
            raise ValueError("Invalid actor demonstration pretraining batch")
        if not bool(torch.isfinite(actor_obs).all()) or not bool(torch.isfinite(action).all()):
            raise ValueError("Demonstration actor inputs must be finite")
        if not steps:
            return {"steps": 0, "initial_mse": 0.0, "final_mse": 0.0}
        features = self.actor_features(actor_obs)
        if update_normalizer:
            self.actor_normalizer.update(features)
        normalized = self.actor_normalizer(features)
        with torch.no_grad():
            initial = F.mse_loss(self.actor(normalized, deterministic=True)[0], action).item()
        # Initialization can fit at the standard rate; online policy changes
        # use the smaller configured actor rate after this pass.
        rates = [group["lr"] for group in self.actor_optimizer.param_groups]
        try:
            for group in self.actor_optimizer.param_groups:
                group["lr"] = self.config.lr
            for _ in range(steps):
                ids = torch.randint(len(action), (batch_size,), device=action.device)
                predicted = self.actor(normalized[ids], deterministic=True)[0]
                loss = F.mse_loss(predicted, action[ids])
                optimize(self.actor_optimizer, loss, self.actor.parameters())
        finally:
            for group, rate in zip(self.actor_optimizer.param_groups, rates):
                group["lr"] = rate
        with torch.no_grad():
            final = F.mse_loss(self.actor(normalized, deterministic=True)[0], action).item()
        return {"steps": steps, "initial_mse": initial, "final_mse": final}

    def update(self, batch: dict[str, torch.Tensor], *,
               demonstration: dict[str, torch.Tensor] | None = None,
               demonstration_weight: float = 0.0, update_actor: bool = True) -> dict[str, float]:
        if demonstration_weight < 0 or (demonstration_weight and demonstration is None):
            raise ValueError("Demonstration weight requires a nonnegative value and a batch")
        cfg = self.config
        actor_obs = self.actor_normalizer(self.actor_features(batch["actor_obs"]))
        critic_obs = self.critic_normalizer(batch["critic_obs"])
        next_actor_obs = self.actor_normalizer(self.actor_features(batch["next_actor_obs"]))
        next_critic_obs = self.critic_normalizer(batch["next_critic_obs"])
        alpha = self.log_alpha.exp().detach()

        with torch.no_grad():
            next_action, next_logp, _ = self._sample_projected_policy(
                batch["next_actor_obs"], next_actor_obs)
            target_features = torch.cat((next_critic_obs, next_action), dim=-1)
            next_q = torch.minimum(
                self.target1(target_features), self.target2(target_features)
            ).squeeze(-1)
            target = soft_target(
                cfg.reward_scale * batch["reward"], batch["terminated"], next_q, next_logp,
                alpha if cfg.entropy_backup else torch.zeros_like(alpha), cfg.gamma)

        replay_features = torch.cat((critic_obs, batch["action"]), dim=-1)
        q1 = self.q1(replay_features).squeeze(-1)
        q2 = self.q2(replay_features).squeeze(-1)
        q_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        optimize(
            self.q_optimizer, q_loss,
            list(self.q1.parameters()) + list(self.q2.parameters()))

        with torch.no_grad():
            for source, target_network in ((self.q1, self.target1), (self.q2, self.target2)):
                for parameter, target_parameter in zip(source.parameters(), target_network.parameters()):
                    target_parameter.lerp_(parameter, cfg.tau)
        if not update_actor:
            with torch.no_grad():
                log_std = self.actor.network(actor_obs).chunk(2, -1)[1].clamp(-5, self.actor.log_std_max)
            return {
                "q_loss": q_loss.item(), "actor_loss": 0.0, "actor_updated": False,
                "demo_bc_loss": 0.0, "demo_bc_weight": 0.0,
                "alpha": alpha.item(), "policy_logp_mean": next_logp.mean().item(),
                "policy_action_std_mean": next_action.std(0, unbiased=False).mean().item(),
                "q_value_mean": torch.minimum(q1, q2).mean().item(),
                "target_value_mean": target.mean().item(),
                "entropy_bonus_mean": (-alpha * next_logp).mean().item(),
                "policy_gaussian_std_mean": log_std.exp().mean().item(),
            }

        self.q1.requires_grad_(False)
        self.q2.requires_grad_(False)
        try:
            action, logp, active_dims = self._sample_projected_policy(
                batch["actor_obs"], actor_obs)
            policy_features = torch.cat((critic_obs, action), dim=-1)
            q = torch.minimum(
                self.q1(policy_features), self.q2(policy_features)).squeeze(-1)
            actor_loss = (alpha * logp - q).mean()
            bc_loss = torch.zeros((), device=actor_loss.device)
            if demonstration is not None and demonstration_weight:
                demo_obs = self.actor_normalizer(self.actor_features(demonstration["actor_obs"]))
                demo_action = self.actor(demo_obs, deterministic=True)[0]
                bc_loss = F.mse_loss(demo_action, demonstration["action"])
                actor_loss = actor_loss + demonstration_weight * bc_loss
            optimize(self.actor_optimizer, actor_loss, self.actor.parameters())
        finally:
            self.q1.requires_grad_(True)
            self.q2.requires_grad_(True)

        target_entropy = active_dims * self.target_entropy_per_dim
        alpha_loss = -(self.log_alpha * (logp.detach() + target_entropy)).mean()
        optimize(self.alpha_optimizer, alpha_loss, [self.log_alpha])
        with torch.no_grad():
            if cfg.min_alpha > 0:
                self.log_alpha.clamp_(min=math.log(cfg.min_alpha))
            log_std = self.actor.network(actor_obs).chunk(2, -1)[1].clamp(-5, self.actor.log_std_max)
        return {
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
            "actor_updated": True,
            "demo_bc_loss": bc_loss.item(),
            "demo_bc_weight": demonstration_weight,
            "alpha": self.log_alpha.exp().item(),
            "policy_logp_mean": logp.detach().mean().item(),
            "policy_action_std_mean": action.detach().std(dim=0, unbiased=False).mean().item(),
            "q_value_mean": q.detach().mean().item(),
            "target_value_mean": target.detach().mean().item(),
            "entropy_bonus_mean": (-alpha * logp.detach()).mean().item(),
            "policy_gaussian_std_mean": log_std.exp().mean().item(),
            "target_entropy_per_dim": self.target_entropy_per_dim,
            "active_entropy_dims_mean": active_dims.mean().item(),
            "policy_entropy_error_mean": (-logp.detach() - target_entropy).mean().item(),
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
            "action_projection": (
                self.action_projector.name if self.action_projector is not None else "none"),
            "entropy_contract": {
                "name": "std_cap_feasible_active_dims_v1",
                "target_per_dim": self.target_entropy_per_dim,
            },
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
        projection_name = (
            self.action_projector.name if self.action_projector is not None else "none")
        if state.get("action_projection", "none") != projection_name:
            raise ValueError("Asymmetric SAC action projection differs from checkpoint")
        self.load_state_dict(state["model"])
        if training:
            for optimizer, saved in zip(self.optimizers, state["optimizers"]):
                optimizer.load_state_dict(saved)
