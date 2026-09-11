"""DDPM DPPO variant: PPO on individual transitions of the denoising MDP.

See https://arxiv.org/abs/2409.00588 and irom-princeton/dppo. This independent
implementation uses joint Gaussian log likelihoods without density clipping,
all denoising steps trainable, and receding-horizon execution of one action.
"""

from dataclasses import dataclass, asdict
import math
import torch
from torch.nn import functional as F
from .common import mlp, optimize


@dataclass
class DPPOConfig:
    actor_lr: float = 1e-5
    critic_lr: float = 3e-4
    gamma: float = .99
    gae_lambda: float = .95
    gamma_denoising: float = .99
    clip_base: float = .001
    clip_final: float = .1
    epochs: int = 5
    minibatch_size: int = 2048
    target_kl: float = .02
    critic_warmup: int = 5


def clipped_objective(new_logp, old_logp, advantage, epsilon):
    logratio = new_logp - old_logp
    # Far outside the trust region is rejected by the caller, not silently clipped.
    ratio = logratio.exp()
    loss = -torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage).mean()
    kl = ((ratio - 1) - logratio).mean()
    return loss, kl


class DPPO:
    def __init__(self, policy, config=None):
        self.policy, self.config = policy, config or DPPOConfig()
        self.device = next(policy.parameters()).device
        self.critic = mlp(policy.obs_dim, 1, policy.config.hidden).to(self.device)
        self.actor_optimizer = torch.optim.Adam(policy.network.parameters(), lr=self.config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.config.critic_lr)

    def value(self, obs):
        # Observation statistics stay frozen from BC throughout fine-tuning.
        return self.critic(self.policy.normalizer(obs)).squeeze(-1)

    def update(self, obs, chains, old_logps, advantages, returns, update_actor=True):
        """One uniformly sampled denoising transition per state per epoch.

        This is an unbiased minibatch estimate of the mean over denoising steps,
        avoiding K-fold replication of the 223-dimensional observation tensor.
        """
        cfg = self.config
        count, steps = len(obs), self.policy.config.denoising_steps
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
        metrics = dict(actor_loss=0.0, value_loss=0.0, kl=0.0, actor_updates=0)
        actor_stopped = not update_actor
        for _ in range(cfg.epochs):
            order = torch.randperm(count, device=self.device)
            for ids in order.split(cfg.minibatch_size):
                value_loss = .5 * F.mse_loss(self.value(obs[ids]), returns[ids])
                optimize(self.critic_optimizer, value_loss, self.critic.parameters())
                metrics["value_loss"] = value_loss.item()
                if actor_stopped:
                    continue
                denoise_ids = torch.randint(steps, (len(ids),), device=self.device)
                t = steps - 1 - denoise_ids
                new_logp = self.policy.transition_log_prob(
                    obs[ids], chains[ids, denoise_ids], chains[ids, denoise_ids + 1], t)
                old_logp = old_logps[ids, denoise_ids]
                progress = denoise_ids.float() / max(steps - 1, 1)
                epsilon = cfg.clip_base + (cfg.clip_final - cfg.clip_base) * torch.expm1(3 * progress) / math.expm1(3)
                if steps == 1:
                    epsilon.fill_(cfg.clip_final)
                discount = cfg.gamma_denoising ** t.float()
                logratio = new_logp - old_logp
                if not torch.isfinite(logratio).all() or logratio.abs().max() > 20:
                    actor_stopped = True
                    metrics["trust_region_stopped"] = True
                    continue
                actor_loss, kl = clipped_objective(new_logp, old_logp, advantages[ids] * discount, epsilon)
                metrics["kl"] = kl.item()
                if kl.item() > cfg.target_kl:
                    actor_stopped = True
                    continue
                optimize(self.actor_optimizer, actor_loss, self.policy.network.parameters())
                metrics["actor_loss"] = actor_loss.item()
                metrics["actor_updates"] += 1
        return metrics

    def checkpoint(self):
        state = self.policy.checkpoint()
        state.update(algorithm="dppo", dppo_config=asdict(self.config), critic=self.critic.state_dict(),
                     actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict())
        return state

    def restore(self, state):
        self.critic.load_state_dict(state["critic"])
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
