"""State-conditioned DDPM action chunks, with explicit stochastic transition densities.

The RL sampler keeps *unclipped* Gaussian latent transitions in its chain. Only
the environment action is clipped. This makes stored/recomputed log densities
agree, including the last denoising step, whose variance has a positive floor.
"""

from dataclasses import dataclass, asdict
import math
import torch
from torch import nn
from torch.nn import functional as F
from .common import mlp, ObservationNormalizer, gaussian_log_prob


@dataclass
class DiffusionConfig:
    horizon: int = 4
    denoising_steps: int = 20
    hidden: int = 256
    min_std: float = .1


class DiffusionPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, config=None):
        super().__init__()
        self.config = config or DiffusionConfig()
        cfg = self.config
        if min(cfg.horizon, cfg.denoising_steps, cfg.hidden) < 1 or not 0 < cfg.min_std <= 1:
            raise ValueError("Invalid diffusion dimensions or noise floor")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.normalizer = ObservationNormalizer(obs_dim)
        self.network = mlp(obs_dim + cfg.horizon * action_dim + 32, cfg.horizon * action_dim, cfg.hidden)
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, -math.log(10000), 16)))
        grid = torch.linspace(0, cfg.denoising_steps, cfg.denoising_steps + 1, dtype=torch.float64)
        cumulative = torch.cos(((grid / cfg.denoising_steps + .008) / 1.008) * math.pi / 2).square()
        betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(.0001, .999).float()
        alphas = 1 - betas
        abar = alphas.cumprod(0)
        previous = torch.cat((torch.ones(1), abar[:-1]))
        self.register_buffer("abar", abar)
        self.register_buffer("coef_x0", betas * previous.sqrt() / (1 - abar))
        self.register_buffer("coef_xt", (1 - previous) * alphas.sqrt() / (1 - abar))
        self.register_buffer("std", (betas * (1 - previous) / (1 - abar)).sqrt().clamp_min(cfg.min_std))

    def predict_noise(self, obs, x, t):
        time = t[:, None] * self.frequencies[None]
        features = torch.cat((self.normalizer(obs), x.flatten(1), time.sin(), time.cos()), -1)
        return self.network(features).reshape_as(x)

    def transition(self, obs, x, t):
        abar = self.abar[t, None, None]
        x0 = ((x - (1 - abar).sqrt() * self.predict_noise(obs, x, t)) / abar.sqrt()).clamp(-1, 1)
        mean = self.coef_x0[t, None, None] * x0 + self.coef_xt[t, None, None] * x
        return mean, self.std[t, None, None]

    def transition_log_prob(self, obs, previous, following, t):
        mean, std = self.transition(obs, previous, t)
        # Joint likelihood of the full auxiliary action chunk (not a Gaussian fit to its output).
        return gaussian_log_prob(following, mean, std).sum((-1, -2))

    @torch.no_grad()
    def sample(self, obs, deterministic=False):
        cfg = self.config
        shape = (len(obs), cfg.horizon, self.action_dim)
        # Reproducible mean-path evaluation; training uses an independent Gaussian prior.
        x = torch.zeros(shape, device=obs.device) if deterministic else torch.randn(shape, device=obs.device)
        chain, logps = [x], []
        for step in reversed(range(cfg.denoising_steps)):
            t = torch.full((len(obs),), step, device=obs.device, dtype=torch.long)
            mean, std = self.transition(obs, x, t)
            x = mean if deterministic else mean + std * torch.randn_like(x)
            logps.append(gaussian_log_prob(x, mean, std).sum((-1, -2)))
            chain.append(x)
        return x.clamp(-1, 1), torch.stack(chain, 1), torch.stack(logps, 1)

    def loss(self, obs, actions, t=None, noise=None):
        if t is None:
            t = torch.randint(self.config.denoising_steps, (len(obs),), device=obs.device)
        if noise is None:
            noise = torch.randn_like(actions)
        abar = self.abar[t, None, None]
        x = abar.sqrt() * actions + (1 - abar).sqrt() * noise
        return F.mse_loss(self.predict_noise(obs, x, t), noise)

    def checkpoint(self):
        return dict(algorithm="diffusion_bc", diffusion_config=asdict(self.config),
                    obs_dim=self.obs_dim, action_dim=self.action_dim, policy=self.state_dict())


def load_diffusion(state, device="cpu"):
    if state.get("algorithm") not in ("diffusion_bc", "dppo"):
        raise ValueError("Expected a native diffusion_bc/dppo checkpoint; arbitrary LeRobot weights are not compatible")
    policy = DiffusionPolicy(state["obs_dim"], state["action_dim"], DiffusionConfig(**state["diffusion_config"]))
    policy.load_state_dict(state["policy"], strict=True)
    return policy.to(device)
