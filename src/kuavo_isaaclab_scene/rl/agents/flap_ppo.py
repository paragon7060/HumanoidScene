"""Bounded PPO policy for flap pick; observation and task contracts stay unchanged."""

import math
import torch
from torch import nn
from torch.distributions import Normal
from rsl_rl.modules import ActorCritic
from rsl_rl.networks import EmpiricalNormalization
from rsl_rl.algorithms import PPO


class LimitedNormalization(EmpiricalNormalization):
    def forward(self, x):
        # Rare binary/history features must not become 80+ standard deviations.
        return ((x - self._mean) / (self._std.clamp_min(.1) + self.eps)).clamp(-5., 5.)


class FlapActorCritic(ActorCritic):
    """Tanh Gaussian with matching likelihood/entropy and bounded latent noise.

    action_mean/action_std intentionally expose LATENT Gaussian parameters:
    RSL-RL's analytic KL is valid under the shared invertible tanh transform.
    Inference returns tanh(latent mean), not that internal Gaussian mean.
    """
    min_std, max_std = .05, .35

    def __init__(self, obs, obs_groups, num_actions, init_noise_std=.15, **kwargs):
        if kwargs.get('state_dependent_std', False):
            raise ValueError('FlapActorCritic currently uses per-action state-independent noise')
        kwargs['noise_std_type'] = 'log'
        super().__init__(obs, obs_groups, num_actions, init_noise_std=init_noise_std, **kwargs)
        if not self.min_std < init_noise_std < self.max_std:
            raise ValueError('Initial noise must be strictly inside (.05, .35)')
        del self.log_std
        fraction = (init_noise_std-self.min_std)/(self.max_std-self.min_std)
        self.std_logit = nn.Parameter(torch.full((num_actions,), math.log(fraction/(1-fraction))))
        for side in ('actor', 'critic'):
            if getattr(self, side+'_obs_normalization'):
                old = getattr(self, side+'_obs_normalizer')
                setattr(self, side+'_obs_normalizer', LimitedNormalization(old._mean.shape[-1]))
        # Initially hold near the current actuator target, without a gripper bias.
        last = [layer for layer in self.actor.modules() if isinstance(layer, nn.Linear)][-1]
        nn.init.orthogonal_(last.weight, gain=.01)
        nn.init.zeros_(last.bias)

    def _mean(self, obs):
        # Avoid numerical saturation of tanh/atanh while retaining near-full actions.
        return 3. * torch.tanh(self.actor(obs) / 3.)

    def _update_distribution(self, obs):
        mean = self._mean(obs)
        std = self.min_std + (self.max_std-self.min_std)*torch.sigmoid(self.std_logit)
        self.distribution = Normal(mean, std.expand_as(mean))

    def act(self, obs, **kwargs):
        self._update_distribution(self.actor_obs_normalizer(self.get_actor_obs(obs)))
        return torch.tanh(self.distribution.sample())

    def act_inference(self, obs):
        return torch.tanh(self._mean(self.actor_obs_normalizer(self.get_actor_obs(obs))))

    @staticmethod
    def log_jacobian(latent):
        return 2. * (math.log(2.) - latent - torch.nn.functional.softplus(-2.*latent))

    def get_actions_log_prob(self, actions):
        latent = torch.atanh(actions.clamp(-1.+1e-6, 1.-1e-6))
        return (self.distribution.log_prob(latent) - self.log_jacobian(latent)).sum(-1)

    @property
    def entropy(self):
        # Reparameterized Monte Carlo entropy of the executed distribution.
        latent = self.distribution.rsample()
        return (self.distribution.entropy() + self.log_jacobian(latent)).sum(-1)


class FlapPPO(PPO):
    def update(self):
        # Fixed-size sample keeps diagnostics independent of the full rollout size.
        s = self.storage
        with torch.no_grad():
            count = s.num_envs*s.num_transitions_per_env
            ids = torch.linspace(0, count-1, min(count, 4096), device=self.device).long()
            obs = s.observations.flatten(0, 1)[ids].clone()
            actions = s.actions.flatten(0, 1)[ids].clone()
            old_log = s.actions_log_prob.flatten()[ids].clone()
            old_mu = s.mu.flatten(0, 1)[ids].clone()
            old_sigma = s.sigma.flatten(0, 1)[ids].clone()
            returns = s.returns.flatten()[ids].clone()
            values = s.values.flatten()[ids].clone()
        result = super().update()
        with torch.no_grad():
            self.policy.act(obs)
            ratio = (self.policy.get_actions_log_prob(actions)-old_log).exp()
            mu, sigma = self.policy.action_mean, self.policy.action_std
            kl = (torch.log(sigma/old_sigma)+(old_sigma.square()+(old_mu-mu).square())/(2*sigma.square())-.5).sum(-1)
            result.update(policy_kl_sample=kl.mean().item(),
                policy_clip_fraction_sample=((ratio-1).abs() > self.clip_param).float().mean().item(),
                action_near_limit_sample=(actions.abs() > .95).float().mean().item(),
                latent_std_min=sigma.min().item(), latent_std_max=sigma.max().item(),
                explained_variance_sample=(1-(returns-values).var(unbiased=False)/returns.var(unbiased=False).clamp_min(1e-8)).item())
        return result


def configure_flap_ppo(agent):
    # RSL-RL 3.1.2 resolves its configured classes in this module's namespace.
    import rsl_rl.runners.on_policy_runner as runner
    runner.FlapActorCritic = FlapActorCritic
    runner.FlapPPO = FlapPPO
    agent.policy.class_name = 'FlapActorCritic'
    agent.policy.init_noise_std = .15
    agent.policy.noise_std_type = 'log'
    agent.algorithm.class_name = 'FlapPPO'
    agent.algorithm.entropy_coef = .001
    agent.num_steps_per_env = 64
    agent.algorithm.num_mini_batches = 32
    agent.algorithm.num_learning_epochs = 4
