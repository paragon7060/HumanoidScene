"""Temporally coherent off-policy collection, with independent episode noise.

SAC's target/actor distributions remain its ordinary Gaussian policy. This
optional behavior sampler changes only actual TRAIN collection. Replay stores
the projected goal that executed, not the latent noise or a teacher action.
"""
import math

import torch


class CorrelatedGoalExploration:
    def __init__(self, correlation, num_envs, action_dim, device):
        if not math.isfinite(correlation) or not 0<=correlation<=.995:
            raise ValueError('Goal exploration correlation must be within0..0.995')
        if num_envs<1 or action_dim<1:
            raise ValueError('Positive environment/action counts required')
        self.correlation=correlation
        self.noise=torch.zeros(num_envs,action_dim,device=device)
        self.initialized=torch.zeros(num_envs,dtype=torch.bool,device=device)

    def _validate_ids(self, ids):
        if ids.ndim!=1 or ids.dtype!=torch.long or len(ids.unique())!=len(ids) \
                or (ids<0).any() or (ids>=len(self.noise)).any():
            raise ValueError('Distinct valid global environment IDs required')

    def sample_noise(self, ids):
        self._validate_ids(ids)
        innovation=torch.randn_like(self.noise[ids])
        value=self.correlation*self.noise[ids]+math.sqrt(1-self.correlation**2)*innovation
        # Start each episode in the stationary N(0,1) marginal. There is no
        # accidental initial ramp in variance or shared noise across hands.
        value=torch.where(self.initialized[ids,None],value,innovation)
        self.noise[ids]=value;self.initialized[ids]=True
        return value

    @torch.no_grad()
    def act(self, agent, observation, ids, *, body_latent_offset=None, jaw_behavior=None,
            greedy_mask=None):
        if greedy_mask is not None:
            self._validate_ids(ids)
            if (not isinstance(greedy_mask,torch.Tensor)
                    or greedy_mask.dtype != torch.bool or greedy_mask.shape != (len(ids),)
                    or greedy_mask.device != observation.device or len(observation) != len(ids)):
                raise ValueError('One boolean collection mode per measured active environment required')
            # No random jaw draw or latent offset affects greedy episodes.
            # Both paths return the same projected executed-goal coordinates.
            action = agent.act(observation, deterministic=True)
            exploring = ~greedy_mask
            if bool(exploring.any()):
                offset = None if body_latent_offset is None else body_latent_offset[exploring]
                action[exploring] = self.act(agent, observation[exploring], ids[exploring],
                    body_latent_offset=offset, jaw_behavior=jaw_behavior)
            return action
        if hasattr(agent,'act_with_latent_noise'):
            options={} if body_latent_offset is None else dict(body_latent_offset=body_latent_offset)
            if jaw_behavior is not None:
                return jaw_behavior.act(agent,observation,self.sample_noise(ids),**options)
            return agent.act_with_latent_noise(observation,self.sample_noise(ids),**options)
        if jaw_behavior is not None:
            raise ValueError('Joint jaw behavior requires an explicit hybrid goal agent')
        if body_latent_offset is not None:
            raise ValueError('Episode-arm behavior requires an explicit hybrid goal agent')
        normalized=agent.actor_normalizer(agent.actor_features(observation))
        mean,log_std=agent.actor.network(normalized).chunk(2,-1)
        std=log_std.clamp(agent.actor.log_std_min,agent.actor.log_std_max).exp()
        action=(mean+std*self.sample_noise(ids)).tanh()
        return agent.action_projector(observation,action) if agent.action_projector else action
