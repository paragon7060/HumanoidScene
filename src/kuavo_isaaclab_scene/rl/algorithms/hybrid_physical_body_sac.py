"""Hybrid SAC whose Q and success imitation use literal executed body commands."""
import math
import torch
from torch.nn import functional as F

from .hybrid_goal_sac import HybridGoalSAC


class HybridPhysicalBodySAC(HybridGoalSAC):
    frozen_jaw_parameters=None

    def validate_critic_auxiliary(self,batch,weight):
        if not math.isfinite(weight) or weight<0 or (weight and batch is None):
            raise ValueError('Native multi-step critic requires finite nonnegative weight and real rows')
        if batch is None:return
        n=len(batch['reward'])
        shapes=dict(actor_obs=(n,self.actor_obs_dim),next_actor_obs=(n,self.actor_obs_dim),
            critic_obs=(n,self.critic_obs_dim),next_critic_obs=(n,self.critic_obs_dim),
            action=(n,21),reward=(n,),terminated=(n,),bootstrap_discount=(n,),n_steps=(n,))
        if not n or set(batch)!=set(shapes) or any(
                batch[k].shape!=shape or not torch.isfinite(batch[k]).all() for k,shape in shapes.items()) \
                or batch['terminated'].dtype!=torch.bool or batch['n_steps'].dtype!=torch.int64 \
                or (batch['n_steps']<1).any() or (batch['n_steps']>900).any() \
                or (batch['action'].abs()>1.00001).any() or not (batch['action'][:,19:].abs()==1).all():
            raise ValueError('Malformed literal physical multi-step critic rows')
        expected=self.config.gamma**batch['n_steps'].to(batch['reward'])
        expected=expected.masked_fill(batch['terminated'],0.)
        if not torch.allclose(batch['bootstrap_discount'],expected,atol=1e-7,rtol=1e-6):
            raise ValueError('Native multi-step discount must match its actual horizon and terminal')

    def parameters_at(self,normalized):
        mean,log_std=self.actor.network(normalized).chunk(2,-1)
        if self.frozen_jaw_parameters is None:
            raise ValueError('Physical residual SAC requires its frozen actor-only jaw baseline')
        with torch.no_grad():reference_raw,reference_effective=self.frozen_jaw_parameters(normalized)
        logits=reference_effective+20*(mean[:,19:21]-reference_raw)
        return mean[:,:19],log_std[:,:19].clamp(self.actor.log_std_min,self.actor.log_std_max),logits

    def success_body_loss(self,raw,requested_body,labels):
        # Native labels are physical commands. They are not hypothetical
        # inverse residuals; compare the policy's projected command directly.
        physical=self.action_projector(raw,torch.cat((requested_body,labels[:,19:21]),-1))
        return F.mse_loss(physical[:,:19],labels[:,:19])

    def checkpoint(self):
        return super().checkpoint()|dict(algorithm='hybrid_physical_body_sac')

    def restore(self,state,training=True):
        if state.get('algorithm')!='hybrid_physical_body_sac':
            raise ValueError('Physical-body Q cannot restore a goal-action algorithm')
        return super().restore(state|dict(algorithm='hybrid_goal_sac'),training=training)
