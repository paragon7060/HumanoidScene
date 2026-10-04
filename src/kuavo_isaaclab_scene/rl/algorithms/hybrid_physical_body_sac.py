"""Hybrid SAC whose Q and success imitation use literal executed body commands."""
import torch
from torch.nn import functional as F

from .hybrid_goal_sac import HybridGoalSAC


class HybridPhysicalBodySAC(HybridGoalSAC):
    frozen_jaw_parameters=None

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
