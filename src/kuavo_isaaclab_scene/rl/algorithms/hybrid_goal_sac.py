"""SAC for 19 continuous goals and two genuinely binary grippers.

At each measured state, the two masked Bernoulli policies enumerate their
four Q actions. Enumeration is a policy expectation, never fabricated replay
or simulator transitions. Far-away jaws have only the open action and earn
no discrete entropy. Collection and Q use the actual -1/+1 jaw commands.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .asymmetric_sac import AsymmetricSAC
from .common import gaussian_log_prob,optimize


class HybridGoalSAC(AsymmetricSAC):
    continuous_dims=19
    discrete_entropy_target_per_jaw=.35
    discrete_prior_weight=.05

    def __init__(self,*args,validated_jaw_prior_confidence=0.,jaw_prior_residual_gain=1.,**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(validated_jaw_prior_confidence) or not (
                validated_jaw_prior_confidence==0 or .5<validated_jaw_prior_confidence<1) \
                or not math.isfinite(jaw_prior_residual_gain) or jaw_prior_residual_gain<=0:
            raise ValueError('Jaw prior needs confidence within(0.5,1) and positive residual gain')
        self.validated_jaw_prior_confidence=validated_jaw_prior_confidence
        self.jaw_prior_residual_gain=jaw_prior_residual_gain
        # The pilot owns/checkpoints this frozen actor. A callable avoids
        # registering another copy in this module's state/optimizer tree.
        self.validated_jaw_prior=None
        if self.action_dim!=21 or self.action_projector is None or not self.action_projector.free_grippers:
            raise ValueError('Hybrid goal SAC needs19 goals and two independently projected grippers')
        self.log_alpha_discrete=nn.Parameter(torch.tensor(math.log(.01),device=self.log_alpha.device))
        self.discrete_alpha_optimizer=torch.optim.Adam([self.log_alpha_discrete],lr=self.config.lr)

    def parameters_at(self, normalized):
        mean,log_std=self.actor.network(normalized).chunk(2,-1)
        logits=mean[:,19:21]
        if self.validated_jaw_prior_confidence:
            if self.validated_jaw_prior is None:
                raise ValueError('Confident jaw policy requires its frozen validated actor')
            with torch.no_grad():reference=self.validated_jaw_prior(normalized)
            prior=self.prior_jaw_logits(normalized,reference=reference)
            logits=prior+self.jaw_prior_residual_gain*(logits-reference)
        return mean[:,:19],log_std[:,:19].clamp(self.actor.log_std_min,self.actor.log_std_max),logits

    def prior_jaw_logits(self,normalized,reference=None):
        if self.validated_jaw_prior is None:
            raise ValueError('Confident jaw policy requires its frozen validated actor')
        if reference is None:
            with torch.no_grad():reference=self.validated_jaw_prior(normalized)
        confidence=self.validated_jaw_prior_confidence
        magnitude=math.log(confidence/(1-confidence))
        return torch.where(reference>0,magnitude,-magnitude)

    def continuous_sample(self, normalized, *, deterministic=False,noise=None):
        mean,log_std,logits=self.parameters_at(normalized)
        std=log_std.exp()
        latent=mean if deterministic else mean+std*(torch.randn_like(mean) if noise is None else noise)
        correction=2*(math.log(2)-latent-F.softplus(-2*latent))
        logp=(gaussian_log_prob(latent,mean,std)-correction).sum(-1)
        return latent.tanh(),logp,logits

    def projected_command(self, observation,body,closed):
        return self.action_projector(observation,torch.cat((body,closed.to(body)*2-1),-1))

    @torch.no_grad()
    def act(self, actor_obs, deterministic=False):
        normalized=self.actor_normalizer(self.actor_features(actor_obs))
        body,_,logits=self.continuous_sample(normalized,deterministic=deterministic)
        closed=logits>0 if deterministic else torch.rand_like(logits)<logits.sigmoid()
        return self.projected_command(actor_obs,body,closed)

    @torch.no_grad()
    def act_with_latent_noise(self, actor_obs,noise):
        if noise.shape!=(len(actor_obs),21):raise ValueError('One21-D behavior noise per environment required')
        normalized=self.actor_normalizer(self.actor_features(actor_obs))
        body,_,logits=self.continuous_sample(normalized,noise=noise[:,:19])
        # Normal CDF gives stationary uniform marginals (and Bernoulli(p) at
        # a fixed state). Feedback-correlated collection remains off-policy;
        # target/actor expectations use the independent categorical policy.
        uniform=.5*(1+torch.erf(noise[:,19:21]/math.sqrt(2)))
        return self.projected_command(actor_obs,body,uniform<logits.sigmoid())

    def enumerate_jaws(self,raw,body,logits):
        n=len(raw)
        bits=torch.tensor([[0,0],[0,1],[1,0],[1,1]],device=raw.device,dtype=torch.bool)
        near=self.action_projector.entropy_mask(raw)[:,19:21].bool()
        closed_probability=logits.sigmoid();open_probability=(-logits).sigmoid()
        probability=torch.where(bits[None],closed_probability[:,None],open_probability[:,None])
        probability=torch.where(near[:,None],probability,(~bits)[None].to(probability))
        weights=probability.prod(-1)
        logs=torch.where(bits[None],F.logsigmoid(logits)[:,None],F.logsigmoid(-logits)[:,None])
        logp=(logs*near[:,None]).sum(-1)
        requested=torch.cat((body[:,None].expand(-1,4,-1),
                             (bits.to(body)*2-1)[None].expand(n,-1,-1)),-1)
        observations=raw[:,None].expand(-1,4,-1).reshape(n*4,-1)
        projected=self.action_projector(observations,requested.reshape(n*4,21)).reshape(n,4,21)
        return projected,weights,logp,near

    def branch_values(self,critic,actions,target=False):
        n=len(critic);features=torch.cat((critic[:,None].expand(-1,4,-1),actions),-1).reshape(n*4,-1)
        a,b=(self.target1,self.target2) if target else (self.q1,self.q2)
        return torch.minimum(a(features),b(features)).reshape(n,4)

    def continuous_entropy_target(self,normalized):
        mean,_,_=self.parameters_at(normalized)
        result=torch.full_like(mean,self.target_entropy_per_dim)
        if self.config.max_policy_std<=1:
            result+=2*(math.log(2)-mean-F.softplus(-2*mean))-self.config.max_policy_std**2
        return result.sum(-1)

    def update(self,batch,*,demonstration=None,demonstration_weight=0.,
               teacher=None,teacher_weight=0.,update_actor=True):
        if demonstration is not None or demonstration_weight:
            raise ValueError('Hybrid goals require actual TRAIN replay, not old delta demonstrations')
        if teacher_weight<0 or (teacher_weight and teacher is None):
            raise ValueError('Nonnegative prior weight and labels required')
        if not bool((batch['action'][:,19:21].abs()==1).all()):
            raise ValueError('Hybrid replay must contain actual binary gripper commands')
        ao=self.actor_normalizer(self.actor_features(batch['actor_obs']))
        co=self.critic_normalizer(batch['critic_obs'])
        na=self.actor_normalizer(self.actor_features(batch['next_actor_obs']))
        nc=self.critic_normalizer(batch['next_critic_obs'])
        alpha=self.log_alpha.exp().detach();discrete_alpha=self.log_alpha_discrete.exp().detach()
        with torch.no_grad():
            body,continuous_logp,logits=self.continuous_sample(na)
            actions,probability,discrete_logp,_=self.enumerate_jaws(batch['next_actor_obs'],body,logits)
            q=self.branch_values(nc,actions,target=True)
            entropy=alpha*continuous_logp[:,None]+discrete_alpha*discrete_logp
            next_value=(probability*(q-entropy if self.config.entropy_backup else q)).sum(-1)
            target=self.config.reward_scale*batch['reward']+self.config.gamma*torch.where(
                batch['terminated'],torch.zeros_like(next_value),next_value)
        replay=torch.cat((co,batch['action']),-1)
        q1=self.q1(replay).squeeze(-1);q2=self.q2(replay).squeeze(-1)
        q_loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        optimize(self.q_optimizer,q_loss,[*self.q1.parameters(),*self.q2.parameters()])
        with torch.no_grad():
            for a,b in ((self.q1,self.target1),(self.q2,self.target2)):
                for p,t in zip(a.parameters(),b.parameters()):t.lerp_(p,self.config.tau)
        report=dict(q_loss=q_loss.item(),actor_loss=0.,actor_updated=False,
            demo_bc_loss=0.,demo_bc_weight=0.,teacher_bc_loss=0.,teacher_bc_weight=0.,
            discrete_prior_loss=0.,discrete_prior_weight=0.,
            alpha=alpha.item(),alpha_discrete=discrete_alpha.item(),
            policy_logp_mean=continuous_logp.mean().item(),
            policy_action_std_mean=body.std(0,unbiased=False).mean().item(),
            q_value_mean=torch.minimum(q1,q2).mean().item(),target_value_mean=target.mean().item(),
            entropy_bonus_mean=(-(probability*entropy).sum(-1)).mean().item(),
            policy_gaussian_std_mean=self.parameters_at(ao)[1].exp().mean().item())
        if not update_actor:return report
        self.q1.requires_grad_(False);self.q2.requires_grad_(False)
        try:
            body,continuous_logp,logits=self.continuous_sample(ao)
            actions,probability,discrete_logp,near=self.enumerate_jaws(batch['actor_obs'],body,logits)
            q=self.branch_values(co,actions)
            scale=q.detach().abs().mean().clamp_min(1).reciprocal() if self.config.actor_q_normalize else 1.
            actor_loss=(probability*(alpha*continuous_logp[:,None]+discrete_alpha*discrete_logp-scale*q)).sum(-1).mean()
            teacher_loss=torch.zeros((),device=ao.device);discrete_prior=torch.zeros_like(teacher_loss)
            if teacher is not None and teacher_weight:
                normalized=self.actor_normalizer(self.actor_features(teacher['actor_obs']))
                mean,_,jaw_logits=self.parameters_at(normalized)
                labels=teacher['action']
                teacher_loss=F.mse_loss(mean.tanh(),labels[:,:19])
                teacher_logits=(self.prior_jaw_logits(normalized) if self.validated_jaw_prior_confidence
                    else labels[:,19:21].clamp(-.999999,.999999).atanh())
                p=teacher_logits.sigmoid()
                discrete_prior=(p*(F.logsigmoid(teacher_logits)-F.logsigmoid(jaw_logits))+
                    (1-p)*(F.logsigmoid(-teacher_logits)-F.logsigmoid(-jaw_logits))).mean()
                actor_loss+=teacher_weight*teacher_loss+self.discrete_prior_weight*discrete_prior
            optimize(self.actor_optimizer,actor_loss,self.actor.parameters())
        finally:
            self.q1.requires_grad_(True);self.q2.requires_grad_(True)
        target_entropy=self.continuous_entropy_target(ao).detach()
        optimize(self.alpha_optimizer,-(self.log_alpha*(continuous_logp.detach()+target_entropy)).mean(),[self.log_alpha])
        observed_entropy=-(probability*discrete_logp).sum(-1).detach()
        target_discrete=self.discrete_entropy_target_per_jaw*near.sum(-1)
        optimize(self.discrete_alpha_optimizer,
            (self.log_alpha_discrete*(observed_entropy-target_discrete)).mean(),[self.log_alpha_discrete])
        with torch.no_grad():
            if self.config.min_alpha>0:self.log_alpha.clamp_(min=math.log(self.config.min_alpha))
            if math.isfinite(self.config.max_alpha):self.log_alpha.clamp_(max=math.log(self.config.max_alpha))
            self.log_alpha_discrete.clamp_(max=math.log(.1))
        report.update(actor_loss=actor_loss.item(),actor_updated=True,
            teacher_bc_loss=teacher_loss.item(),teacher_bc_weight=teacher_weight,
            discrete_prior_loss=discrete_prior.item(),alpha=self.log_alpha.exp().item(),
            discrete_prior_weight=self.discrete_prior_weight if teacher is not None and teacher_weight else 0.,
            alpha_discrete=self.log_alpha_discrete.exp().item(),
            discrete_entropy_mean=observed_entropy.mean().item(),
            target_discrete_entropy_mean=target_discrete.float().mean().item(),
            near_jaw_count_mean=near.sum(-1).float().mean().item(),
            close_probability_mean=logits.sigmoid().mean().item())
        return report

    @property
    def optimizers(self):
        return (*super().optimizers,self.discrete_alpha_optimizer)

    def checkpoint(self):
        state=super().checkpoint();state['algorithm']='hybrid_goal_sac'
        state['hybrid_contract']=self.hybrid_contract
        return state

    @property
    def hybrid_contract(self):
        result=dict(continuous_dims=19,binary_jaws=2,exact_branches=4,
            discrete_entropy_target_per_jaw=self.discrete_entropy_target_per_jaw,
            discrete_prior_weight=self.discrete_prior_weight)
        if self.validated_jaw_prior_confidence:
            result.update(validated_jaw_prior_confidence=self.validated_jaw_prior_confidence,
                jaw_prior_residual_gain=self.jaw_prior_residual_gain,
                jaw_policy='confident_validated_binary_prior_plus_trainable_logit_residual_v1')
        return result

    def restore(self,state,training=True):
        if state.get('algorithm')!='hybrid_goal_sac' or state.get('hybrid_contract')!=self.hybrid_contract:
            raise ValueError('Hybrid SAC cannot import a continuous-only Q/optimizer')
        if training and len(state.get('optimizers',[]))!=4:
            raise ValueError('Hybrid SAC requires both entropy optimizer states')
        super().restore(state,training)
