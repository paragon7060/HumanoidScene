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

    def body_from_latent(self, normalized, body, raw=None):
        """Map a squashed body sample once, before physical jaw projection."""
        return body

    def body_log_probability(self, normalized, per_dim, raw=None):
        return per_dim.sum(-1)

    def body_entropy_target(self, normalized, per_dim, raw=None):
        return per_dim.sum(-1)

    def continuous_sample(self, normalized, *, deterministic=False,noise=None,body_latent_offset=None,raw=None):
        mean,log_std,logits=self.parameters_at(normalized)
        if body_latent_offset is not None:
            if body_latent_offset.shape!=mean.shape or not torch.isfinite(body_latent_offset).all():
                raise ValueError('Behavior offset needs one finite19-D latent vector per measured state')
            mean=mean+body_latent_offset
        std=log_std.exp()
        latent=mean if deterministic else mean+std*(torch.randn_like(mean) if noise is None else noise)
        correction=2*(math.log(2)-latent-F.softplus(-2*latent))
        logp=self.body_log_probability(normalized,gaussian_log_prob(latent,mean,std)-correction,raw)
        return self.body_from_latent(normalized,latent.tanh(),raw),logp,logits

    def projected_command(self, observation,body,closed):
        return self.action_projector(observation,torch.cat((body,closed.to(body)*2-1),-1))

    @torch.no_grad()
    def act(self, actor_obs, deterministic=False):
        normalized=self.actor_normalizer(self.actor_features(actor_obs))
        body,_,logits=self.continuous_sample(normalized,deterministic=deterministic,raw=actor_obs)
        closed=logits>0 if deterministic else torch.rand_like(logits)<logits.sigmoid()
        return self.projected_command(actor_obs,body,closed)

    @torch.no_grad()
    def act_with_latent_noise(self, actor_obs,noise,*,body_latent_offset=None):
        if noise.shape!=(len(actor_obs),21):raise ValueError('One21-D behavior noise per environment required')
        normalized=self.actor_normalizer(self.actor_features(actor_obs))
        body,_,logits=self.continuous_sample(normalized,noise=noise[:,:19],body_latent_offset=body_latent_offset,raw=actor_obs)
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

    def continuous_entropy_target(self,normalized,raw=None):
        mean,_,_=self.parameters_at(normalized)
        result=torch.full_like(mean,self.target_entropy_per_dim)
        if self.config.max_policy_std<=1:
            result+=2*(math.log(2)-mean-F.softplus(-2*mean))-self.config.max_policy_std**2
        return self.body_entropy_target(normalized,result,raw)

    def success_body_loss(self,raw,requested_body,labels):
        return F.mse_loss(requested_body,labels[:,:19])

    def successful_jaw_loss(self, raw, logits, labels):
        near = self.action_projector.entropy_mask(raw)[:,19:21].bool()
        loss = (F.binary_cross_entropy_with_logits(logits[near], (labels[:,19:21][near]+1)/2)
                if bool(near.any()) else logits.new_zeros(()))
        return loss, {}

    def actor_jaw_regularization(self, logits, near):
        """Optional subclass objective; ordinary hybrid SAC stays unchanged."""
        return None

    def actor_body_regularization(self, normalized, raw):
        """Optional subclass objective; default adds no loss or extra forward."""
        return None

    def validate_critic_auxiliary(self,batch,weight):
        if batch is not None or weight:
            raise ValueError('This hybrid learner does not support auxiliary critic targets')

    @torch.no_grad()
    def critic_target(self,batch,alpha,discrete_alpha,*,bootstrap_discount=None):
        # Terminal placeholders must never enter a domain-constrained decoder.
        bootstrap=~batch['terminated'].bool()
        value=torch.zeros_like(batch['reward'])
        statistics=dict(bootstrapped_rows=int(bootstrap.sum()),policy_logp_mean=0.,
                        policy_action_std_mean=0.,entropy_bonus_mean=0.)
        if bool(bootstrap.any()):
            next_actor=batch['next_actor_obs'][bootstrap]
            na=self.actor_normalizer(self.actor_features(next_actor))
            nc=self.critic_normalizer(batch['next_critic_obs'][bootstrap])
            body,continuous_logp,logits=self.continuous_sample(na,raw=next_actor)
            actions,probability,discrete_logp,_=self.enumerate_jaws(next_actor,body,logits)
            q=self.branch_values(nc,actions,target=True)
            entropy=alpha*continuous_logp[:,None]+discrete_alpha*discrete_logp
            value[bootstrap]=(probability*(q-entropy if self.config.entropy_backup else q)).sum(-1)
            statistics.update(policy_logp_mean=continuous_logp.mean().item(),
                policy_action_std_mean=body.std(0,unbiased=False).mean().item(),
                entropy_bonus_mean=(-(probability*entropy).sum(-1)).mean().item())
        discount=self.config.gamma if bootstrap_discount is None else bootstrap_discount
        return self.config.reward_scale*batch['reward']+discount*value,statistics

    def update(self,batch,*,demonstration=None,demonstration_weight=0.,
               teacher=None,teacher_weight=0.,update_actor=True,
               successful_train=None,success_goal_weight=0.,success_jaw_weight=0.,
               critic_auxiliary=None,critic_auxiliary_weight=0.):
        self.validate_critic_auxiliary(critic_auxiliary,critic_auxiliary_weight)
        if demonstration is not None or demonstration_weight:
            raise ValueError('Hybrid goals require actual TRAIN replay, not old delta demonstrations')
        if teacher_weight<0 or (teacher_weight and teacher is None):
            raise ValueError('Nonnegative prior weight and labels required')
        if not all(math.isfinite(v) and v>=0 for v in (success_goal_weight,success_jaw_weight)) or (
                (success_goal_weight or success_jaw_weight) and successful_train is None):
            raise ValueError('Successful TRAIN labels require nonnegative retention weights')
        if successful_train is not None:
            labels=successful_train['action']
            if labels.ndim!=2 or labels.shape[1]!=21 or not torch.isfinite(labels).all() \
                    or (labels.abs()>1.00001).any() or not (labels[:,19:21].abs()==1).all() \
                    or successful_train['actor_obs'].shape!=(len(labels),self.actor_obs_dim) \
                    or not torch.isfinite(successful_train['actor_obs']).all():
                raise ValueError('Successful TRAIN labels must retain real bounded goals and binary jaws')
        if not bool((batch['action'][:,19:21].abs()==1).all()):
            raise ValueError('Hybrid replay must contain actual binary gripper commands')
        ao=self.actor_normalizer(self.actor_features(batch['actor_obs']))
        co=self.critic_normalizer(batch['critic_obs'])
        alpha=self.log_alpha.exp().detach();discrete_alpha=self.log_alpha_discrete.exp().detach()
        target,target_statistics=self.critic_target(batch,alpha,discrete_alpha)
        replay=torch.cat((co,batch['action']),-1)
        q1=self.q1(replay).squeeze(-1);q2=self.q2(replay).squeeze(-1)
        one_step_q_loss=F.mse_loss(q1,target)+F.mse_loss(q2,target)
        q_loss=one_step_q_loss
        auxiliary_statistics={}
        if critic_auxiliary is not None and critic_auxiliary_weight:
            auxiliary_target,aux_stats=self.critic_target(critic_auxiliary,alpha,discrete_alpha,
                bootstrap_discount=critic_auxiliary['bootstrap_discount'])
            auxiliary_features=torch.cat((self.critic_normalizer(critic_auxiliary['critic_obs']),
                                          critic_auxiliary['action']),-1)
            auxiliary_loss=(F.mse_loss(self.q1(auxiliary_features).squeeze(-1),auxiliary_target)+
                            F.mse_loss(self.q2(auxiliary_features).squeeze(-1),auxiliary_target))
            q_loss=q_loss+critic_auxiliary_weight*auxiliary_loss
            auxiliary_statistics=dict(one_step_q_loss=one_step_q_loss.item(),
                native_nstep_q_loss=auxiliary_loss.item(),native_nstep_weight=critic_auxiliary_weight,
                native_nstep_rows=len(auxiliary_target),native_nstep_target_mean=auxiliary_target.mean().item(),
                native_nstep_bootstrapped_rows=aux_stats['bootstrapped_rows'],
                native_nstep_terminal_rows=int(critic_auxiliary['terminated'].sum()),
                native_nstep_horizon_mean=critic_auxiliary['n_steps'].float().mean().item())
        optimize(self.q_optimizer,q_loss,[*self.q1.parameters(),*self.q2.parameters()])
        with torch.no_grad():
            for a,b in ((self.q1,self.target1),(self.q2,self.target2)):
                for p,t in zip(a.parameters(),b.parameters()):t.lerp_(p,self.config.tau)
        report=dict(q_loss=q_loss.item(),actor_loss=0.,actor_updated=False,
            demo_bc_loss=0.,demo_bc_weight=0.,teacher_bc_loss=0.,teacher_bc_weight=0.,
            discrete_prior_loss=0.,discrete_prior_weight=0.,
            alpha=alpha.item(),alpha_discrete=discrete_alpha.item(),
            **target_statistics,
            q_value_mean=torch.minimum(q1,q2).mean().item(),target_value_mean=target.mean().item(),
            policy_gaussian_std_mean=self.parameters_at(ao)[1].exp().mean().item(),
            success_goal_loss=0.,success_jaw_loss=0.,success_goal_weight=0.,success_jaw_weight=0.)
        report.update(auxiliary_statistics)
        if not update_actor:return report
        self.q1.requires_grad_(False);self.q2.requires_grad_(False)
        try:
            body,continuous_logp,logits=self.continuous_sample(ao,raw=batch['actor_obs'])
            actions,probability,discrete_logp,near=self.enumerate_jaws(batch['actor_obs'],body,logits)
            q=self.branch_values(co,actions)
            scale=q.detach().abs().mean().clamp_min(1).reciprocal() if self.config.actor_q_normalize else 1.
            actor_loss=(probability*(alpha*continuous_logp[:,None]+discrete_alpha*discrete_logp-scale*q)).sum(-1).mean()
            teacher_loss=torch.zeros((),device=ao.device);discrete_prior=torch.zeros_like(teacher_loss)
            success_goal_loss=torch.zeros_like(teacher_loss);success_jaw_loss=torch.zeros_like(teacher_loss)
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
            if successful_train is not None and (success_goal_weight or success_jaw_weight):
                raw=successful_train['actor_obs'];normalized=self.actor_normalizer(self.actor_features(raw))
                mean,_,jaw_logits=self.parameters_at(normalized);labels=successful_train['action']
                success_goal_loss=self.success_body_loss(raw,mean.tanh(),labels)
                success_jaw_loss, success_jaw_statistics = self.successful_jaw_loss(raw, jaw_logits, labels)
                report.update(success_jaw_statistics)
                actor_loss+=success_goal_weight*success_goal_loss+success_jaw_weight*success_jaw_loss
            regularization = self.actor_jaw_regularization(logits, near)
            if regularization is not None:
                extra_loss, extra_statistics = regularization
                actor_loss = actor_loss+extra_loss
                report.update(extra_statistics)
            body_regularization = self.actor_body_regularization(ao, batch['actor_obs'])
            if body_regularization is not None:
                extra_loss, extra_statistics = body_regularization
                actor_loss = actor_loss+extra_loss
                report.update(extra_statistics)
            optimize(self.actor_optimizer,actor_loss,self.actor.parameters())
        finally:
            self.q1.requires_grad_(True);self.q2.requires_grad_(True)
        target_entropy=self.continuous_entropy_target(ao,raw=batch['actor_obs']).detach()
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
            close_probability_mean=logits.sigmoid().mean().item(),
            success_goal_loss=success_goal_loss.item(),success_jaw_loss=success_jaw_loss.item(),
            success_goal_weight=success_goal_weight,success_jaw_weight=success_jaw_weight)
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
