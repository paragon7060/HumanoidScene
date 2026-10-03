"""SAC continuation of a physically tested goal student, without a live path.

Q actions are normalized absolute goals, never physical delta commands. Only
measured transitions whose inverse goals reproduce the executed command may
seed Q. Initial box anchor and episode clock are explicit Markov context.
"""
from dataclasses import replace
import math
from pathlib import Path

import torch

from ...algorithms.asymmetric_sac import AsymmetricReplayBuffer, AsymmetricSAC
from ...algorithms.sac import SACConfig
from ...runners.storage import save_checkpoint
from .executed_replay import read_executed_successes,merge_executed_successes
from .pose_student import PoseGoalCoordinates, PoseStudent


def reward_discount(contract):
    """Use the same discount for Bellman backups and potential shaping."""
    discount = float(contract['discount'])
    if not math.isfinite(discount) or not 0 < discount < 1:
        raise ValueError('Invalid physical reward discount')
    shaped = contract.get('reward_profile', {}).get('weights', {}).get('discount', discount)
    if float(shaped) != discount:
        raise ValueError('Reward potential and physical discounts differ')
    return discount


class GoalGripperProjector:
    name='v2_pose_goal_assigned_flap_close_gate_0p12m'

    def __init__(self, prior=None, max_prior_deviation=0.):
        if (not math.isfinite(max_prior_deviation) or not 0<=max_prior_deviation<=1
                or (max_prior_deviation and prior is None)):
            raise ValueError('A bounded goal policy needs a frozen neural prior and radius in0..1')
        self.prior,self.max_prior_deviation=prior,max_prior_deviation
        if max_prior_deviation:
            self.name += '_neural_prior_radius_'+format(max_prior_deviation,'.8g')

    def near(self, observation):
        # ActorFeatures places the selected box at86:108, then the38 relations.
        relation=observation[:,108:144].reshape(-1,2,2,9)
        assignment=observation[:,144:146]
        left=assignment.argmax(-1)
        flap=torch.stack((left,1-left),-1)
        rows=torch.arange(len(observation),device=observation.device)[:,None]
        hands=torch.arange(2,device=observation.device)[None]
        distance=relation[rows,hands,flap,:3].norm(dim=-1)
        return (distance<=.12)&(assignment.sum(-1)>.5)[:,None]

    def __call__(self, observation, action):
        result=action.clone()
        if self.max_prior_deviation:
            with torch.no_grad():
                reference=self.prior.agent.act(observation[:,:-2],deterministic=True)
            result=torch.maximum(reference-self.max_prior_deviation,
                                 torch.minimum(result,reference+self.max_prior_deviation)).clamp(-1,1)
        result[:,22:24]=torch.where(self.near(observation),result[:,22:24],-1.)
        return result

    def entropy_mask(self, observation):
        result=observation.new_ones(len(observation),24)
        result[:,22:24]=self.near(observation)
        return result


def widen_bc_actor(student, agent):
    """Add explicit anchor context without changing the initial mean policy."""
    with torch.no_grad():
        old=student.agent.actor.state_dict();new=agent.actor.state_dict()
        for key,value in old.items():
            if key=='network.0.weight':
                new[key].zero_();new[key][:,:value.shape[1]].copy_(value)
            else:new[key].copy_(value)
        agent.actor.load_state_dict(new)
        n=student.agent.actor_obs_dim
        agent.actor_normalizer.mean[:n].copy_(student.agent.actor_normalizer.mean)
        agent.actor_normalizer.var[:n].copy_(student.agent.actor_normalizer.var)
        agent.actor_normalizer.count.copy_(student.agent.actor_normalizer.count)


class PoseGoalSACPilot:
    artifact_type='pose_goal_sac_no_live_reference'

    def __init__(self, checkpoint, dataset, physical_contract, directory, *,
                 training=True, device='cpu'):
        self.device,self.training=device,training
        self.directory=Path(directory)
        self.coordinates=PoseGoalCoordinates()
        saved=torch.load(checkpoint,map_location=device,weights_only=True)
        if saved.get('artifact_type')==self.artifact_type and saved.get('format_version')!=1:
            raise ValueError('Goal SAC checkpoint format differs')
        prior=saved.get('bc_prior',saved)
        self.prior=PoseStudent(prior,device)
        self.coordinates=self.prior.coordinates
        self.prior.validate_physical_contract(physical_contract)
        self.prior.agent.requires_grad_(False)
        self.harmonics=prior.get('time_harmonics',0)
        self.clock_horizon=prior.get('clock_horizon',410)
        if not prior.get('initial_box_relative_goals'):
            raise ValueError('Goal SAC requires the tested initial-box-relative student')
        self.center,self.scale=self.prior.center,self.prior.scale
        self.actor_dim=prior['actor_obs_dim']+2
        self.anchor=None
        self.actor_updates=self.critic_updates=self.online_rows=0
        legacy=saved.get('artifact_type')==self.artifact_type
        self.fade_updates=int(saved.get('goal_contract',{}).get('demo_fade_updates',4000 if legacy else 20000))
        self.prior_weight=float(saved.get('goal_contract',{}).get('frozen_network_prior_initial_weight',10.))
        self.prior_floor=float(saved.get('goal_contract',{}).get('frozen_network_prior_weight_floor',0. if legacy else 10.))
        self.prior_radius=float(saved.get('goal_contract',{}).get('frozen_network_prior_radius',0. if legacy else .05))
        if (self.fade_updates<1 or not all(math.isfinite(v) for v in (self.prior_weight,self.prior_floor,self.prior_radius))
                or not 0<=self.prior_floor<=self.prior_weight or not 0<=self.prior_radius<=1):
            raise ValueError('Invalid goal warm-start schedule or neural prior bounds')
        self.latest={};self.online_history=[]
        self.reward_discount=reward_discount(physical_contract)
        self.discount_alignment=saved.get('discount_alignment')
        self.discount_warmup_updates=0
        config=replace(self.prior.agent.config,actor_lr=1e-6,min_policy_std=.0001,
                       initial_policy_std=.001,max_policy_std=.003,gamma=self.reward_discount)
        if saved.get('artifact_type')==self.artifact_type:
            config=SACConfig(**saved['config'])
        self.agent=AsymmetricSAC(self.actor_dim,533,24,config,device,
                                action_projector=GoalGripperProjector(self.prior,self.prior_radius))
        self.replay=AsymmetricReplayBuffer(20000,self.actor_dim,533,24,device)
        measured,audit=(merge_executed_successes(dataset,physical_contract)
                        if isinstance(dataset,(list,tuple)) else
                        read_executed_successes(dataset,physical_contract))
        self.audit=audit
        measured={key:value.to(device) for key,value in measured.items()}
        ends=torch.where(measured['terminated'].reshape(-1).bool())[0].cpu().tolist()
        self.seed_episodes=audit['successful_episodes']
        if not ends or len(ends)!=self.seed_episodes or ends[-1]!=len(measured['action'])-1:
            raise ValueError('Goal seeds require complete separate actual successful episodes')
        starts={0,*(end+1 for end in ends[:-1])}
        seed=[]
        for i in range(len(measured['action'])):
            raw=measured['actor_obs'][i:i+1]
            if i in starts:
                seed_anchor=self.coordinates.box_anchor(raw)
                episode_index=0
            goal=self.coordinates.encode_physical(raw,measured['action'][i:i+1])
            goal[:,19:21]-=seed_anchor
            z=(goal-self.center)/self.scale
            if not torch.isfinite(z).all() or (z.abs()>1.00001).any():
                raise ValueError('Actual seed goal exceeds the student goal bounds')
            ao,co=self.observations(raw,measured['critic_obs'][i:i+1],episode_index,seed_anchor)
            na,nc=self.observations(measured['next_actor_obs'][i:i+1],
                measured['next_critic_obs'][i:i+1],episode_index+1,seed_anchor)
            # These are measured OFF-policy commands. A new neural prior bound
            # applies to generated actions, never to historical Q action labels.
            z=GoalGripperProjector()(ao,z)
            if not torch.allclose(self.physical(raw,z,seed_anchor),measured['action'][i:i+1],atol=1e-5,rtol=0):
                raise ValueError('Goal decoding/projection does not reproduce the executed seed command')
            seed.append(dict(actor_obs=ao,critic_obs=co,action=z,
                next_actor_obs=na,next_critic_obs=nc,
                reward=measured['reward'][i:i+1],terminated=measured['terminated'][i:i+1]))
            episode_index+=1
        self.seed={key:torch.cat([row[key] for row in seed]) for key in seed[0]}
        if saved.get('artifact_type')==self.artifact_type:
            if saved.get('goal_contract')!=self.contract:
                raise ValueError('Goal SAC context, bounds or seed physical contract differs')
            self.agent.restore(saved,training=training)
            self.actor_updates=saved['actor_updates'];self.critic_updates=saved['critic_updates']
            previous=Path(checkpoint).parent/'pose_goal_experience.pt'
            if training and previous.exists():
                data=torch.load(previous,map_location=device,weights_only=True)
                if data.get('goal_contract')!=self.contract:raise ValueError('Goal replay contract differs')
                rows=data['executed_goal_transitions']
                for key,storage in self.replay.data.items():
                    if rows[key].shape[1:]!=storage.shape[1:] or not torch.isfinite(rows[key]).all():
                        raise ValueError('Malformed executed goal replay')
                if (rows['action'].abs()>1.00001).any():raise ValueError('Unbounded executed goal action')
                self.replay.add(**rows)
                self.online_history.append({key:value.cpu() for key,value in rows.items()})
            warmup=saved.get('pending_discount_critic_warmup',0)
            if warmup and training:
                if (not isinstance(warmup,int) or not 1<=warmup<=10000
                        or config.gamma!=self.reward_discount or not self.replay.size):
                    raise ValueError('Discount migration requires matching gamma and actual replay')
                # Revalue measured transitions at the new horizon. The actor,
                # its optimizer and both observation normalizers stay fixed.
                for _ in range(warmup):
                    actual=self.replay.sample(205,self.device)
                    seed=self.sample_seed(51)
                    batch={key:torch.cat((value,seed[key])) for key,value in actual.items()}
                    self.latest=self.agent.update(batch,update_actor=False)
                    self.critic_updates+=1;self.discount_warmup_updates+=1
        elif saved.get('artifact_type')==PoseStudent.artifact_type:
            widen_bc_actor(self.prior,self.agent)
            with torch.no_grad():
                self.agent.actor.network[-1].weight[24:].zero_()
                self.agent.actor.network[-1].bias[24:].fill_(math.log(config.initial_policy_std))
            self.agent.critic_normalizer.update(self.seed['critic_obs'])
            self.agent.critic_normalizer.var.clamp_(min=.01)
            if training:
                for _ in range(500):
                    self.latest=self.agent.update(self.sample_seed(256),update_actor=False)
                    self.critic_updates+=1
        else:raise ValueError('Not a BC goal warm start or a goal SAC checkpoint')

    @property
    def contract(self):
        contract=dict(name='initial_box_relative_absolute_pose_goal_sac_v1',
            action_coordinates=self.coordinates.name,actor_dim=self.actor_dim,critic_dim=533,
            time_harmonics=self.harmonics,goal_center=self.center.tolist(),goal_scale=self.scale.tolist(),
            source_sha256=self.audit['source_dataset_sha256'],runtime_reference_path_required=False,
            initial_box_anchor_in_observation=True,projection=self.agent.action_projector.name,
            actor_lr=self.agent.config.actor_lr,demo_fraction_initial=.2,demo_fade_updates=self.fade_updates,
            frozen_network_prior_initial_weight=self.prior_weight,prior_fade_updates=self.fade_updates)
        # Legacy410-step checkpoints retain their exact context contract.
        if 'clock_horizon' in self.prior.state:contract['clock_horizon']=self.clock_horizon
        if self.prior.state.get('shelf_conditioned_clock_fit'):
            contract['shelf_conditioning']='perceived_selected_box_shelf; no_dynamic_pose_feedback_in_prior'
        if 'actor_clock_limit' in self.prior.state:
            contract['actor_clock_limit']=self.prior.state['actor_clock_limit']
        # Exact legacy resume remains possible. New constraints are explicit
        # policy-only migrations; the physical decoder/reward are unchanged.
        if self.prior_floor:contract['frozen_network_prior_weight_floor']=self.prior_floor
        if self.prior_radius:contract['frozen_network_prior_radius']=self.prior_radius
        if self.seed_episodes>1:contract['seed_episode_context']='separate_clock_and_initial_box_anchor'
        return contract

    def prior_weight_at(self, updates):
        return max(self.prior_floor,self.prior_weight*max(0.,1-updates/self.fade_updates))

    def observations(self,raw,critic,index,anchor):
        features=self.coordinates.observations(raw,index,self.harmonics,self.clock_horizon,
            condition_on_shelf=self.prior.state.get('shelf_conditioned_clock_fit',False),
            clock_limit=self.prior.state.get('actor_clock_limit'))
        clock=raw.new_full((len(raw),1),min(index,self.clock_horizon)/self.clock_horizon)
        return torch.cat((features,anchor.expand(len(raw),-1)),-1),torch.cat((critic,clock,anchor.expand(len(raw),-1)),-1)

    def physical(self,raw,z,anchor):
        goal=self.center+self.scale*z
        goal=goal.clone();goal[:,19:21]+=anchor
        return self.coordinates.decode(raw,goal)

    def sample_seed(self,count):
        ids=torch.randint(len(self.seed['reward']),(count,),device=self.device)
        return {key:value[ids] for key,value in self.seed.items()}

    @torch.no_grad()
    def act(self,raw,critic,index):
        if self.anchor is None:self.anchor=self.coordinates.box_anchor(raw).clone()
        ao,co=self.observations(raw,critic,index,self.anchor)
        z=self.agent.act(ao,deterministic=not self.training or self.online_rows<64)
        return self.physical(raw,z,self.anchor),(ao.detach(),co.detach(),z.detach())

    def observe(self,previous,next_raw,next_critic,reward,terminated,index):
        if not self.training:return
        ao,co,z=previous
        na,nc=self.observations(next_raw,next_critic,index+1,self.anchor)
        batch=dict(actor_obs=ao,critic_obs=co,action=z,reward=reward.detach(),
            next_actor_obs=na.detach(),next_critic_obs=nc.detach(),terminated=terminated.detach())
        self.replay.add(**batch)
        self.online_history.append({key:value.cpu() for key,value in batch.items()})
        self.online_rows+=len(ao)
        if self.online_rows<64:return
        with torch.enable_grad():
            for _ in range(2):
                fade=max(0.,1-self.actor_updates/self.fade_updates)
                demo_count=round(256*.2*fade)
                actual=self.replay.sample(256-demo_count,self.device)
                if demo_count:
                    seed=self.sample_seed(demo_count)
                    actual={key:torch.cat((value,seed[key])) for key,value in actual.items()}
                # A frozen neural-network prior on current state is actor-only;
                # it provides no recorded path, reward or hypothetical Q row.
                with torch.no_grad():
                    prior_action=self.prior.agent.act(actual['actor_obs'][:,:-2],deterministic=True)
                self.latest=self.agent.update(actual,
                    teacher=dict(actor_obs=actual['actor_obs'],action=prior_action),
                    teacher_weight=self.prior_weight_at(self.actor_updates))
                self.actor_updates+=1;self.critic_updates+=1

    def report(self):
        return dict(training=self.training,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            online_rows=self.online_rows,seed_rows=len(self.seed['reward']),
            seed_episodes=self.seed_episodes,
            runtime_reference_path_required=False,demo_fraction=.2*max(0.,1-self.actor_updates/self.fade_updates),
            frozen_network_prior_weight=self.prior_weight_at(self.actor_updates),
            frozen_network_prior_radius=self.prior_radius,
            min_policy_std=self.agent.config.min_policy_std,max_policy_std=self.agent.config.max_policy_std,
            learner_discount=self.agent.config.gamma,reward_discount=self.reward_discount,
            discount_mismatch=self.agent.config.gamma!=self.reward_discount,
            discount_warmup_updates=self.discount_warmup_updates,discount_alignment=self.discount_alignment,
            replay_size=self.replay.size,latest=self.latest,goal_contract=self.contract)

    def save(self,final=False):
        state=self.agent.checkpoint()|dict(artifact_type=self.artifact_type,goal_contract=self.contract,
            bc_prior=self.prior.state,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            action_coordinates=self.coordinates.name,reference_runtime_dependency=False)
        if self.discount_alignment:
            state['discount_alignment']=self.discount_alignment
        save_checkpoint(self.directory,state,self.actor_updates,keep=None)
        if final and self.online_history:
            rows={key:torch.cat([batch[key] for batch in self.online_history])[-20000:] for key in self.replay.data}
            torch.save(dict(goal_contract=self.contract,executed_goal_transitions=rows),
                       self.directory/'pose_goal_experience.pt')
