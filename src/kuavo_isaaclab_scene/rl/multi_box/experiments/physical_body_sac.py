"""Held-base SAC learning directly in executed physical body coordinates.

Source goal policies contribute frozen actors only. Native TRAIN commands can
seed this fresh Q after checking physical/base feedback and measured outcomes.
Old goal Q, inverse goal labels, DEV/FINAL and changed-physics probes cannot.
"""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path

import torch

from ...algorithms.asymmetric_sac import AsymmetricReplayBuffer
from ...algorithms.hybrid_physical_body_sac import HybridPhysicalBodySAC
from ...algorithms.sac import SACConfig
from ...runners.storage import save_checkpoint
from .executed_replay import PHYSICAL_KEYS
from .pose_goal_sac import reward_discount
from .physical_body_actions import (
    FrozenGoalCommandPrior,PhysicalBodyProjector,PHYSICAL_COLUMNS,actor_only_snapshot,full_command,
)
from .staged_goal_sac import StagedGoalSACPilot
from .staged_physics import require_current_lift_contract
from .staged_train_success import TrainSuccessBank,FORMAT as GOAL_BANK_FORMAT

BANK_FORMAT='actual_TRAIN_success_held_physical_body_transitions_v1'
REPLAY_FILE='physical_body_experience.pt'


def physical_signature(contract):
    require_current_lift_contract(contract)
    result={key:deepcopy(contract.get(key)) for key in (*PHYSICAL_KEYS,'flap_pose_source')}
    if 'physics_dynamics' in contract:result['physics_dynamics']=deepcopy(contract['physics_dynamics'])
    return result


class PhysicalTrainSuccessBank(TrainSuccessBank):
    def report(self):
        return super().report()|dict(action_coordinates='literal_physical_body21')

    def state(self):
        return super().state()|dict(format=BANK_FORMAT,action_coordinates='literal_physical_body21')

    def restore(self,state):
        if state.get('format')!=BANK_FORMAT or state.get('action_coordinates')!='literal_physical_body21':
            raise ValueError('Physical success replay cannot import requested-goal labels')
        super().restore(state|dict(format=GOAL_BANK_FORMAT))


class PhysicalBodySACPilot(StagedGoalSACPilot):
    artifact_type='staged_held_physical_body_hybrid_sac_v1'
    agent_class=HybridPhysicalBodySAC

    def __init__(self,warm_start,physical_contract,directory,stage,*,checkpoint=None,
                 frozen_goal_state=None,training=True,device='cpu'):
        if warm_start.training:raise ValueError('The controller prior must remain frozen')
        self.warm_start,self.coordinates=warm_start,warm_start.coordinates
        self.physical_contract=physical_signature(physical_contract)
        self.directory,self.device,self.training,self.stage=Path(directory),device,training,stage
        self.anchor=None;self.actor_dim=480;self.critic_dim=539
        self.replay_capacity=100000;self.warmup=1024;self.fade=20000
        self.actor_updates=self.critic_updates=self.online_rows=0
        self.success_schedule_actor_origin=0.;self.latest={};self.latest_actor={};self.history=[]
        self.exploration_correlation=.98;self.episode_arm_exploration=None
        self.goal_exploration=self.arm_behavior=None
        self.seed_provenance=None
        saved=torch.load(checkpoint,map_location=device,weights_only=True) if checkpoint else None
        if saved:
            if saved.get('artifact_type')!=self.artifact_type:
                raise ValueError('Physical body SAC cannot restore goal Q or an unrelated checkpoint')
            snapshot=saved['frozen_goal_actor'];self.frozen_warm_start=saved['frozen_warm_start']
        else:
            if frozen_goal_state is None:raise ValueError('An explicit frozen source actor is required')
            snapshot=(deepcopy(frozen_goal_state['frozen_goal_actor']) if
                      frozen_goal_state.get('artifact_type')=='frozen_goal_actor_and_pose_prior_inputs_v1'
                      else actor_only_snapshot(frozen_goal_state))
            self.frozen_warm_start=deepcopy(frozen_goal_state['frozen_warm_start'])
        self.frozen_goal_actor=deepcopy(snapshot)
        source_contract=snapshot['goal_contract']
        if source_contract['physical_contract']!=self.physical_contract \
                or source_contract['shelf_templates']!=stage.templates or source_contract['waypoint_format']!=stage.name:
            raise ValueError('Physical/held-waypoint contract differs from the frozen source actor')
        self.command_prior=FrozenGoalCommandPrior(snapshot,warm_start,device)
        config=replace(warm_start.agent.config,actor_lr=3e-5,initial_policy_std=.05,
            min_policy_std=.02,max_policy_std=.1,gamma=reward_discount(physical_contract),
            freeze_actor_normalizer=True,actor_feature_mode='flat',critic_layer_norm=True,actor_q_normalize=True)
        if saved and saved['config']!=asdict(config):raise ValueError('Physical body learner configuration differs')
        self.agent=self.agent_class(self.actor_dim,self.critic_dim,21,config,device,
            action_projector=PhysicalBodyProjector(self.command_prior))
        self.agent.frozen_jaw_parameters=self.command_prior.jaw_parameters
        self.success_bank=PhysicalTrainSuccessBank(self.actor_dim,self.critic_dim)
        self.replay=AsymmetricReplayBuffer(self.replay_capacity,self.actor_dim,self.critic_dim,21,device)
        if saved:
            if saved['physical_body_contract']!=self.contract:raise ValueError('Physical body action/reward context differs')
            self.agent.restore(saved,training=training)
            self.actor_updates=saved['actor_updates'];self.critic_updates=saved['critic_updates']
            self.latest_actor=saved.get('latest_actor_metrics',{})
            self.seed_provenance=deepcopy(saved.get('native_TRAIN_seed_provenance'))
            experience=Path(checkpoint).parent/REPLAY_FILE
            if training:
                if not experience.is_file():raise ValueError('Physical body continuation requires its literal replay')
                data=torch.load(experience,map_location=device,weights_only=True)
                if data.get('physical_body_contract')!=self.contract:
                    raise ValueError('Physical replay cannot restore old goal or altered-MDP data')
                self.add_native_rows(data['executed_physical_body_transitions'])
                self.success_bank.restore(data['successful_train_transitions'])
            else:self.success_bank.restore(saved['successful_train_transitions'])
        else:
            self.agent.actor.load_state_dict(self.command_prior.actor.state_dict())
            self.agent.actor_normalizer.load_state_dict(self.command_prior.normalizer.state_dict())
            with torch.no_grad():
                self.agent.actor_normalizer.mean[-1]+=.5-.05
                last=self.agent.actor.network[-1]
                last.weight[:19].zero_();last.bias[:19].zero_()
                last.weight[21:40].zero_();last.bias[21:40].fill_(torch.log(torch.tensor(config.initial_policy_std)))
        if not training:self.agent.requires_grad_(False)

    @property
    def radius(self):return .5

    @property
    def contract(self):
        return dict(name=self.artifact_type,action_coordinates='literal_executed_physical_body21',
            actor_dim=480,critic_dim=539,physical_columns=list(PHYSICAL_COLUMNS),
            context_order=['held_phase','held_x_rack_m','held_y_rack_m','sin_held_yaw','cos_held_yaw','physical_residual_gain'],
            physical_contract=self.physical_contract,waypoint_format=self.stage.name,shelf_templates=self.stage.templates,
            residual_gain=.5,replay_capacity=self.replay_capacity,initial_critic_warmup=self.warmup,
            actor_update_interval=4,fresh_Q=True,old_goal_Q_and_optimizer_imported=False,
            native_seed_actions='literal_recorded_commands_after_validating_unchanged_held_base_feedback',
            gripper_policy='two_masked_Bernoulli_exact_four_action_expectation',
            body_success_loss='projected_physical_commands_not_inverse_residual_labels',
            residual_entropy='pre_projection_latent_policy',exploration_correlation=.98,
            exploration_std_initial=.05,exploration_std_min=.02,exploration_std_cap=.1,
            success_bank_format=BANK_FORMAT,success_replay_initial=.2,success_replay_final=.05,
            source_actor_updates=self.frozen_goal_actor['source_actor_updates'],
            source_goal_controller_contract=self.frozen_goal_actor['goal_contract'],
            runtime_VR_or_IK_teacher=False,box_fixed=False)

    def add_native_rows(self,rows):
        n=len(rows['reward']);shapes=dict(actor_obs=(n,480),critic_obs=(n,539),action=(n,21),
            next_actor_obs=(n,480),next_critic_obs=(n,539),reward=(n,),terminated=(n,))
        if set(rows)!=set(shapes) or n>self.replay.capacity or any(
                rows[k].shape!=shape or not torch.isfinite(rows[k]).all() for k,shape in shapes.items()) \
                or rows['terminated'].dtype!=torch.bool or (rows['action'].abs()>1.00001).any() \
                or not (rows['action'][:,19:].abs()==1).all() \
                or not all((rows[k][:,-6]==1).all() for k in ('actor_obs','critic_obs','next_actor_obs','next_critic_obs')):
            raise ValueError('Malformed literal physical held-body replay')
        self.replay.add(**rows);self.history.append({k:v.detach().cpu().clone() for k,v in rows.items()})

    @torch.no_grad()
    def act(self,raw,critic,index,*,exploration_ids=None):
        if self.anchor is None:self.anchor=self.coordinates.box_anchor(raw).clone()
        ao,co=self.observations(raw,critic,index)
        deterministic=not self.training or self.replay.size<64
        if deterministic:action=self.agent.act(ao,True)
        else:
            if self.goal_exploration is None:self.reset_exploration(len(raw))
            ids=torch.arange(len(raw),device=raw.device) if exploration_ids is None else exploration_ids
            action=self.goal_exploration.act(self.agent,ao,ids)
        return full_command(self.coordinates,ao,action),(ao.detach(),co.detach(),action.detach())

    def observe(self,previous,next_raw,next_critic,reward,terminated,index):
        if not self.training:return
        ao,co,action=previous;na,nc=self.observations(next_raw,next_critic,index+1)
        rows=dict(actor_obs=ao,critic_obs=co,action=action,next_actor_obs=na,next_critic_obs=nc,
            reward=reward.detach(),terminated=terminated.detach())
        self.add_native_rows(rows);self.online_rows+=len(ao)
        if self.replay.size<64:return
        with torch.enable_grad():
            for _ in range(2):
                batch,count=self.success_bank.mix(self.replay.sample(256,self.device),self.success_replay_fraction,self.device)
                self.agent.update_normalizers(batch['actor_obs'],batch['critic_obs'])
                update_actor=self.critic_updates>=self.warmup and self.critic_updates%4==0
                teacher=None;weight=0.
                if update_actor:
                    with torch.no_grad():
                        labels=torch.zeros(len(batch['reward']),21,device=self.device)
                        _,logits=self.command_prior.jaw_parameters(self.agent.actor_normalizer(batch['actor_obs']))
                        labels[:,19:]=logits.tanh()
                        teacher=dict(actor_obs=batch['actor_obs'],action=labels)
                    weight=.2*max(0.,1-self.actor_updates/5000)
                success=self.success_bank.sample(64,self.device) if self.success_bank.size and update_actor else None
                self.latest=self.agent.update(batch,teacher=teacher,teacher_weight=weight,update_actor=update_actor,
                    successful_train=success,success_goal_weight=1. if success else 0.,success_jaw_weight=.05 if success else 0.)
                self.latest.update(successful_train_rows_in_Q_batch=count,body_loss_coordinates='physical_commands')
                if update_actor:self.latest_actor=dict(self.latest,critic_update=self.critic_updates+1)
                self.actor_updates+=int(update_actor);self.critic_updates+=1

    def report(self):
        return dict(training=self.training,actor_updates=self.actor_updates,critic_updates=self.critic_updates,
            online_rows=self.online_rows,replay_size=self.replay.size,action_coordinates='physical_body21',
            fresh_goal_Q_migration=False,old_goal_Q_and_optimizer_imported=False,
            physical_residual_gain=.5,min_policy_std=.02,max_policy_std=.1,
            successful_train_bank=self.success_bank.report(),successful_train_replay_fraction=self.success_replay_fraction,
            native_TRAIN_seed_provenance=self.seed_provenance,latest_actor_metrics=self.latest_actor)

    def save(self,final=False):
        if getattr(self,'_last_saved',None)!=self.critic_updates:
            state=self.agent.checkpoint()|dict(artifact_type=self.artifact_type,physical_body_contract=self.contract,
                frozen_goal_actor=self.frozen_goal_actor,frozen_warm_start=self.frozen_warm_start,
                actor_updates=self.actor_updates,critic_updates=self.critic_updates,latest_actor_metrics=self.latest_actor,
                native_TRAIN_seed_provenance=self.seed_provenance,successful_train_transitions=self.success_bank.state())
            save_checkpoint(self.directory,state,self.critic_updates,keep=None);self._last_saved=self.critic_updates
        if final:
            remaining=self.replay.capacity;recent=[]
            for batch in reversed(self.history):
                n=min(remaining,len(batch['reward']))
                if n:recent.append({k:v[-n:] for k,v in batch.items()})
                remaining-=n
                if not remaining:break
            recent.reverse();rows={k:torch.cat([b[k] for b in recent]).cpu() if recent else v[:0].cpu().clone()
                                  for k,v in self.replay.data.items()}
            data=dict(physical_body_contract=self.contract,executed_physical_body_transitions=rows,
                successful_train_transitions=self.success_bank.state(),native_TRAIN_seed_provenance=self.seed_provenance)
            pending=self.directory/(REPLAY_FILE+'.pending');torch.save(data,pending);pending.replace(self.directory/REPLAY_FILE)
