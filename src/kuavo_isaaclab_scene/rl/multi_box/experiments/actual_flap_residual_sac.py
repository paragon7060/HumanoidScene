"""Measured articulated flap perception and learned corrections to a frozen actor.

Only nominal actor snapshots migrate. Q/actions stay absolute executed goals;
new perception and correction coordinates require fresh replay and critics.
"""
from copy import deepcopy
from dataclasses import replace
import math

import torch
from torch.nn import functional as F

from ...algorithms.common import ObservationNormalizer
from ...algorithms.sac import SquashedActor
from ...algorithms.hybrid_goal_sac import HybridGoalSAC
from ..observations.flap_supplement import SUPPLEMENTAL_DIM,supplemental_perception_contract
from .pose_goal_sac import GoalGripperProjector
from .staged_goal_sac import CONTEXT_DIM,StagedGoalProjector
from .staged_hybrid_goal_sac import StagedHybridGoalSACPilot


class AbsoluteGoalJawProjector:
    """Idempotent physical bounds and the unchanged production nominal jaw gate."""
    name='absolute_body_goals_production_near_flap_binary_jaws_v1'
    free_grippers=True

    def __init__(self):self.gate=GoalGripperProjector()

    def __call__(self,observation,action):
        result=action.clamp(-1,1)
        jaws=torch.where(self.gate.near(observation),result[:,19:21],-1.)
        return torch.cat((result[:,:19],jaws),-1)

    def entropy_mask(self,observation):
        return torch.cat((observation.new_ones(len(observation),19),self.gate.near(observation).to(observation)),-1)


class BoundedCorrectionHybridSAC(HybridGoalSAC):
    def critic_action_features(self,raw,actions):
        encoder=getattr(self,'goal_servo_critic_encoder',None)
        return encoder(raw,actions) if encoder is not None else super().critic_action_features(raw,actions)

    def successful_jaw_loss(self, raw, logits, labels):
        config = getattr(self, 'success_jaw_balance_config', None)
        if config is None:
            return super().successful_jaw_loss(raw, logits, labels)
        from .success_jaw_balance import balanced_success_jaw_loss
        near = self.action_projector.entropy_mask(raw)[:,19:21].bool()
        return balanced_success_jaw_loss(logits, near, labels[:,19:21],
            raw[:,94:98].argmax(-1), config)

    def actor_jaw_regularization(self, logits, near):
        config = getattr(self, 'jaw_saturation_config', None)
        if config is None:
            return None
        from .jaw_saturation import jaw_saturation_penalty
        return jaw_saturation_penalty(logits, near, config)

    def actor_body_regularization(self, normalized, raw):
        config = getattr(self, 'body_saturation_config', None)
        if config is None:
            return None
        from .body_saturation import body_saturation_penalty
        mean = self.parameters_at(normalized)[0]
        active = self.anchor_and_scale(raw)[1] > 1e-8
        return body_saturation_penalty(mean, active, config)

    def update(self, *args, **kwargs):
        result = super().update(*args, **kwargs)
        if getattr(self, 'measured_train_credit_enabled', False):
            result = {k.replace('native_nstep_', 'measured_nstep_'): v for k, v in result.items()}
        return result

    def validate_critic_auxiliary(self, batch, weight):
        if not getattr(self, 'measured_train_credit_enabled', False):
            return super().validate_critic_auxiliary(batch, weight)
        from .measured_train_credit import validate_measured_credit_batch
        validate_measured_credit_batch(self, batch, weight)

    def anchor_and_scale(self,raw):
        if raw is None or getattr(self,'executed_body_anchor',None) is None:
            raise ValueError('Bounded correction needs a measured raw state and frozen executed actor')
        with torch.no_grad():anchor=self.executed_body_anchor(raw)
        # Shrink symmetrically at bounds, rather than clip a residual into an
        # unmodelled point mass. Fixed coordinates have neither Q gradient nor
        # continuous entropy. Every other coordinate has an affine Jacobian.
        scale=(1-anchor.abs()).clamp(min=0,max=self.correction_radius)
        return anchor,scale

    def body_from_latent(self,normalized,body,raw=None):
        anchor,scale=self.anchor_and_scale(raw)
        return anchor+scale*body

    def body_log_probability(self,normalized,per_dim,raw=None):
        _,scale=self.anchor_and_scale(raw)
        return ((per_dim-scale.clamp_min(1e-8).log())*(scale>1e-8)).sum(-1)

    def body_entropy_target(self,normalized,per_dim,raw=None):
        _,scale=self.anchor_and_scale(raw)
        return ((per_dim+scale.clamp_min(1e-8).log())*(scale>1e-8)).sum(-1)

    def success_body_loss(self,raw,requested_body,labels):
        physical=self.body_from_latent(self.actor_normalizer(raw),requested_body,raw)
        return F.mse_loss(physical,labels[:,:19])

    @property
    def hybrid_contract(self):
        result=super().hybrid_contract|dict(body_sample_coordinates='zero_centered_bounded_actor_correction_v1',
            Q_action_coordinates='actual_absolute_projected_goals',
            correction_radius=self.correction_radius,
            entropy_jacobian='per_state_symmetric_available_goal_radius_including_inactive_bounds')
        encoder=getattr(self,'goal_servo_critic_encoder',None)
        if encoder is not None:
            result.update(Q_action_coordinates=encoder.contract['critic_action_coordinates'],
                critic_action_encoding=encoder.contract)
        return result


def actor_anchor_state(source):
    """An explicitly actor-only snapshot; never serialize source Q or replay."""
    if source.get('artifact_type')!=StagedHybridGoalSACPilot.artifact_type:
        raise ValueError('Correction initialization requires the nominal staged hybrid actor')
    model={k:v.detach().clone() for k,v in source['model'].items()
           if k.startswith(('actor.','actor_normalizer.'))}
    if not model or 'frozen_actor_prior' not in source:
        raise ValueError('Original confident jaw reference is required')
    return dict(format_version=1,source_artifact_type=source['artifact_type'],
        source_actor_updates=source['actor_updates'],source_critic_updates_not_imported=source['critic_updates'],
        source_prior_schedule_actor_origin=source.get('prior_schedule_actor_origin',0.),
        source_goal_contract=deepcopy(source['goal_contract']),source_hidden=source['config']['hidden'],
        model=model,frozen_actor_prior=deepcopy(source['frozen_actor_prior']),
        source_Q_replay_entropy_and_optimizers_imported=False)


class ActualFlapResidualSACPilot(StagedHybridGoalSACPilot):
    artifact_type='staged_actual_flap_bounded_actor_correction_hybrid_sac_v1'
    agent_class=BoundedCorrectionHybridSAC
    supplemental_observation_dim=SUPPLEMENTAL_DIM
    correction_radius=.15

    def __init__(self,*args,body_anchor_state=None,checkpoint=None,device='cpu',
                 measured_train_credit=None,jaw_behavior=None,jaw_saturation=None,
                 success_jaw_balance=None,body_behavior=None,body_saturation=None,critic_episode_clock=None,**kwargs):
        saved=torch.load(checkpoint,map_location=device,weights_only=True) if checkpoint else None
        from ..observations.task_timing import resolve_critic_episode_clock
        self.critic_episode_clock = resolve_critic_episode_clock(saved, critic_episode_clock)
        if saved is not None and saved.get('artifact_type')!=self.artifact_type:
            raise ValueError('Old observation/control replay cannot resume an actual-flap correction learner')
        if saved is not None:
            if body_anchor_state is not None:raise ValueError('Continuation already owns its frozen body anchor')
            body_anchor_state=saved.get('body_anchor_state')
        if body_anchor_state is None:raise ValueError('A validated nominal body actor snapshot is required')
        self.body_anchor_state=deepcopy(body_anchor_state)
        from .measured_train_credit import measured_credit_config, VARIANT, TERMINAL_VARIANT
        stored = saved.get('measured_train_credit') if saved else None
        if stored is not None and stored not in (
                measured_credit_config(VARIANT), measured_credit_config(TERMINAL_VARIANT)):
            raise ValueError('Saved measured TRAIN credit configuration differs')
        requested = measured_credit_config(measured_train_credit)
        if stored is not None and measured_train_credit is not None and requested != stored:
            raise ValueError('Requested measured TRAIN credit differs from checkpoint')
        # This opt-in changes only the learner objective, not the physical
        # replay contract or existing model/optimizer coordinates.
        self.measured_train_credit = deepcopy(stored if stored is not None else requested)
        self.measured_credit_bank = None
        self.latest_measured_credit_collection = {}
        from .jaw_behavior_exploration import jaw_behavior_config, VARIANTS as JAW_VARIANTS
        stored_jaw = saved.get('jaw_behavior') if saved else None
        if stored_jaw is not None and (not isinstance(stored_jaw,dict) or stored_jaw.get('variant') not in JAW_VARIANTS \
                or stored_jaw != jaw_behavior_config(stored_jaw['variant'])):
            raise ValueError('Saved TRAIN jaw behavior configuration differs')
        if stored_jaw is not None and not isinstance(saved.get('jaw_behavior_statistics'), dict):
            raise ValueError('Saved TRAIN jaw behavior statistics are missing')
        requested_jaw = jaw_behavior_config(jaw_behavior)
        if stored_jaw is not None and jaw_behavior is not None and requested_jaw != stored_jaw:
            raise ValueError('Requested TRAIN jaw behavior differs from checkpoint')
        self.jaw_behavior = deepcopy(stored_jaw if stored_jaw is not None else requested_jaw)
        self.jaw_behavior_origin = deepcopy(saved.get('jaw_behavior_origin')) if stored_jaw else None
        self._saved_jaw_behavior = stored_jaw
        self.jaw_behavior_sampler = None
        from .body_behavior_exploration import body_behavior_config, body_behavior_statistics, VARIANTS as BODY_VARIANTS
        stored_body = saved.get('body_behavior') if saved else None
        if stored_body is not None and (not isinstance(stored_body,dict)
                or stored_body.get('variant') not in BODY_VARIANTS
                or stored_body != body_behavior_config(stored_body['variant'])):
            raise ValueError('Saved TRAIN body behavior configuration differs')
        if stored_body is not None and not isinstance(saved.get('body_behavior_statistics'), dict):
            raise ValueError('Saved TRAIN body behavior statistics are missing')
        requested_body = body_behavior_config(body_behavior)
        if stored_body is not None and body_behavior is not None and requested_body != stored_body:
            raise ValueError('Requested TRAIN body behavior differs from checkpoint')
        self.body_behavior = deepcopy(stored_body if stored_body is not None else requested_body)
        self._saved_body_behavior = stored_body
        self.body_behavior_origin = deepcopy(saved.get('body_behavior_origin')) if stored_body else None
        if stored_body is not None and not isinstance(self.body_behavior_origin, dict):
            raise ValueError('Saved TRAIN body behavior origin is missing')
        self._body_behavior_statistics = body_behavior_statistics(
            saved.get('body_behavior_statistics') if stored_body else None)
        self.body_behavior_sampler = None
        from .body_saturation import body_saturation_config, VARIANT as BODY_SATURATION_VARIANT
        stored_body_saturation = saved.get('body_saturation') if saved else None
        if stored_body_saturation is not None and stored_body_saturation != body_saturation_config(BODY_SATURATION_VARIANT):
            raise ValueError('Saved body saturation penalty configuration differs')
        requested_body_saturation = body_saturation_config(body_saturation)
        if stored_body_saturation is not None and body_saturation is not None and requested_body_saturation != stored_body_saturation:
            raise ValueError('Requested body saturation penalty differs from checkpoint')
        self.body_saturation = deepcopy(stored_body_saturation if stored_body_saturation is not None else requested_body_saturation)
        self._saved_body_saturation = stored_body_saturation
        self.body_saturation_origin = deepcopy(saved.get('body_saturation_origin')) if stored_body_saturation else None
        if stored_body_saturation is not None and not isinstance(self.body_saturation_origin, dict):
            raise ValueError('Saved body saturation penalty origin is missing')
        from .jaw_saturation import jaw_saturation_config, VARIANTS as SATURATION_VARIANTS
        stored_saturation = saved.get('jaw_saturation') if saved else None
        if stored_saturation is not None and (not isinstance(stored_saturation, dict)
                or stored_saturation.get('variant') not in SATURATION_VARIANTS
                or stored_saturation != jaw_saturation_config(stored_saturation['variant'])):
            raise ValueError('Saved jaw saturation penalty configuration differs')
        requested_saturation = jaw_saturation_config(jaw_saturation)
        if stored_saturation is not None and jaw_saturation is not None and requested_saturation != stored_saturation:
            raise ValueError('Requested jaw saturation penalty differs from checkpoint')
        self.jaw_saturation = deepcopy(stored_saturation if stored_saturation is not None else requested_saturation)
        self._saved_jaw_saturation = stored_saturation
        self.jaw_saturation_origin = deepcopy(saved.get('jaw_saturation_origin')) if stored_saturation else None
        if stored_saturation is not None and not isinstance(self.jaw_saturation_origin, dict):
            raise ValueError('Saved jaw saturation penalty origin is missing')
        from .success_jaw_balance import success_jaw_balance_config, VARIANT as BALANCE_VARIANT
        stored_balance = saved.get('success_jaw_balance') if saved else None
        if stored_balance is not None and stored_balance != success_jaw_balance_config(BALANCE_VARIANT):
            raise ValueError('Saved successful TRAIN jaw balance configuration differs')
        requested_balance = success_jaw_balance_config(success_jaw_balance)
        if stored_balance is not None and success_jaw_balance is not None and requested_balance != stored_balance:
            raise ValueError('Requested successful TRAIN jaw balance differs from checkpoint')
        self.success_jaw_balance = deepcopy(stored_balance if stored_balance is not None else requested_balance)
        self._saved_success_jaw_balance = stored_balance
        self.success_jaw_balance_origin = deepcopy(saved.get('success_jaw_balance_origin')) if stored_balance else None
        if stored_balance is not None and not isinstance(self.success_jaw_balance_origin, dict):
            raise ValueError('Saved successful TRAIN jaw balance origin is missing')
        if not checkpoint:
            # Defaults are specific to this opt-in contract. Existing pilot
            # construction and all ordinary RL settings remain unchanged.
            required=dict(free_grippers=True,normalize_prior_loss_by_radius=True,
                anchor_prior_to_initial_policy=True,fixed_prior_radius=self.correction_radius,
                validated_jaw_prior_confidence=body_anchor_state['source_goal_contract']['validated_jaw_prior_confidence'],
                jaw_prior_residual_gain=body_anchor_state['source_goal_contract']['jaw_prior_residual_gain'])
            if any(k in kwargs and kwargs[k]!=v for k,v in required.items()):
                raise ValueError('Correction controller requires its explicit independent-jaw/frozen-anchor contract')
            kwargs.update(required)
        super().__init__(*args,checkpoint=checkpoint,device=device,**kwargs)
        if self.body_saturation is not None and self.body_saturation_origin is None:
            self.body_saturation_origin = dict(source_checkpoint=str(checkpoint) if checkpoint else None,
                actor_updates_at_activation=self.actor_updates,critic_updates_at_activation=self.critic_updates,
                old_replay_rows_at_activation=self.replay.size,
                model_Q_normalizers_and_four_optimizer_states_kept=True,old_replay_kept_without_relabeling=True,
                scope='future_real_TRAIN_actor_updates')
        if self.body_behavior is not None:
            if not self.exploration_correlation or self.episode_arm_exploration is not None:
                raise ValueError('TRAIN body behavior requires correlated collection without the old480-D arm sampler')
            if self.body_behavior_origin is None:
                self.body_behavior_origin = dict(source_checkpoint=str(checkpoint) if checkpoint else None,
                    actor_updates_at_activation=self.actor_updates,critic_updates_at_activation=self.critic_updates,
                    old_replay_rows_at_activation=self.replay.size,
                    model_Q_normalizers_and_four_optimizer_states_kept=True,
                    old_replay_kept_with_original_behavior=True,old_rows_not_relabelled=True,
                    scope='future_real_TRAIN_collection')
        if self.success_jaw_balance is not None and self.success_jaw_balance_origin is None:
            self.success_jaw_balance_origin = dict(source_checkpoint=str(checkpoint) if checkpoint else None,
                actor_updates_at_activation=self.actor_updates,critic_updates_at_activation=self.critic_updates,
                old_replay_rows_at_activation=self.replay.size,
                model_Q_normalizers_and_four_optimizer_states_kept=True,
                actual_success_labels_body_loss_and_Q_replay_kept=True,
                scope='future_real_TRAIN_successful_jaw_NLL_updates')
        if self.jaw_saturation is not None and self.jaw_saturation_origin is None:
            self.jaw_saturation_origin = dict(source_checkpoint=str(checkpoint) if checkpoint else None,
                actor_updates_at_activation=self.actor_updates,critic_updates_at_activation=self.critic_updates,
                old_replay_rows_at_activation=self.replay.size,
                model_Q_normalizers_and_four_optimizer_states_kept=True,
                old_replay_kept_without_relabeling=True,
                source_checkpoint_actor_regularization='off',scope='future_real_TRAIN_actor_updates')
        if self.jaw_behavior is not None:
            if not self.exploration_correlation:
                raise ValueError('TRAIN joint jaw behavior requires correlated goal collection')
            if stored_jaw is not None and not isinstance(self.jaw_behavior_origin, dict):
                raise ValueError('Saved TRAIN jaw behavior origin is missing')
            if self.jaw_behavior_origin is None:
                self.jaw_behavior_origin = dict(source_checkpoint=str(checkpoint) if checkpoint else None,
                    actor_updates_at_activation=self.actor_updates,critic_updates_at_activation=self.critic_updates,
                    old_replay_rows_at_activation=self.replay.size,
                    old_replay_kept_with_original_behavior=True,old_rows_not_relabelled=True,
                    source_checkpoint_behavior='policy',scope='future_real_TRAIN_collection')
            from .jaw_behavior_exploration import JointJawBehaviorExploration
            self.jaw_behavior_sampler = JointJawBehaviorExploration(self.jaw_behavior,
                saved.get('jaw_behavior_statistics') if stored_jaw else None)
        if saved is None:
            source=self.body_anchor_state
            self.prior_schedule_actor_origin=-max(0.,source['source_actor_updates']-
                source['source_prior_schedule_actor_origin'])

    def learning_config(self,config):
        return replace(config,actor_lr=1e-5,initial_policy_std=.08,min_policy_std=.03,max_policy_std=.12)

    def reset_exploration(self, num_envs):
        if self.body_behavior_sampler is not None:
            self._body_behavior_statistics = self.body_behavior_sampler.report()
        super().reset_exploration(num_envs)
        self.body_behavior_sampler = None
        if self.body_behavior is not None and self.training:
            from .body_behavior_exploration import RampedArmBehaviorExploration
            self.body_behavior_sampler = RampedArmBehaviorExploration(num_envs, self.device,
                self.body_behavior, self._body_behavior_statistics)
            self.arm_behavior = self.body_behavior_sampler

    def body_behavior_extras(self):
        if self.body_behavior is None:
            return {}
        return dict(body_behavior=self.body_behavior,body_behavior_origin=self.body_behavior_origin,
            body_behavior_statistics=(self.body_behavior_sampler.report() if self.body_behavior_sampler
                is not None else deepcopy(self._body_behavior_statistics)))

    def nominal_view(self,raw):
        prefix=self.warm_start.actor_dim
        if raw.shape[1]!=self.actor_dim:raise ValueError('Measured augmented actor width differs')
        result=torch.cat((raw[:,:prefix],raw[:,-CONTEXT_DIM:]),-1).clone()
        result[:,-1]=self.body_anchor_state['source_goal_contract']['fixed_prior_radius']
        return result

    def normalized_nominal_view(self,normalized):
        # Destination mean is shifted for the changed radius coordinate.
        # With frozen normalization this reproduces the old normalized view,
        # including the original jaw residual subtraction reference.
        return torch.cat((normalized[:,:self.warm_start.actor_dim],normalized[:,-CONTEXT_DIM:]),-1)

    def _snapshot(self,model):
        n=self.warm_start.actor_dim+CONTEXT_DIM
        result=torch.nn.ModuleDict(dict(
            actor=SquashedActor(n,21,self.agent.config.hidden),actor_normalizer=ObservationNormalizer(n))).to(self.device)
        expected=result.state_dict()
        if set(model)!=set(expected) or any(v.shape!=expected[k].shape or not torch.isfinite(v).all()
                                           for k,v in model.items()):
            raise ValueError('Malformed frozen nominal actor snapshot')
        result.load_state_dict(model);result.requires_grad_(False)
        return result

    def configure_controller(self,saved):
        snapshot=self.body_anchor_state;source=snapshot['source_goal_contract']
        from .region_workplaces import frozen_anchor_templates
        from ..rewards.precision_capture import frozen_capture_actor_contract
        from .staged_physics import frozen_cpu_actor_contract
        if snapshot.get('format_version')!=1 or snapshot.get('source_Q_replay_entropy_and_optimizers_imported') is not False \
                or frozen_cpu_actor_contract(frozen_capture_actor_contract(source['physical_contract']))!=frozen_cpu_actor_contract(
                    frozen_capture_actor_contract({k:v for k,v in self.physical_contract.items() if k!='flap_dynamics'})) \
                or source['source_warm_start']!=self.warm_start.contract \
                or source['goal_center']!=self.center.tolist() or source['goal_scale']!=self.scale.tolist() \
                or source['action_coordinates']!=self.coordinates.name \
                or source['shelf_templates']!=frozen_anchor_templates(self.stage.templates) \
                or source['fixed_prior_radius']!=.05 or source['actor_dim']!=self.actor_dim-SUPPLEMENTAL_DIM \
                or source['critic_dim']!=self.critic_dim-SUPPLEMENTAL_DIM-int(self.critic_episode_clock is not None) \
                or snapshot['source_hidden']!=self.agent.config.hidden \
                or self.fixed_prior_radius!=self.correction_radius or not self.free_grippers:
            raise ValueError('Frozen anchor must match nominal physical goals, .05 radius, safety and perception')
        self.body_anchor=self._snapshot(snapshot['model'])
        self.source_projector=StagedGoalProjector(self.prior,self.prior.agent.actor_obs_dim,True,.05)
        self.agent.action_projector=AbsoluteGoalJawProjector()
        self.agent.correction_radius=self.correction_radius
        self.agent.executed_body_anchor=self.executed_body_anchor
        if self.measured_train_credit is not None:
            from .measured_train_credit import MeasuredTrainCreditBank
            self.measured_credit_bank = MeasuredTrainCreditBank(self.actor_dim, self.critic_dim,
                self.agent.config.gamma, self.measured_train_credit)
        self.agent.measured_train_credit_enabled = self.measured_train_credit is not None
        self.agent.jaw_saturation_config = self.jaw_saturation
        self.agent.body_saturation_config = self.body_saturation
        self.agent.success_jaw_balance_config = self.success_jaw_balance
        with torch.no_grad():
            old=self.body_anchor['actor'].state_dict();new=self.agent.actor.state_dict()
            prefix=self.warm_start.actor_dim
            for key,value in old.items():
                if key=='network.0.weight':
                    new[key].zero_();new[key][:,:prefix].copy_(value[:,:prefix]);new[key][:,-CONTEXT_DIM:].copy_(value[:,-CONTEXT_DIM:])
                else:new[key].copy_(value)
            self.agent.actor.load_state_dict(new)
            norm=self.agent.actor_normalizer;source_norm=self.body_anchor['actor_normalizer']
            norm.mean[:prefix].copy_(source_norm.mean[:prefix]);norm.var[:prefix].copy_(source_norm.var[:prefix])
            norm.mean[-CONTEXT_DIM:].copy_(source_norm.mean[-CONTEXT_DIM:])
            norm.mean[-1].add_(self.correction_radius-.05)
            norm.var[-CONTEXT_DIM:].copy_(source_norm.var[-CONTEXT_DIM:]);norm.count.copy_(source_norm.count)
            output=self.agent.actor.network[-1]
            output.weight[:19].zero_();output.bias[:19].zero_()
            output.weight[21:].zero_();output.bias[21:].fill_(math.log(self.agent.config.initial_policy_std))

    @torch.no_grad()
    def executed_body_anchor(self,raw):
        nominal=self.nominal_view(raw)
        requested=self.body_anchor['actor'](self.body_anchor['actor_normalizer'](nominal),deterministic=True)[0]
        return self.source_projector(nominal,requested)[:,:19]

    def make_frozen_actor_prior(self,state):
        return self._snapshot(self.body_anchor_state['frozen_actor_prior'] if state is None else state['frozen_actor_prior'])

    def _validated_jaw_logits(self,normalized):
        return self.frozen_actor_prior['actor'].network(self.normalized_nominal_view(normalized)).chunk(2,-1)[0][:,19:21]

    @property
    def prior_weight(self):return 0.

    @property
    def contract(self):
        result = super().contract|dict(supplemental_perception=supplemental_perception_contract(),
            body_controller='frozen_executed_nominal_actor_plus_learned_bounded_correction_v1',
            body_correction_radius=self.correction_radius,
            body_correction_bounds='symmetric_min_radius_and_distance_to_absolute_goal_bound',
            body_latent_to_physical_mapping='exactly_once_before_idempotent_physical_projection',
            nominal_frozen_anchor_radius=.05,prior_actor_loss_disabled=True,
            success_body_labels='measured_absolute_executed_goal_MSE_after_correction_mapping',
            jaw_proximity_gate='unchanged_production_nominal_hand_flap_gate',
            frozen_anchor_source_actor_updates=self.body_anchor_state['source_actor_updates'],
            extra_features_location='after_nominal_features_before_last_six_held_context',
            old_observation_control_Q_replay_imported=False)
        if self.critic_episode_clock is not None:
            result['critic_episode_clock'] = self.critic_episode_clock
        return result

    def checkpoint_extras(self):
        result = dict(body_anchor_state=self.body_anchor_state, **self.body_behavior_extras())
        if self.critic_episode_clock is not None:
            result['critic_episode_clock'] = self.critic_episode_clock
        if self.body_saturation is not None:
            result.update(body_saturation=self.body_saturation,body_saturation_origin=self.body_saturation_origin)
        if self.success_jaw_balance is not None:
            result.update(success_jaw_balance=self.success_jaw_balance,
                success_jaw_balance_origin=self.success_jaw_balance_origin)
        if self.jaw_saturation is not None:
            result.update(jaw_saturation=self.jaw_saturation,jaw_saturation_origin=self.jaw_saturation_origin)
        if self.jaw_behavior is not None:
            result.update(jaw_behavior=self.jaw_behavior,jaw_behavior_origin=self.jaw_behavior_origin,
                jaw_behavior_statistics=self.jaw_behavior_sampler.report())
        if self.measured_train_credit is not None:
            result.update(measured_train_credit=self.measured_train_credit,
                measured_train_credit_bank_report=self.measured_credit_bank.report())
        return result

    def restore_experience_extras(self, state):
        if state.get('critic_episode_clock') != self.critic_episode_clock:
            raise ValueError('Critic episode clock checkpoint/replay provenance differs')
        if state.get('body_saturation') != self._saved_body_saturation:
            raise ValueError('Body saturation checkpoint/replay provenance differs')
        if self._saved_body_saturation is not None and state.get('body_saturation_origin') != self.body_saturation_origin:
            raise ValueError('Body saturation replay origin differs')
        if state.get('body_behavior') != self._saved_body_behavior:
            raise ValueError('TRAIN body behavior checkpoint/replay provenance differs')
        if self._saved_body_behavior is not None and state.get('body_behavior_origin') != self.body_behavior_origin:
            raise ValueError('TRAIN body behavior replay origin differs')
        if state.get('success_jaw_balance') != self._saved_success_jaw_balance:
            raise ValueError('Successful TRAIN jaw balance checkpoint/replay provenance differs')
        if self._saved_success_jaw_balance is not None and state.get('success_jaw_balance_origin') != self.success_jaw_balance_origin:
            raise ValueError('Successful TRAIN jaw balance replay origin differs')
        if state.get('jaw_saturation') != self._saved_jaw_saturation:
            raise ValueError('Jaw saturation checkpoint/replay provenance differs')
        if self._saved_jaw_saturation is not None and state.get('jaw_saturation_origin') != self.jaw_saturation_origin:
            raise ValueError('Jaw saturation replay origin differs')
        if state.get('jaw_behavior') != self._saved_jaw_behavior:
            raise ValueError('TRAIN jaw behavior checkpoint/replay provenance differs')
        if self._saved_jaw_behavior is not None and state.get('jaw_behavior_origin') != self.jaw_behavior_origin:
            raise ValueError('TRAIN jaw behavior replay origin differs')
        stored = state.get('measured_train_credit')
        if stored is not None and stored != self.measured_train_credit:
            raise ValueError('Measured TRAIN credit replay configuration differs')
        if stored is not None:
            self.measured_credit_bank.restore(state['measured_train_credit_bank'])

    def experience_extras(self):
        result = self.body_behavior_extras()
        if self.critic_episode_clock is not None:
            result['critic_episode_clock'] = self.critic_episode_clock
        if self.body_saturation is not None:
            result.update(body_saturation=self.body_saturation,body_saturation_origin=self.body_saturation_origin)
        if self.success_jaw_balance is not None:
            result.update(success_jaw_balance=self.success_jaw_balance,
                success_jaw_balance_origin=self.success_jaw_balance_origin)
        if self.jaw_saturation is not None:
            result.update(jaw_saturation=self.jaw_saturation,jaw_saturation_origin=self.jaw_saturation_origin)
        if self.jaw_behavior is not None:
            result.update(jaw_behavior=self.jaw_behavior,jaw_behavior_origin=self.jaw_behavior_origin,
                jaw_behavior_statistics=self.jaw_behavior_sampler.report())
        if self.measured_train_credit is not None:
            result.update(measured_train_credit=self.measured_train_credit,
                measured_train_credit_bank=self.measured_credit_bank.state())
        return result

    def critic_auxiliary_options(self):
        bank = self.measured_credit_bank
        if bank is None or not bank.size:
            return {}
        return dict(critic_auxiliary=bank.sample(self.measured_train_credit['batch_size'], self.device),
            critic_auxiliary_weight=self.measured_train_credit['critic_weight'])

    def add_measured_training_wave(self, wave, outcomes, batches, *, source_run):
        if self.measured_credit_bank is not None:
            from .measured_train_credit import add_measured_training_wave
            self.latest_measured_credit_collection = add_measured_training_wave(
                self.measured_credit_bank, wave, outcomes, batches, source_run=source_run)

    def report(self):
        result = super().report()
        result.update(self.body_behavior_extras())
        if self.body_behavior and self.body_behavior.get('unselected_policy_sampling'):
            stats=result['body_behavior_statistics']
            result['TRAIN_policy_mode_statistics']=dict(
                drawn_episodes=stats['episodes_drawn'],
                exploratory_episodes=stats['biased_episodes'],
                greedy_episodes=stats['episodes_drawn']-stats['biased_episodes'],
                exploratory_held_rows=stats['biased_episode_rows'],
                greedy_held_rows=stats['held_collection_rows']-stats['biased_episode_rows'],
                same_original20percent_arm_selection=True)
        if self.body_saturation is not None:
            result.update(body_saturation=self.body_saturation,body_saturation_origin=self.body_saturation_origin)
        if self.success_jaw_balance is not None:
            result.update(success_jaw_balance=self.success_jaw_balance,
                success_jaw_balance_origin=self.success_jaw_balance_origin)
        if self.jaw_saturation is not None:
            result.update(jaw_saturation=self.jaw_saturation,jaw_saturation_origin=self.jaw_saturation_origin)
        if self.jaw_behavior is not None:
            result.update(jaw_behavior=self.jaw_behavior,jaw_behavior_origin=self.jaw_behavior_origin,
                jaw_behavior_statistics=self.jaw_behavior_sampler.report())
        if self.measured_train_credit is not None:
            result.update(measured_train_credit=self.measured_train_credit,
                measured_train_credit_bank=self.measured_credit_bank.report(),
                measured_train_credit_collection=self.latest_measured_credit_collection)
        return result
