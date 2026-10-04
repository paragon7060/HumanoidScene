"""Fresh-Q SAC after measured neutral-arm base approach and base pose hold.

The frozen learned goal network initializes the remaining 21 goals. Neither its
24-action Q nor old replay is compatible with this controller. Only executed
held-phase transitions enter the new replay. No live IK/VR teacher is required.
"""
from dataclasses import replace
from copy import deepcopy
import math
from pathlib import Path

import torch

from ...algorithms.asymmetric_sac import AsymmetricReplayBuffer, AsymmetricSAC
from ...runners.storage import save_checkpoint
from .pose_goal_sac import GoalGripperProjector, reward_discount
from .executed_replay import PHYSICAL_KEYS


GOAL_COLUMNS = (*range(19), 22, 23)
CONTEXT_DIM = 6


def staged_context(raw, stage, radius):
    if stage.phase != 'held_grasp' or stage.manipulation_start is None:
        raise ValueError('SAC replay starts only after physically confirmed base settling')
    target = stage.target_xy.to(raw).expand(len(raw), -1)
    heading = (stage.target_yaw.to(raw)[:,None] if isinstance(stage.target_yaw,torch.Tensor)
               else raw.new_full((len(raw), 1), stage.target_yaw))
    return torch.cat((raw.new_ones(len(raw), 1), target, heading.sin(), heading.cos(),
                      raw.new_full((len(raw), 1), radius)), -1)


def held_goal_coordinates(coordinates, raw, remaining, stage):
    """One physical decoder: neural arm/torso/jaw goals, measured held base goal."""
    if remaining.shape != (len(raw), 21) or not torch.isfinite(remaining).all():
        raise ValueError('Expected 21 finite non-base goals')
    goal = raw.new_empty(len(raw), 24)
    goal[:, list(GOAL_COLUMNS)] = remaining
    goal[:, 19:21] = stage.target_xy
    goal[:, 21] = stage.target_yaw
    return coordinates.decode(raw, goal)


class StagedGoalProjector:
    name = 'staged_goal_21_near_flap_jaws_frozen_network_radius_context_v1'

    def __init__(self, prior, feature_width, free_grippers=False):
        self.prior = prior
        self.feature_width = feature_width
        self.gate = GoalGripperProjector()
        self.free_grippers = free_grippers

    def __call__(self, observation, action):
        with torch.no_grad():
            reference = self.prior.agent.act(observation[:, :self.feature_width], deterministic=True)
            reference = reference[:, list(GOAL_COLUMNS)]
        radius = observation[:, -1:].clamp(.05, .15)
        result = torch.maximum(reference-radius, torch.minimum(action, reference+radius)).clamp(-1, 1)
        if self.free_grippers:
            result[:, 19:21] = action[:, 19:21].clamp(-1,1)
        result[:, 19:21] = torch.where(self.gate.near(observation), result[:, 19:21], -1.)
        return result

    def entropy_mask(self, observation):
        mask = observation.new_ones(len(observation), 21)
        mask[:, 19:21] = self.gate.near(observation)
        return mask


def copy_remaining_actor(source, destination):
    """Copy the old mean/normalization; no critic or optimizer tensors migrate."""
    with torch.no_grad():
        old, new = source.actor.state_dict(), destination.actor.state_dict()
        last = len(source.actor.network)-1
        for key, value in old.items():
            if key == 'network.0.weight':
                new[key].zero_()
                new[key][:, :value.shape[1]].copy_(value)
            elif key in (f'network.{last}.weight', f'network.{last}.bias'):
                ids = [*GOAL_COLUMNS, *(24+i for i in GOAL_COLUMNS)]
                new[key].copy_(value[ids])
            else:
                new[key].copy_(value)
        destination.actor.load_state_dict(new)
        width = source.actor_normalizer.mean.numel()
        destination.actor_normalizer.mean[:width].copy_(source.actor_normalizer.mean)
        destination.actor_normalizer.var[:width].copy_(source.actor_normalizer.var)
        destination.actor_normalizer.count.copy_(source.actor_normalizer.count)
        output = destination.actor.network[-1]
        output.weight[21:].zero_()
        output.bias[21:].fill_(math.log(destination.config.initial_policy_std))


def initialize_staged_actor_only(pilot, source):
    """Reuse matching held-goal behavior after a lift/solver contract change.

    Source Q, normalizer, entropy, optimizer, counters, and replay stay out.
    The destination must be freshly constructed. This is initialization,
    never a continuation of training under a different terminal predicate.
    """
    from .staged_physics import frozen_prior_lift_contract
    if pilot.critic_updates or pilot.actor_updates or pilot.replay.size \
            or any(opt.state for opt in pilot.agent.optimizers):
        raise ValueError('Actor-only migration requires fresh Q, replay and optimizers')
    def actor_contract(value):
        value=deepcopy(value)
        physical=frozen_prior_lift_contract(value['physical_contract'])
        physical.pop('physics_dynamics',None)  # Dynamics may change only for fresh Q.
        value['physical_contract']=physical
        return value
    if source.get('artifact_type')!=pilot.artifact_type \
            or actor_contract(source['goal_contract'])!=actor_contract(pilot.contract):
        raise ValueError('Actor-only migration requires matching coordinates, goals and safety')
    current=pilot.agent.state_dict()
    names=[key for key in current if key.startswith(('actor.','actor_normalizer.'))]
    for key in names:
        value=source['model'].get(key)
        if value is None or value.shape!=current[key].shape or not torch.isfinite(value).all():
            raise ValueError('Actor-only migration contains malformed tensors')
    with torch.no_grad():
        for key in names:current[key].copy_(source['model'][key])
    if pilot.frozen_actor_prior is not None:
        pilot.frozen_actor_prior.load_state_dict({key:current[key] for key in names})
    return dict(actor_only=True,source_actor_updates=source['actor_updates'],
                source_critic_updates_not_imported=source['critic_updates'],
                destination_actor_updates=0,destination_critic_updates=0,
                actual_replay_rows=0,optimizer_states_imported=False)


class StagedGoalSACPilot:
    artifact_type = 'staged_base_hold_remaining_goal_sac_v1'
    agent_class = AsymmetricSAC

    def __init__(self, warm_start, physical_contract, directory, stage, *,
                 checkpoint=None, training=True, device='cpu', free_grippers=False,
                 gripper_logit_scale=1., replay_capacity=20000,
                 normalize_prior_loss_by_radius=False,actor_min_replay_rows=64,
                 anchor_prior_to_initial_policy=False,exploration_correlation=0.):
        if warm_start.training:
            raise ValueError('The warm-start network must remain frozen')
        self.warm_start = warm_start
        self.coordinates = warm_start.coordinates
        self.prior = warm_start.prior
        # Run-specific diagnostic provenance is not an MDP difference. Keep
        # every field used by the common strict physical contract validator.
        self.physical_contract = {key:physical_contract.get(key)
                                  for key in (*PHYSICAL_KEYS, 'flap_pose_source')}
        if 'physics_dynamics' in physical_contract:
            from .staged_physics import staged_solver_contract
            dynamics=physical_contract['physics_dynamics']
            if not isinstance(dynamics,dict) or dynamics!=staged_solver_contract(dynamics.get('solver')):
                raise ValueError('Staged dynamics identity differs')
            self.physical_contract['physics_dynamics']=dict(dynamics)
        self.center = warm_start.center[list(GOAL_COLUMNS)].clone()
        self.scale = warm_start.scale[list(GOAL_COLUMNS)].clone()
        self.directory = Path(directory)
        self.device, self.training, self.stage = device, training, stage
        self.anchor = None
        self.actor_updates = self.critic_updates = self.online_rows = 0
        self.prior_schedule_actor_origin=0.
        self.actor_dim = warm_start.actor_dim+CONTEXT_DIM
        self.critic_dim = 533+CONTEXT_DIM
        self.warmup = 2048
        self.fade = 20000
        self.latest = {}
        self.latest_actor = {}
        self.history = []
        saved = torch.load(checkpoint,map_location=device,weights_only=True) if checkpoint else None
        if saved:
            free_grippers=saved.get('goal_contract',{}).get('gripper_prior_bound') is False
            gripper_logit_scale=saved.get('goal_contract',{}).get('gripper_logit_scale',1.)
            replay_capacity=saved.get('goal_contract',{}).get('replay_capacity',20000)
            normalize_prior_loss_by_radius=saved.get('goal_contract',{}).get('normalize_prior_loss_by_radius',False)
            actor_min_replay_rows=saved.get('goal_contract',{}).get('actor_min_replay_rows',64)
            anchor_prior_to_initial_policy=saved.get('goal_contract',{}).get('actor_prior_source')=='frozen_validated_remaining_goal_actor'
            exploration_correlation=saved.get('goal_contract',{}).get('exploration_correlation',0.)
            self.prior_schedule_actor_origin=saved.get('prior_schedule_actor_origin',0.)
        if not math.isfinite(exploration_correlation) or not 0<=exploration_correlation<=.995:
            raise ValueError('Goal exploration correlation must be within0..0.995')
        self.exploration_correlation=exploration_correlation
        self.goal_exploration=None
        if type(normalize_prior_loss_by_radius) is not bool:
            raise ValueError('Prior loss normalization must be an explicit boolean')
        self.normalize_prior_loss_by_radius=normalize_prior_loss_by_radius
        if type(anchor_prior_to_initial_policy) is not bool or (
                anchor_prior_to_initial_policy and not normalize_prior_loss_by_radius):
            raise ValueError('Validated actor anchoring requires normalized prior loss')
        self.anchor_prior_to_initial_policy=anchor_prior_to_initial_policy
        if type(replay_capacity) is not int or not 1024<=replay_capacity<=2000000:
            raise ValueError('Staged replay capacity must be within1024..2000000')
        self.replay_capacity=replay_capacity
        if type(actor_min_replay_rows) is not int or not 64<=actor_min_replay_rows<=replay_capacity:
            raise ValueError('Actor collection warmup must be within64..replay capacity')
        if not math.isfinite(self.prior_schedule_actor_origin):
            raise ValueError('Malformed actor prior schedule origin')
        self.actor_min_replay_rows=actor_min_replay_rows
        if type(free_grippers) is not bool or not math.isfinite(gripper_logit_scale) \
                or not 0 < gripper_logit_scale <= 1 or (not free_grippers and gripper_logit_scale != 1):
            raise ValueError('Soft gripper logits require independent jaw exploration')
        self.free_grippers,self.gripper_logit_scale=free_grippers,gripper_logit_scale
        self.frozen_warm_start = warm_start.agent.checkpoint() | dict(
            format_version=1, artifact_type=warm_start.artifact_type, goal_contract=warm_start.contract,
            bc_prior=self.prior.state, actor_updates=warm_start.actor_updates,
            critic_updates=warm_start.critic_updates, action_coordinates=self.coordinates.name,
            reference_runtime_dependency=False)
        config = replace(warm_start.agent.config, actor_lr=1e-6,
            initial_policy_std=.005, min_policy_std=.001, max_policy_std=.02,
            gamma=reward_discount(physical_contract), freeze_actor_normalizer=True,
            actor_feature_mode='flat', critic_layer_norm=True)
        projector = StagedGoalProjector(self.prior, self.prior.agent.actor_obs_dim,free_grippers)
        self.agent = self.agent_class(self.actor_dim, self.critic_dim, 21, config, device,
                                   action_projector=projector)
        copy_remaining_actor(warm_start.agent, self.agent)
        if free_grippers:
            with torch.no_grad():
                self.agent.actor.network[-1].weight[19:21].mul_(gripper_logit_scale)
                self.agent.actor.network[-1].bias[19:21].mul_(gripper_logit_scale)
        self.replay = AsymmetricReplayBuffer(replay_capacity, self.actor_dim, self.critic_dim, 21, device)
        if checkpoint:
            state = saved
            if state.get('artifact_type') != self.artifact_type or state.get('goal_contract') != self.contract:
                raise ValueError('Staged SAC requires the same phase/waypoint/remaining-goal contract')
            self.agent.restore(state, training=training)
            self.actor_updates = state['actor_updates']
            self.critic_updates = state['critic_updates']
            self.latest_actor = state.get('latest_actor_metrics', {})
            previous = Path(checkpoint).parent/'staged_goal_experience.pt'
            if training:
                if not previous.is_file():
                    raise ValueError('Staged SAC continuation requires its actual held-phase replay')
                saved = torch.load(previous, map_location=device, weights_only=True)
                if saved.get('goal_contract') != self.contract:
                    raise ValueError('Staged replay context differs')
                rows = saved['executed_goal_transitions']
                if set(rows) != set(self.replay.data):
                    raise ValueError('Staged replay fields differ')
                n = len(rows['reward'])
                if n > self.replay.capacity:
                    raise ValueError('Invalid staged replay length')
                for key, storage in self.replay.data.items():
                    if rows[key].shape != (n, *storage.shape[1:]) or not torch.isfinite(rows[key]).all():
                        raise ValueError('Malformed staged replay')
                if (rows['action'].abs()>1.00001).any():
                    raise ValueError('Unbounded staged replay action')
                for key in ('actor_obs', 'next_actor_obs', 'critic_obs', 'next_critic_obs'):
                    if not bool((rows[key][:, -CONTEXT_DIM] == 1).all()):
                        raise ValueError('Replay contains an unconfirmed approach phase')
                self.replay.add(**rows)
                self.history.append({k:v.cpu() for k,v in rows.items()})
        self.frozen_actor_prior=None
        if self.anchor_prior_to_initial_policy:
            # Actor-only snapshot. It cannot bring another controller's Q,
            # reward or transitions into the real held-phase replay.
            self.frozen_actor_prior=torch.nn.ModuleDict(dict(
                actor=deepcopy(self.agent.actor),actor_normalizer=deepcopy(self.agent.actor_normalizer)))
            if saved is not None:
                if 'frozen_actor_prior' not in state:
                    raise ValueError('Validated prior actor snapshot is missing')
                self.frozen_actor_prior.load_state_dict(state['frozen_actor_prior'])
            self.frozen_actor_prior.requires_grad_(False)
        if not training:
            self.agent.requires_grad_(False)

    @property
    def contract(self):
        contract=dict(name=self.artifact_type, phase_context='held_grasp_only_after_physical_base_settling',
            context_order=['held_phase', 'held_x_rack_m', 'held_y_rack_m',
                           'sin_held_yaw', 'cos_held_yaw', 'policy_radius'],
            action_columns=list(GOAL_COLUMNS), action_coordinates=self.coordinates.name,
            physical_contract=self.physical_contract, actor_dim=self.actor_dim, critic_dim=self.critic_dim,
            goal_center=self.center.tolist(), goal_scale=self.scale.tolist(),
            waypoint_format=self.stage.name,
            shelf_templates=self.stage.templates,
            source_warm_start=self.warm_start.contract,
            old_Q_or_replay_imported=False, fresh_critic=True, initial_critic_warmup=self.warmup,
            actor_update_interval=4, radius_initial=.05, radius_final=.15,
            prior_initial_weight=2., fade_critic_updates=self.fade,
            old_demo_Q_fraction=0., runtime_reference_path_required=False,
            live_IK_or_privileged_contact_teacher_required=False,
            exploration_std_initial=.005, exploration_std_cap=.02)
        if self.free_grippers:
            contract.update(gripper_prior_bound=False,gripper_logit_scale=self.gripper_logit_scale,
                            imitation_jaw_targets='same_sign_softened_frozen_prior_logits')
        if self.replay_capacity!=20000:contract['replay_capacity']=self.replay_capacity
        if self.normalize_prior_loss_by_radius:
            contract.update(normalize_prior_loss_by_radius=True,
                prior_loss_units='mean_squared_normalized_goal_error_divided_by_current_radius_squared',
                prior_fade_units='actor_updates_after_schedule_origin',fade_actor_updates=self.fade//4)
        if self.actor_min_replay_rows!=64:contract['actor_min_replay_rows']=self.actor_min_replay_rows
        if self.anchor_prior_to_initial_policy:
            contract['actor_prior_source']='frozen_validated_remaining_goal_actor'
        if self.exploration_correlation:
            contract.update(exploration_correlation=self.exploration_correlation,
                collection_noise='independent_per_environment_AR1_pre_tanh_Gaussian',
                critic_actor_target_noise='standard_SAC_Gaussian_unchanged')
        return contract

    @property
    def progress(self):
        if self.normalize_prior_loss_by_radius:
            return min(1.,max(0,self.actor_updates-self.prior_schedule_actor_origin)/(self.fade//4))
        return min(1., max(0, self.critic_updates-self.warmup)/self.fade)

    @property
    def radius(self):
        return .05+.10*self.progress

    @property
    def prior_weight(self):
        weight=2.*(1-self.progress)
        return weight/self.radius**2 if self.normalize_prior_loss_by_radius else weight

    def observations(self, raw, critic, index):
        ao, co = self.warm_start.observations(raw, critic, index, self.anchor)
        context = staged_context(raw, self.stage, self.radius)
        return torch.cat((ao, context), -1), torch.cat((co, context), -1)

    @torch.no_grad()
    def act(self, raw, critic, index, *, exploration_ids=None):
        if self.anchor is None:
            self.anchor = self.coordinates.box_anchor(raw).clone()
        ao, co = self.observations(raw, critic, index)
        deterministic=not self.training or self.replay.size<64
        if self.exploration_correlation and not deterministic:
            if self.goal_exploration is None:
                # The single-environment runner has one identity throughout
                # an episode. Vector callers must reset with their full count.
                self.reset_exploration(len(raw))
            ids=(torch.arange(len(raw),device=raw.device) if exploration_ids is None else exploration_ids)
            action=self.goal_exploration.act(self.agent,ao,ids)
        else:
            action=self.agent.act(ao,deterministic=deterministic)
        physical = held_goal_coordinates(self.coordinates, raw, self.center+self.scale*action, self.stage)
        return physical, (ao.detach(), co.detach(), action.detach())

    def reset_exploration(self, num_envs):
        from .correlated_goal_exploration import CorrelatedGoalExploration
        self.goal_exploration=(CorrelatedGoalExploration(self.exploration_correlation,
            num_envs,21,self.device) if self.exploration_correlation else None)

    def observe(self, previous, next_raw, next_critic, reward, terminated, index):
        if not self.training:
            return
        ao, co, action = previous
        na, nc = self.observations(next_raw, next_critic, index+1)
        batch = dict(actor_obs=ao, critic_obs=co, action=action, next_actor_obs=na,
                     next_critic_obs=nc, reward=reward.detach(), terminated=terminated.detach())
        self.replay.add(**batch)
        self.history.append({k:v.detach().cpu() for k,v in batch.items()})
        self.online_rows += len(ao)
        if self.replay.size < 64:
            return
        with torch.enable_grad():
            for _ in range(2):
                actual = self.replay.sample(256, self.device)
                self.agent.update_normalizers(actual['actor_obs'], actual['critic_obs'])
                update_actor = (self.critic_updates >= self.warmup and self.critic_updates % 4 == 0
                                and self.replay.size>=self.actor_min_replay_rows)
                teacher, weight = None, self.prior_weight if update_actor else 0.
                if weight:
                    with torch.no_grad():
                        if self.frozen_actor_prior is not None:
                            normalized=self.frozen_actor_prior['actor_normalizer'](actual['actor_obs'])
                            labels=self.frozen_actor_prior['actor'](normalized,deterministic=True)[0]
                        else:
                            labels = self.prior.agent.act(actual['actor_obs'][:, :self.prior.agent.actor_obs_dim],
                                                          deterministic=True)[:, list(GOAL_COLUMNS)]
                            if self.free_grippers:
                                labels[:,19:21]=(labels[:,19:21].clamp(-.999999,.999999).atanh()
                                                 *self.gripper_logit_scale).tanh()
                    teacher = dict(actor_obs=actual['actor_obs'], action=labels)
                self.latest = self.agent.update(actual, teacher=teacher, teacher_weight=weight,
                                                update_actor=update_actor)
                if update_actor:
                    self.latest_actor = dict(self.latest, critic_update=self.critic_updates+1)
                self.actor_updates += int(update_actor)
                self.critic_updates += 1

    def report(self):
        return dict(training=self.training, actor_updates=self.actor_updates,
            critic_updates=self.critic_updates, online_rows=self.online_rows,
            replay_size=self.replay.size, seed_rows=0, old_Q_or_replay_imported=False,
            gripper_prior_bound=not self.free_grippers,gripper_logit_scale=self.gripper_logit_scale,
            prior_weight=2.*(1-self.progress), effective_prior_mse_weight=self.prior_weight,
            normalize_prior_loss_by_radius=self.normalize_prior_loss_by_radius,prior_radius=self.radius,
            critic_warmup_remaining=max(0, self.warmup-self.critic_updates),
            actor_collection_warmup_remaining=max(0,self.actor_min_replay_rows-self.replay.size),
            prior_schedule_actor_origin=self.prior_schedule_actor_origin,
            exploration_correlation=self.exploration_correlation,
            min_policy_std=self.agent.config.min_policy_std,
            max_policy_std=self.agent.config.max_policy_std,
            runtime_reference_path_required=False, latest=self.latest,
            latest_actor=self.latest_actor, goal_contract=self.contract)

    def save(self, final=False):
        if not final and getattr(self, '_last_saved', None) == self.critic_updates:
            return
        if getattr(self, '_last_saved', None) != self.critic_updates:
            state = self.agent.checkpoint() | dict(artifact_type=self.artifact_type,
                goal_contract=self.contract, frozen_warm_start=self.frozen_warm_start,
                actor_updates=self.actor_updates, critic_updates=self.critic_updates,
                latest_actor_metrics=self.latest_actor,
                prior_schedule_actor_origin=self.prior_schedule_actor_origin,
                reference_runtime_dependency=False)
            if self.frozen_actor_prior is not None:
                state['frozen_actor_prior']=self.frozen_actor_prior.state_dict()
            # A closed evaluation wave must not rewrite an already uploaded
            # checkpoint with the same critic counter and remote filename.
            save_checkpoint(self.directory, state, self.critic_updates, keep=None)
            self._last_saved = self.critic_updates
        if final:
            # torch.save serializes the entire underlying storage of a view,
            # including an empty CPU slice or discarded history prefix.
            # Own only the retained rows before writing the replay artifact.
            remaining=self.replay.capacity;recent=[]
            for batch in reversed(self.history):
                count=min(remaining,len(batch['reward']))
                if count:recent.append({k:v[-count:] for k,v in batch.items()})
                remaining-=count
                if not remaining:break
            recent.reverse()
            rows = {k:(torch.cat([b[k] for b in recent]).detach().cpu()
                       if recent else v[:0].detach().cpu().clone())
                    for k,v in self.replay.data.items()}
            torch.save(dict(goal_contract=self.contract, executed_goal_transitions=rows),
                       self.directory/'staged_goal_experience.pt')
