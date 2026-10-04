"""Literal physical body commands with a frozen, learned goal-controller prior.

The Q action is what the servo executes, not an inverse label of a clipped goal.
Base feedback remains the independently measured held-waypoint controller.
"""
from copy import deepcopy
import math

import torch
from torch import nn

from ...algorithms.common import ObservationNormalizer
from ...algorithms.sac import SquashedActor
from .pose_goal_sac import GoalGripperProjector
from .staged_goal_sac import GOAL_COLUMNS
from ..geometry.upright_torso import planar_position

PHYSICAL_COLUMNS = tuple(range(3,18)) + (22,23,18,19,20,21)
PRIOR_FORMAT = 'frozen_goal_actor_to_physical_body_prior_v1'


def actor_only_snapshot(state):
    contract=state.get('goal_contract',{})
    if state.get('artifact_type')!='staged_base_hold_remaining_hybrid_sac_v1' \
            or contract.get('actor_dim')!=480 or contract.get('critic_dim')!=539 \
            or contract.get('fixed_prior_radius')!=.05 or not state['config'].get('freeze_actor_normalizer'):
        raise ValueError('Physical body prior requires the matching frozen hybrid actor and fixed normalization')
    model=state['model']
    def subset(prefix):return {k[len(prefix):]:v.detach().cpu().clone() for k,v in model.items() if k.startswith(prefix)}
    reference={k[len('actor.'):]:v.detach().cpu().clone() for k,v in
               state.get('frozen_actor_prior',{}).items() if k.startswith('actor.')}
    if contract.get('validated_jaw_prior_confidence') and not reference:
        raise ValueError('Frozen binary prior actor is missing')
    return dict(format=PRIOR_FORMAT,goal_contract=deepcopy(contract),config=deepcopy(state['config']),
        actor=subset('actor.'),normalizer=subset('actor_normalizer.'),validated_actor=reference,
        source_actor_updates=state['actor_updates'],source_critic_counter_not_imported=state['critic_updates'],
        source_Q_and_optimizer_imported=False)


def servo_raw_actor(observation):
    """Recover only measured proprio/servo fields, without inventing box tokens.

    PoseGoalCoordinates uses raw0:86 and raw412:440 to decode body and base.
    Their selected-target feature positions are literal copies in actor480.
    The removed last-action shortcut and unused token fields stay zero.
    """
    if observation.ndim!=2 or observation.shape[1]!=480 or not torch.isfinite(observation).all() \
            or not bool((observation[:,-6]==1).all()) or not bool((observation[:,173]>.5).all()):
        raise ValueError('Physical decoding needs measured actor480 held phase and valid servo telemetry')
    raw=observation.new_zeros(len(observation),464)
    raw[:,:86]=observation[:,:86]
    raw[:,412:440]=observation[:,146:174]
    return raw


def body_command(full):
    if full.ndim!=2 or full.shape[1]!=24 or not torch.isfinite(full).all():
        raise ValueError('Expected literal finite physical24 commands')
    return full[:,list(PHYSICAL_COLUMNS)]


def full_command(coordinates,observation,body):
    if body.shape!=(len(observation),21) or not torch.isfinite(body).all() \
            or (body.abs()>1.00001).any() or not (body[:,19:].abs()==1).all():
        raise ValueError('Physical body command must be bounded21 with binary jaws')
    raw=servo_raw_actor(observation)
    goal=raw.new_zeros(len(raw),24)
    goal[:,19:21]=observation[:,-5:-3]
    goal[:,21]=torch.atan2(observation[:,-3],observation[:,-2])
    physical=coordinates.decode(raw,goal)
    physical[:,list(PHYSICAL_COLUMNS)]=body
    return physical


def goal_body_command(coordinates,observation,goals):
    """Decode body21 independently of the unused base-plane wheel solver."""
    if goals.shape!=(len(observation),21) or not torch.isfinite(goals).all():
        raise ValueError('Expected finite non-base goals21')
    raw=servo_raw_actor(observation)
    targets=raw[:,:20]+raw[:,416:436]
    joint=targets[:,coordinates.joints.joint_columns]
    torso=planar_position(targets[:,:2],coordinates.links.to(raw))
    return torch.cat(((goals[:,:17]-joint)/raw.new_tensor(coordinates.joints.scales),
        (goals[:,17:19]-torso)/(.1/30),goals[:,19:21]),-1).clamp(-1,1)


class FrozenGoalCommandPrior(nn.Module):
    """Only trained actors/normalization; source Q and optimizers are excluded."""
    def __init__(self,snapshot,warm,device):
        super().__init__()
        if snapshot.get('format')!=PRIOR_FORMAT or snapshot.get('source_Q_and_optimizer_imported') is not False:
            raise ValueError('Expected an explicit actor-only physical prior snapshot')
        self.snapshot=deepcopy(snapshot)
        config=snapshot['config'];contract=snapshot['goal_contract']
        self.actor=SquashedActor(480,21,config['hidden'],config['max_policy_std'],config['min_policy_std'])
        self.actor.load_state_dict(snapshot['actor'])
        self.normalizer=ObservationNormalizer(480);self.normalizer.load_state_dict(snapshot['normalizer'])
        self.reference=None
        if snapshot['validated_actor']:
            self.reference=SquashedActor(480,21,config['hidden'],config['max_policy_std'],config['min_policy_std'])
            self.reference.load_state_dict(snapshot['validated_actor'])
        self.coordinates=warm.coordinates;self.bc_prior=warm.prior
        self.feature_width=warm.prior.agent.actor_obs_dim
        self.confidence=contract.get('validated_jaw_prior_confidence',0.)
        self.jaw_gain=contract.get('jaw_prior_residual_gain',1.)
        self.register_buffer('center',torch.tensor(contract['goal_center']))
        self.register_buffer('scale',torch.tensor(contract['goal_scale']))
        self.gate=GoalGripperProjector()
        self.to(device);self.requires_grad_(False)

    def jaw_parameters(self,normalized):
        raw_logits=self.actor.network(normalized).chunk(2,-1)[0][:,19:21]
        effective=raw_logits
        if self.confidence:
            reference=self.reference.network(normalized).chunk(2,-1)[0][:,19:21]
            magnitude=math.log(self.confidence/(1-self.confidence))
            effective=torch.where(reference>0,magnitude,-magnitude)+self.jaw_gain*(raw_logits-reference)
        return raw_logits,effective

    @torch.no_grad()
    def forward(self,observation):
        # The new learner's last context value is its physical residual gain.
        # The old actor was trained on a fixed goal radius; preserve that
        # literal input while the new Q/actor record their new action context.
        source_observation=observation.clone();source_observation[:,-1]=.05
        normalized=self.normalizer(source_observation)
        mean=self.actor.network(normalized).chunk(2,-1)[0]
        requested=mean.tanh()
        reference=self.bc_prior.agent.act(observation[:,:self.feature_width],deterministic=True)[:,list(GOAL_COLUMNS)]
        requested[:,:19]=torch.maximum(reference[:,:19]-.05,
                                      torch.minimum(requested[:,:19],reference[:,:19]+.05)).clamp(-1,1)
        _,logits=self.jaw_parameters(normalized)
        requested[:,19:]=torch.where(self.gate.near(observation),torch.where(logits>0,1.,-1.),-1.)
        return goal_body_command(self.coordinates,observation,self.center+self.scale*requested)


class PhysicalBodyProjector:
    name='physical_body_delta_servo_residual_0p5_frozen_goal_actor_v1'
    free_grippers=True
    def __init__(self,prior,gain=.5):
        if gain not in (.5,2.):raise ValueError('Physical residual gain must be .5 or full-range2')
        self.prior,self.gain=prior,gain
        if gain==2.:self.name='physical_body_delta_servo_residual_full_range_2_frozen_goal_actor_v1'
        self.gate=GoalGripperProjector()

    def __call__(self,observation,requested):
        baseline=self.prior(observation)
        result=requested.clone()
        result[:,:19]=(baseline[:,:19]+self.gain*requested[:,:19]).clamp(-1,1)
        result[:,19:]=torch.where(self.gate.near(observation),requested[:,19:],-1.)
        return result

    def entropy_mask(self,observation):
        result=torch.ones(len(observation),21,device=observation.device,dtype=observation.dtype)
        result[:,19:]=self.gate.near(observation)
        return result
