"""Fresh SAC in known arm joint bounds, around an actor-only regional source.

All actor/replay goals are bounded in the NEW coordinates. The frozen source
still evaluates its ORIGINAL goal mapping; only its resulting body targets are
converted. Source Q, replay, rewards and privileged contact teachers are absent.
"""
from copy import deepcopy
from dataclasses import replace

import torch
from torch import nn

from ...algorithms.common import ObservationNormalizer
from ...algorithms.sac import SquashedActor
from .actor_train_memory import structure_sha256
from .conservative_servo_retention import conservative_actor_contract, ACTOR_LR
from .regional_actor_servo import (
    RegionalActorMemorySACPilot, RegionRoutedNetwork, install_regional_actor,
    validate_regional_actor_state,
)
from .servo_success_retention import ServoRetentionGentleSACPilot
from .servo_critic import BodyServoCriticEncoder
from .tensor_arm_kinematics import TensorArmKinematics

SOURCE_FORMAT = 'frozen_original_regional_actor_only_for_URDF_goal_SAC_v1'


def urdf_goal_coordinates(center, scale):
    """Change only 14 absolute arm targets; physical decoder is untouched."""
    center, scale = torch.as_tensor(center).clone(), torch.as_tensor(scale).clone()
    if center.shape != (21,) or scale.shape != (21,) or not torch.isfinite(center).all() \
            or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('Finite original 21-D goal normalization required')
    kinematics = TensorArmKinematics(device=center.device, dtype=center.dtype)
    lower, upper = kinematics.lower.flatten() + .01, kinematics.upper.flatten() - .01
    center[1:15] = (lower + upper) / 2
    scale[1:15] = (upper - lower) / 2
    return center, scale


def convert_original_body_goals(goals, source_center, source_scale, center, scale):
    if goals.ndim != 2 or goals.shape[1] != 19 or not torch.isfinite(goals).all():
        raise ValueError('Finite original absolute body goals required')
    source_center, source_scale, center, scale = [torch.as_tensor(v).to(goals)[:19]
        for v in (source_center, source_scale, center, scale)]
    result = (source_center + source_scale * goals - center) / scale
    # All non-arm coordinates keep precisely the original representation.
    result = torch.cat((goals[:, :1], result[:, 1:15].clamp(-1, 1), goals[:, 15:]), -1)
    return result


def regional_source_snapshot(state, checkpoint_SHA256):
    if state.get('artifact_type') != RegionalActorMemorySACPilot.artifact_type:
        raise ValueError('URDF initialization requires the compatible regional actor source')
    validate_regional_actor_state(state)
    contract = state['goal_contract']
    if (contract.get('actor_dim'), contract.get('critic_dim'), state.get('action_dim')) != (518, 578, 21) \
            or contract.get('fixed_prior_radius') != .30 or contract.get('body_correction_radius') != .30 \
            or len(checkpoint_SHA256) != 64:
        raise ValueError('Source perception, radius or checkpoint provenance differs')
    return dict(format=SOURCE_FORMAT, source_artifact_type=state['artifact_type'],
        source_checkpoint_SHA256=checkpoint_SHA256,
        source_actor_updates=state['actor_updates'], source_goal_contract=deepcopy(contract),
        source_hidden=state['config']['hidden'],
        source_frozen_body_anchor_SHA256=structure_sha256(state['body_anchor_state']),
        model={k: v.detach().clone() for k, v in state['model'].items()
            if k.startswith(('actor.', 'actor_normalizer.'))},
        source_Q_replay_reward_entropy_and_optimizers_imported=False)


def frozen_regional_source(snapshot, device):
    required = {'format', 'source_artifact_type', 'source_checkpoint_SHA256',
        'source_actor_updates', 'source_goal_contract', 'source_hidden',
        'source_frozen_body_anchor_SHA256', 'model',
        'source_Q_replay_reward_entropy_and_optimizers_imported'}
    if set(snapshot) != required or snapshot.get('format') != SOURCE_FORMAT \
            or snapshot.get('source_artifact_type') != RegionalActorMemorySACPilot.artifact_type \
            or snapshot.get('source_Q_replay_reward_entropy_and_optimizers_imported') is not False \
            or len(snapshot.get('source_checkpoint_SHA256', '')) != 64:
        raise ValueError('Malformed frozen regional actor-only source')
    normalizer = ObservationNormalizer(518).to(device)
    normalization = {k.removeprefix('actor_normalizer.'): v
        for k, v in snapshot['model'].items() if k.startswith('actor_normalizer.')}
    normalizer.load_state_dict(normalization)
    actor = SquashedActor(518, 21, snapshot['source_hidden']).to(device)
    actor.network = RegionRoutedNetwork(actor.network, normalizer)
    model = nn.ModuleDict(dict(actor=actor, actor_normalizer=normalizer))
    expected = model.state_dict()
    if set(expected) != set(snapshot['model']) or any(
            v.shape != expected[k].shape or not torch.isfinite(v).all()
            for k, v in snapshot['model'].items()):
        raise ValueError('Frozen regional actor shapes or finite tensors differ')
    model.load_state_dict(snapshot['model'])
    for key, value in (('region_mean', normalizer.mean[94:98]),
            ('region_scale', normalizer.var[94:98].clamp_min(1e-4).sqrt())):
        if not torch.equal(getattr(actor.network, key), value):
            raise ValueError('Frozen source routing differs from its observation normalization')
    model.requires_grad_(False)
    return model


@torch.no_grad()
def original_regional_body_goal(raw, nominal_body, snapshot, source_model):
    source_raw = raw.clone()
    source_raw[:, -1] = snapshot['source_goal_contract']['fixed_prior_radius']
    mean = source_model['actor'].network(source_model['actor_normalizer'](source_raw)).chunk(2, -1)[0][:, :19]
    available = (1 - nominal_body.abs()).clamp(min=0, max=.30)
    return nominal_body + available * mean.tanh()


def urdf_regional_contract():
    return dict(name='fresh_URDF_normalized_arm_goals_frozen_regional_actor_v1',
        arm_goal_columns=list(range(1, 15)), arm_joint_margin_rad=.01,
        arm_goal_normalization='known_S63_URDF_midpoint_and_half_range',
        non_arm_goal_normalization_unchanged=True,
        actor_and_replay_goals_bounded_minus_one_plus_one=True,
        frozen_source_evaluated_in_original_goal_coordinates=True,
        source_arm_anchor_clamped_only_to_known_URDF_margin=True,
        source_Q_replay_rewards_entropy_and_optimizers_imported=False,
        diagnostic_teacher_goals_and_privileged_contact_NOT_used=True,
        production_physical_decoder_and_limits_unchanged=True,
        midpoint_perception_and_original_task_DR_success_safety_unchanged=True,
        fresh_Q_and_actual_future_TRAIN_only=True)


def validate_urdf_regional_state(state):
    contract = state.get('goal_contract', {})
    source = state.get('URDF_regional_source', {})
    if state.get('artifact_type') != URDFRegionalGoalSACPilot.artifact_type \
            or contract.get('URDF_regional_goals') != urdf_regional_contract() \
            or (contract.get('actor_dim'), contract.get('critic_dim'), state.get('action_dim')) != (518, 578, 21) \
            or contract.get('frozen_regional_source_checkpoint_SHA256') != source.get('source_checkpoint_SHA256') \
            or contract.get('new_coordinate_Q_replay_required') is not True \
            or contract.get('actor_training_memory_imported') is not False \
            or state.get('actor_training_memory') is not None:
        raise ValueError('URDF state has incompatible coordinates or imported training data')
    old = source['source_goal_contract']
    if old['goal_center'] != contract['source_goal_center'] or old['goal_scale'] != contract['source_goal_scale'] \
            or old['physical_contract'] != contract['physical_contract'] \
            or old['shelf_templates'] != contract['shelf_templates'] \
            or source['source_frozen_body_anchor_SHA256'] != structure_sha256(state['body_anchor_state']):
        raise ValueError('URDF state source, physical contract or frozen anchor differs')
    center, scale = urdf_goal_coordinates(old['goal_center'], old['goal_scale'])
    if center.tolist() != contract['goal_center'] or scale.tolist() != contract['goal_scale']:
        raise ValueError('Saved arm targets do not use the known URDF normalization')
    frozen_regional_source(source, 'cpu')
    for name, expected in (
            ('region_mean', state['model']['actor_normalizer.mean'][94:98]),
            ('region_scale', state['model']['actor_normalizer.var'][94:98].clamp_min(1e-4).sqrt())):
        if not torch.equal(state['model']['actor.network.' + name], expected):
            raise ValueError('Learned regional routing differs from frozen normalization')


class URDFRegionalGoalSACPilot(ServoRetentionGentleSACPilot):
    artifact_type = 'staged_actual_flap_URDF_regional_goal_servo_retention_sac_v1'

    def __init__(self, *args, checkpoint=None, device='cpu', regional_source=None, **kwargs):
        saved = torch.load(checkpoint, map_location=device, weights_only=True) if checkpoint else None
        if saved:
            if regional_source is not None:
                raise ValueError('URDF continuation already owns its frozen source')
            regional_source = saved.get('URDF_regional_source')
            validate_urdf_regional_state(saved)
        if regional_source is None:
            raise ValueError('An explicit actor-only original regional source is required')
        self.URDF_regional_source = deepcopy(regional_source)
        self._source_goal_center = self._source_goal_scale = None
        self._URDF_configuring_source = True
        if not checkpoint:
            kwargs.setdefault('replay_capacity', 2000000)
        super().__init__(*args, checkpoint=checkpoint, device=device, **kwargs)

    def learning_config(self, config):
        return replace(super().learning_config(config), actor_lr=ACTOR_LR)

    def configure_controller(self, saved):
        self._source_goal_center, self._source_goal_scale = self.center.clone(), self.scale.clone()
        source = self.URDF_regional_source['source_goal_contract']
        if source['goal_center'] != self.center.tolist() or source['goal_scale'] != self.scale.tolist() \
                or self.URDF_regional_source['source_frozen_body_anchor_SHA256'] != structure_sha256(self.body_anchor_state) \
                or source['source_warm_start'] != self.warm_start.contract \
                or source['physical_contract'] != self.physical_contract \
                or source['shelf_templates'] != self.stage.templates:
            raise ValueError('Original source goals, physical contract, anchor or stages differ')
        super().configure_controller(saved)
        self.source_region_actor = frozen_regional_source(self.URDF_regional_source, self.device)
        self.agent.actor_normalizer.load_state_dict(self.source_region_actor['actor_normalizer'].state_dict())
        install_regional_actor(self.agent)
        self.agent.actor.load_state_dict(self.source_region_actor['actor'].state_dict())
        with torch.no_grad():
            for head in self.agent.actor.network.heads:
                output = head[-1]
                output.weight[:19].zero_()
                output.bias[:19].zero_()
                output.weight[21:].zero_()
                output.bias[21:].fill_(torch.log(torch.tensor(self.agent.config.initial_policy_std)).item())
        self.center, self.scale = urdf_goal_coordinates(self.center, self.scale)
        self.agent.goal_servo_critic_encoder = BodyServoCriticEncoder(self.coordinates, self.center, self.scale)
        self._URDF_configuring_source = False

    @torch.no_grad()
    def executed_body_anchor(self, raw):
        nominal = super().executed_body_anchor(raw)
        original = original_regional_body_goal(raw, nominal, self.URDF_regional_source, self.source_region_actor)
        return convert_original_body_goals(original, self._source_goal_center, self._source_goal_scale,
            self.center, self.scale)

    @property
    def contract(self):
        contract = super().contract | dict(actor_update_step=conservative_actor_contract())
        if self._URDF_configuring_source:
            return contract
        return contract | dict(URDF_regional_goals=urdf_regional_contract(),
            body_controller='frozen_original_regional_actor_in_URDF_normalized_goals_plus_fresh_bounded_correction_v1',
            source_goal_center=self._source_goal_center.tolist(), source_goal_scale=self._source_goal_scale.tolist(),
            frozen_regional_source_checkpoint_SHA256=self.URDF_regional_source['source_checkpoint_SHA256'],
            initial_body_mean='zero_preserve_original_regional_source_physical_goals_within_URDF_margin',
            actor_training_memory_imported=False, new_coordinate_Q_replay_required=True,
            regional_actor='independent_perceived_rack_region_heads_unchanged_routing_v1')

    def checkpoint_extras(self):
        return super().checkpoint_extras() | dict(URDF_regional_source=deepcopy(self.URDF_regional_source))

    def experience_extras(self):
        return super().experience_extras() | dict(URDF_regional_source=deepcopy(self.URDF_regional_source))

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if structure_sha256(state.get('URDF_regional_source')) != structure_sha256(self.URDF_regional_source):
            raise ValueError('Checkpoint/replay frozen URDF source provenance differs')

    def report(self):
        return super().report() | dict(URDF_regional_goals=urdf_regional_contract(),
            source_actor_only=True, fresh_coordinate_Q=True)
