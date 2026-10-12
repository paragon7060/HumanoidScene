"""Explicit six-scope continuation of the unchanged upright grasp SAC task.

Only learner head ownership changes. Closed-source replay is a resume input,
never diagnostic data; the command-line preparation gate checks termination
and final backup before loading it. Independent FINAL is not used here.
"""
from copy import deepcopy
import re

import torch

from .actor_train_memory import structure_sha256
from .region_size_actor import SCOPES, SOURCE_HEADS, split_region_size_actor_and_Adam
from .urdf_upright_contact_sac import URDFIndependentHandCloseSACPilot, validate_independent_hand_close_state
from .perceived_contact_exploration import contact_statistics

SOURCE_ARTIFACT = URDFIndependentHandCloseSACPilot.artifact_type
ARTIFACT = 'staged_actual_flap_URDF_upright_region_size_independent_hand_close_sac_v10'
FORMAT = 'closed_compatible_region_size_actor_Adam_continuation_v1'
SUFFIXES = ('0.weight', '0.bias', '2.weight', '2.bias', '4.weight', '4.bias')


def region_size_actor_contract():
    return dict(name='independent_perceived_rack_region_and_box_type_SAC_heads_v1',
                scopes=[list(s) for s in SCOPES], source_head_indices=list(SOURCE_HEADS),
                region_token_columns=[94, 98], box_type_token_columns=[89, 91],
                type_routing='nearest_known_frozen_normalized_one_hot_pattern',
                actor_normalizer_observations_action_decoder_unchanged=True,
                shared_Q_entropy_and_success_objectives_unchanged=True,
                independent_Adam_and_guard_partition_per_scope=True,
                fixed_whole_success_cohort_check_required=True,
                task_physics_reward_randomization_success_safety_curriculum_unchanged=True)


def _legacy_contract(contract):
    result = dict(contract)
    result.update(name=SOURCE_ARTIFACT,
                  regional_actor='independent_perceived_rack_region_heads_unchanged_routing_v1')
    return result


def _patterns(model):
    mean, scale = model['actor_normalizer.mean'], model['actor_normalizer.var'].clamp_min(1e-4).sqrt()
    return ((torch.eye(2, device=mean.device, dtype=mean.dtype)-mean[89:91])/scale[89:91]).clamp(-10, 10)


def validate_region_size_state(state):
    if state.get('artifact_type') != ARTIFACT \
            or state.get('goal_contract', {}).get('regional_actor') != region_size_actor_contract() \
            or state.get('goal_contract', {}).get('name') != ARTIFACT:
        raise ValueError('Region-size continuation requires its explicit learner architecture')
    model = state['model']
    origin = state.get('region_size_initialization', {})
    fields = {'format', 'source_artifact_type', 'source_checkpoint_SHA256', 'source_head_indices',
              'source_actor_updates', 'source_critic_updates', 'actor_normalizer_SHA256',
              'source_model_SHA256', 'source_all_optimizers_SHA256', 'source_success_cohort_SHA256',
              'initial_head_tensor_SHA256', 'source_guard_statistics',
              'source_cumulative_counters_NOT_new_updates', 'no_transition_relabel_or_goal_mapping_change'}
    if set(origin) != fields or any(not re.fullmatch('[0-9a-f]{64}', origin[k]) for k in fields if k.endswith('_SHA256')
            and k != 'initial_head_tensor_SHA256') \
            or any(type(origin[k]) is not int or origin[k] < 0 for k in ('source_actor_updates', 'source_critic_updates')) \
            or origin['source_cumulative_counters_NOT_new_updates'] is not True \
            or origin['no_transition_relabel_or_goal_mapping_change'] is not True:
        raise ValueError('Region-size continuation requires complete explicit migration provenance')
    if origin.get('format') != FORMAT or origin.get('source_artifact_type') != SOURCE_ARTIFACT \
            or not re.fullmatch('[0-9a-f]{64}', origin.get('source_checkpoint_SHA256', '')) \
            or origin.get('source_head_indices') != list(SOURCE_HEADS) \
            or origin.get('source_actor_updates', -1) > state.get('actor_updates', -1) \
            or origin.get('source_critic_updates', -1) > state.get('critic_updates', -1) \
            or origin.get('actor_normalizer_SHA256') != structure_sha256(
                {k: v for k, v in model.items() if k.startswith('actor_normalizer.')}):
        raise ValueError('Region-size continuation provenance or frozen normalizer differs')
    hidden = state['config']['hidden']
    shapes = ((hidden, 518), (hidden,), (hidden, hidden), (hidden,), (42, hidden), (42,))
    expected = {f'actor.network.heads.{i}.{name}': shape
                for i in range(6) for name, shape in zip(SUFFIXES, shapes)}
    actual = {k: v for k, v in model.items() if k.startswith('actor.network.heads.')}
    if set(actual) != set(expected) or any(not isinstance(v, torch.Tensor)
            or tuple(v.shape) != expected[k] or not torch.isfinite(v).all() for k, v in actual.items()):
        raise ValueError('Region-size actor requires six finite complete original-shape heads')
    if len(state.get('optimizers', [])) != 4:
        raise ValueError('Region-size continuation requires all four original optimizer states')
    optimizer = state['optimizers'][0]
    if len(optimizer['param_groups']) != 1 or optimizer['param_groups'][0]['params'] != list(range(36)) \
            or not set(optimizer['state']).issubset(range(36)):
        raise ValueError('Region-size actor Adam parameter ordering differs')
    for index, moments in optimizer['state'].items():
        shape = shapes[index % 6]
        if not {'step', 'exp_avg', 'exp_avg_sq'}.issubset(moments) or any(
                not isinstance(v, torch.Tensor) or not torch.isfinite(v).all()
                or (v.numel() != 1 if k == 'step' else tuple(v.shape) != shape)
                for k, v in moments.items()):
            raise ValueError('Region-size actor Adam contains nonfinite or malformed moments')
    if state['actor_updates'] == origin['source_actor_updates']:
        hashes = [structure_sha256({suffix: model[f'actor.network.heads.{i}.{suffix}']
                                   for suffix in SUFFIXES}) for i in range(6)]
        if hashes != origin.get('initial_head_tensor_SHA256'):
            raise ValueError('Unchanged actor counter has different migrated head tensors')
    patterns = _patterns(model)
    lookup = torch.tensor([[4, 0], [1, 5], [2, -1], [3, -1]], device=patterns.device)
    if torch.equal(patterns[0], patterns[1]) or not torch.equal(
            model.get('actor.network.type_patterns', torch.empty(0)), patterns) \
            or not torch.equal(model.get('actor.network.head_lookup', torch.empty(0)), lookup):
        raise ValueError('Region-size routing differs from its unchanged frozen normalization')
    # Reuse every original physical/support/exploration/retention validator.
    # This view changes architecture metadata only and never mutates state.
    old_model = {k: v for k, v in model.items() if k not in {
        'actor.network.type_patterns', 'actor.network.head_lookup'}
        and not k.startswith(('actor.network.heads.4.', 'actor.network.heads.5.'))}
    legacy = state | dict(artifact_type=SOURCE_ARTIFACT,
                         goal_contract=_legacy_contract(state['goal_contract']), model=old_model)
    # The legacy validator constructs a temporary CPU frozen actor; its
    # random initialization is overwritten by saved weights. Keep that
    # disposable construction from advancing the caller's RNG.
    with torch.random.fork_rng(devices=[]):
        validate_independent_hand_close_state(legacy)


def migrate_region_size_checkpoint(source, source_SHA256):
    """Pure checkpoint preparation; no replay/file/GPU access or optimizer step."""
    if source.get('artifact_type') != SOURCE_ARTIFACT or 'TRAIN_success_candidate' in source \
            or source.get('inference_only') or not re.fullmatch('[0-9a-f]{64}', source_SHA256):
        raise ValueError('Size continuation needs a compatible full checkpoint with known provenance')
    with torch.random.fork_rng(devices=[]):
        validate_independent_hand_close_state(source)
    result = deepcopy(source)
    model = result['model']
    for new_index, old_index in enumerate(SOURCE_HEADS):
        for suffix in SUFFIXES:
            model[f'actor.network.heads.{new_index}.{suffix}'] = \
                source['model'][f'actor.network.heads.{old_index}.{suffix}'].detach().clone()
    model['actor.network.type_patterns'] = _patterns(model).clone()
    model['actor.network.head_lookup'] = torch.tensor([[4, 0], [1, 5], [2, -1], [3, -1]],
                                                     device=model['actor.network.type_patterns'].device)
    old = source['optimizers'][0]
    if len(old['param_groups']) != 1 or len(old['param_groups'][0]['params']) != 24 \
            or len(set(old['param_groups'][0]['params'])) != 24:
        raise ValueError('Size continuation requires the full four-head Adam parameter order')
    ids = old['param_groups'][0]['params']
    optimizer = deepcopy(old)
    optimizer['state'] = {}
    optimizer['param_groups'][0]['params'] = list(range(36))
    for new_index, old_index in enumerate(SOURCE_HEADS):
        for offset in range(6):
            old_id = ids[6*old_index+offset]
            if old_id in old['state']:
                optimizer['state'][6*new_index+offset] = deepcopy(old['state'][old_id])
    result['optimizers'][0] = optimizer
    result['region_size_initialization'] = dict(format=FORMAT, source_artifact_type=SOURCE_ARTIFACT,
        source_checkpoint_SHA256=source_SHA256, source_head_indices=list(SOURCE_HEADS),
        source_actor_updates=source['actor_updates'], source_critic_updates=source['critic_updates'],
        actor_normalizer_SHA256=structure_sha256({k: v for k, v in model.items() if k.startswith('actor_normalizer.')}),
        source_model_SHA256=structure_sha256(source['model']),
        source_all_optimizers_SHA256=structure_sha256(source['optimizers']),
        source_success_cohort_SHA256=structure_sha256(source['success_guard_memory']),
        initial_head_tensor_SHA256=[structure_sha256({suffix: model[f'actor.network.heads.{i}.{suffix}']
            for suffix in SUFFIXES}) for i in range(6)],
        source_guard_statistics=deepcopy(source['success_guard_statistics']),
        source_cumulative_counters_NOT_new_updates=True,
        no_transition_relabel_or_goal_mapping_change=True)
    result['artifact_type'] = ARTIFACT
    result['goal_contract'].update(name=ARTIFACT, regional_actor=region_size_actor_contract())
    validate_region_size_state(result)
    return result


def migrate_region_size_resume_input(experience, source, destination):
    """Retag only compatible closed-source resume metadata; retain exact rows.

    Tensors are read-only here. The actual learner copies them into its replay
    storage. The command-line caller must first verify all writers stopped and
    final Drive backup; this function never reads or writes a raw data file.
    """
    validate_region_size_state(destination)
    if experience.get('goal_contract') != source['goal_contract'] \
            or _legacy_contract(destination['goal_contract']) != source['goal_contract'] \
            or destination['region_size_initialization']['source_model_SHA256'] != structure_sha256(source['model']) \
            or destination['region_size_initialization']['source_all_optimizers_SHA256'] != structure_sha256(source['optimizers']) \
            or structure_sha256(experience.get('successful_train_transitions')) \
               != structure_sha256(source.get('successful_train_transitions')):
        raise ValueError('Size continuation replay must match the actual closed source checkpoint')
    return experience | dict(goal_contract=deepcopy(destination['goal_contract']),
                             region_size_initialization=deepcopy(destination['region_size_initialization']))


class RegionSizeIndependentHandCloseSACPilot(URDFIndependentHandCloseSACPilot):
    artifact_type = ARTIFACT

    def __init__(self, *args, checkpoint=None, **kwargs):
        if checkpoint is None:
            raise ValueError('Region-size SAC requires explicit compatible checkpoint migration')
        super().__init__(*args, checkpoint=checkpoint, **kwargs)

    def validate_saved_state(self, state):
        validate_region_size_state(state)
        self.region_size_initialization = deepcopy(state['region_size_initialization'])
        self._contact_statistics = contact_statistics(state['perceived_contact_statistics'])

    def configure_controller(self, saved):
        super().configure_controller(saved)
        split_region_size_actor_and_Adam(self.agent)

    @property
    def contract(self):
        result = super().contract
        return result if self._URDF_configuring_source else result | dict(regional_actor=region_size_actor_contract())

    def checkpoint_extras(self):
        return super().checkpoint_extras() | dict(region_size_initialization=deepcopy(self.region_size_initialization))

    def experience_extras(self):
        return super().experience_extras() | dict(region_size_initialization=deepcopy(self.region_size_initialization))

    def restore_experience_extras(self, state):
        super().restore_experience_extras(state)
        if state.get('region_size_initialization') != self.region_size_initialization:
            raise ValueError('Region-size checkpoint and replay continuation origins differ')

    def report(self):
        return super().report() | dict(region_size_actor=region_size_actor_contract(),
                                       region_size_initialization=deepcopy(self.region_size_initialization))
