#!/usr/bin/env python3
"""Prepare independent regional SAC actors without importing component Q data."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes
from export_eval_q_videos import restored_agent
from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import (
    ConservativeServoRetentionSACPilot, validate_conservative_actor_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import (
    RegionalActorMemorySACPilot, install_regional_actor, regional_actor_contract,
    validate_regional_actor_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS


def component_contract(state):
    value = deepcopy(state['goal_contract'])
    # A verified actor-memory sampler changes future learning, not coordinates.
    value.pop('name')
    value.pop('actor_training_memory', None)
    return value


def prepare(initial, experience, components, hashes):
    require_pristine_learning_state(initial, experience,
        artifact_types=(ActorMemoryServoRetentionSACPilot.artifact_type,))
    if initial.get('actor_memory_initialization') is not None:
        raise ValueError('Regional preparation requires the original unfitted pristine initialization')
    if set(components) != set(REGIONS) or set(hashes) != set(REGIONS):
        raise ValueError('Exactly one compatible component is required for every rack region')
    if structure_sha256(initial['actor_training_memory']) != structure_sha256(experience['actor_training_memory']):
        raise ValueError('Checkpoint and experience actor-memory origins differ')
    allowed = (ActorMemoryServoRetentionSACPilot.artifact_type,
               ConservativeServoRetentionSACPilot.artifact_type)
    networks, origins = {}, {}
    for region in REGIONS:
        source = components[region]
        validate_conservative_actor_state(source, artifact_type=source['artifact_type'])
        if source['artifact_type'] not in allowed or component_contract(source) != component_contract(initial):
            raise ValueError('Regional component physical/action/controller contracts differ')
        for key in ('config', 'hybrid_contract', 'body_anchor_state', 'frozen_actor_prior', 'frozen_warm_start'):
            if not identical(source[key], initial[key]):
                raise ValueError('Regional component frozen coordinates, prior or learning configuration differ')
        norm = {k:v for k,v in source['model'].items() if k.startswith('actor_normalizer.')}
        original_norm = {k:v for k,v in initial['model'].items() if k.startswith('actor_normalizer.')}
        if not identical(norm, original_norm) or len(hashes[region]) != 64:
            raise ValueError('Regional component normalization or checkpoint identity differs')
        network = {k.removeprefix('actor.network.'):v for k,v in source['model'].items()
                   if k.startswith('actor.network.')}
        if not network or not all(torch.isfinite(v).all() for v in network.values()):
            raise ValueError('Finite regional actor network parameters required')
        networks[region] = network
        origins[region] = dict(checkpoint_SHA256=hashes[region],
            network_SHA256=structure_sha256(network), source_artifact_type=source['artifact_type'],
            source_actor_updates=source['actor_updates'], source_critic_updates=source['critic_updates'],
            offline_actor_initialization=deepcopy(source.get('actor_memory_initialization')),
            only_actor_network_imported=True, source_Q_optimizer_or_replay_imported=False)
    agent, _ = restored_agent(initial)
    install_regional_actor(agent)
    for index, region in enumerate(REGIONS):
        agent.actor.network.heads[index].load_state_dict(networks[region], strict=True)
    state, replay = deepcopy(initial), deepcopy(experience)
    state['model'] = {k:v.detach().cpu().clone() for k,v in agent.state_dict().items()}
    state['optimizers'][0] = agent.actor_optimizer.state_dict()
    state['artifact_type'] = RegionalActorMemorySACPilot.artifact_type
    origin = dict(kind='compatible_actor_only_region_components_v1', components=origins,
        frozen_body_anchor_SHA256=structure_sha256(initial['body_anchor_state']),
        actor_normalizer_SHA256=structure_sha256(original_norm),
        model_selection_on_DEV_NOT_evaluation_training=True,
        new_whole_randomized_DEV_required=True, independent_FINAL_unused=True)
    for value in (state, replay):
        value['goal_contract']['name'] = state['artifact_type']
        value['goal_contract']['regional_actor'] = regional_actor_contract()
        value['regional_actor_initialization'] = deepcopy(origin)
        value.pop('actor_memory_initialization', None)
    assert all(torch.equal(v, state['model'][k]) for k,v in initial['model'].items()
               if not k.startswith('actor.'))
    assert identical(initial['optimizers'][1:], state['optimizers'][1:])
    require_pristine_learning_state(state, replay, artifact_types=(state['artifact_type'],))
    validate_regional_actor_state(state)
    return state, replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('initial-checkpoint', 'training-manifest', 'waypoints', 'training-waves', 'output-dir'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--component', nargs=2, action='append', required=True, metavar=('REGION', 'CHECKPOINT'))
    args = p.parse_args(); torch.set_num_threads(1)
    if len(args.component) != 4 or {r for r,_ in args.component} != set(REGIONS):
        raise ValueError('Supply each of the four region names exactly once')
    paths = dict(args.component)
    sources = (args.initial_checkpoint, args.initial_checkpoint.parent/'staged_goal_experience.pt',
        args.training_manifest, args.waypoints, args.training_waves, *(Path(x) for x in paths.values()))
    data = {str(x):owned_stable_bytes(x) for x in sources}
    hashes = {k:hashlib.sha256(v).hexdigest() for k,v in data.items()}
    def checkpoint(path):
        return torch.load(io.BytesIO(data[str(path)]), map_location='cpu', weights_only=True)
    initial = checkpoint(args.initial_checkpoint)
    state, replay = prepare(initial, checkpoint(sources[1]),
        {r:checkpoint(path) for r,path in paths.items()}, {r:hashes[path] for r,path in paths.items()})
    physical = json.loads(data[str(args.training_manifest)])
    expected = initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in expected} != expected:
        raise ValueError('Physical training manifest differs from original coordinates')
    waves = json.loads(data[str(args.training_waves)])
    if waves[0]['split'] != 'validation' or len(waves[0]['layouts']) != 128:
        raise ValueError('Original full DEV128 requests must be present')
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    torch.save(state, args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay, args.output_dir/'staged_goal_experience.pt')
    assert identical(state, torch.load(args.output_dir/'checkpoint_00000000.pt', weights_only=True))
    assert hashes == {str(x):hashlib.sha256(owned_stable_bytes(x)).hexdigest() for x in sources}
    proof = dict(recorded_UTC=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        regional_actor_initialization=state['regional_actor_initialization'],
        critic_normalizers_Q_targets_frozen_body_anchor_and_prior_unchanged=True,
        all_SAC_counters_optimizer_states_Q_replay_and_reward_banks0=True,
        old_actor_only_TRAIN_memory_preserved=True, no_evaluation_rows_imported=True,
        original_full_DEV_and_TRAIN_requests_and_physical_task_preserved=True,
        component_scores_NOT_new_composite_physical_success=True,
        physical_evaluation_and_training_NOT_started=True, independent_FINAL_unused=True, goal_not_complete=True)
    for name, value in [('training_manifest.json', physical), ('training_waves.json', waves),
            ('DEV128.json', [waves[0]]), ('waypoints.json', json.loads(data[str(args.waypoints)])),
            ('initialization_verification.json', proof), ('manifest.json', dict(
                artifact_type=state['artifact_type'], goal_contract=state['goal_contract'],
                initialization_verification=proof))]:
        (args.output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir), four_independent_region_actors=True,
        all_Q_banks_and_SAC_counters0=True, physical_generalization_NOT_verified=True)), flush=True)


if __name__ == '__main__':
    main()
