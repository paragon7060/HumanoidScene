#!/usr/bin/env python3
"""Fork pristine conservative SAC with verified, actor-only past TRAIN memory."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import (
    ActorTrainMemory, actor_memory_contract, compatibility_contract, structure_sha256,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import (
    ConservativeServoRetentionSACPilot, validate_conservative_actor_state,
)


def prepare(initial, experience, memory, proof):
    require_pristine_learning_state(initial, experience,
        artifact_types=(ConservativeServoRetentionSACPilot.artifact_type,))
    validate_conservative_actor_state(initial)
    if 'actor_training_memory' in initial or 'actor_training_memory' in experience:
        raise ValueError('Actor memory initialization must be fresh')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import body_behavior_statistics
    if any(v.get('body_behavior_statistics') != body_behavior_statistics()
           for v in (initial, experience)):
        raise ValueError('Collection must not have started')
    required = ('writer_and_supervisor_closed_exit0_final_Drive_verified',
        'all27_successful_TRAIN_paths_and12476_rows_own_HDF_and_manifest_exact',
        'actor_observation_and_goal_and_frozen_anchor_and_physics_success_control_compatible',
        'current_stricter_reset_relative_drop10cm_checked_entire_attempt',
        'all_labels_fit_current_actor_bounds_and_jawgate',
        'all_absolute_goals_decode_to_actual_recorded_body_and_jaw_commands',
        'source_TRAIN_and_new_TRAIN_and_all_DEV_FINAL_seeds_disjoint',
        'only_actor_obs_and_action_and_success_provenance_exported',
        'no_critic_reward_next_state_or_terminal_label_exported', 'no_active_HDF_or_replay_read')
    compatible = compatibility_contract(initial['goal_contract'])
    anchor = structure_sha256(initial['body_anchor_state'])
    if any(proof.get(k) is not True for k in required) \
            or proof.get('actor_memory_compatibility') != compatible \
            or proof.get('frozen_body_anchor_SHA256') != anchor \
            or proof.get('source_checkpoint_SHA256') != memory.get('source_checkpoint_SHA256'):
        raise ValueError('Closed physical TRAIN audit does not match this actor memory')
    bank = ActorTrainMemory(memory, compatibility=compatible, frozen_anchor_SHA256=anchor)
    state, replay = deepcopy(initial), deepcopy(experience)
    state['artifact_type'] = ActorMemoryServoRetentionSACPilot.artifact_type
    for value in (state, replay):
        value['goal_contract']['name'] = state['artifact_type']
        value['goal_contract']['actor_training_memory'] = actor_memory_contract()
        value['actor_training_memory'] = deepcopy(bank.state())
    validate_conservative_actor_state(state, artifact_type=state['artifact_type'])
    assert identical(state['model'], initial['model']) and identical(state['optimizers'], initial['optimizers'])
    return state, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('initial-checkpoint', 'actor-memory', 'matching-memory-proof',
                'training-manifest', 'waypoints', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); torch.set_num_threads(1)
    replay_path = args.initial_checkpoint.parent / 'staged_goal_experience.pt'
    paths = (args.initial_checkpoint, replay_path, args.actor_memory,
             args.matching_memory_proof, args.training_manifest, args.waypoints)
    if any(not p.is_file() or p.is_symlink() or p.stat().st_uid != os.getuid() for p in paths):
        raise ValueError('Owned regular closed/pristine inputs required')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    proof = json.loads(args.matching_memory_proof.read_text())
    if hashes[str(args.actor_memory)] != proof['actor_only_dataset_SHA256'] \
            or hashes[str(args.initial_checkpoint)] != proof['validated_destination_initial_checkpoint_SHA256']:
        raise ValueError('Audit hashes differ from the supplied memory or initialization')
    initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    replay = torch.load(replay_path, map_location='cpu', weights_only=True)
    memory = torch.load(args.actor_memory, map_location='cpu', weights_only=True)
    state, experience = prepare(initial, replay, memory, proof)
    physical = json.loads(args.training_manifest.read_text())
    expected = initial['goal_contract']['physical_contract']
    if {k: physical.get(k) for k in expected} != expected:
        raise ValueError('Physical training manifest differs')
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    torch.save(state, args.output_dir / 'checkpoint_00000000.pt')
    torch.save(experience, args.output_dir / 'staged_goal_experience.pt')
    assert identical(state, torch.load(args.output_dir/'checkpoint_00000000.pt', weights_only=True))
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    verification = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        previous_successful_TRAIN_actor_memory=ActorTrainMemory(memory,
            compatibility=compatibility_contract(initial['goal_contract']),
            frozen_anchor_SHA256=structure_sha256(initial['body_anchor_state'])).report(),
        all_initial_models_normalizers_Gaussian_optimizers_unchanged=True,
        Q_replay_success_reward_and_return_banks_empty=True,
        only_old_actor_observations_and_executed_actions_imported=True,
        original_physics_reward_safety_success_and_randomization_preserved=True,
        physical_training_NOT_started=True, goal_not_complete=True)
    for name, data in [('training_manifest.json', physical),
            ('waypoints.json', json.loads(args.waypoints.read_text())),
            ('initialization_verification.json', verification),
            ('manifest.json', dict(artifact_type=state['artifact_type'], goal_contract=state['goal_contract'],
                                  initialization_verification=verification))]:
        (args.output_dir / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),
        actor_memory_rows=sum(len(e['action']) for e in memory['episodes']),
        Q_or_old_rewards_imported=False, physical_training_NOT_started=True)))


if __name__ == '__main__':
    main()
