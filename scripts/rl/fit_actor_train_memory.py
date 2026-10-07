#!/usr/bin/env python3
"""Fit a fresh SAC actor to audited, past successful TRAIN commands.

This is actor-only initialization, not SAC learning or physical evaluation.
Q networks, normalizers, the body anchor and empty SAC optimizers stay intact.
The existing successful-goal/servo/jaw objectives and sampling are reused.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes, statistics
from export_eval_q_videos import restored_agent
from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.algorithms.common import optimize
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import (
    ActorTrainMemory, compatibility_contract, structure_sha256,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import validate_conservative_actor_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS


@torch.no_grad()
def command_summary(agent, memory):
    result = {}
    for region in REGIONS:
        paths = []
        for e in memory.episodes[region]:
            raw, labels = e['actor_obs'], e['action']
            greedy, logits = [], []
            for chunk in raw.split(512):
                normalized = agent.actor_normalizer(agent.actor_features(chunk))
                greedy.append(agent.act(chunk, True))
                logits.append(agent.parameters_at(normalized)[2])
            greedy, logits = torch.cat(greedy), torch.cat(logits)
            paths.append(statistics(raw[-64:], labels[-64:], greedy[-64:], logits[-64:],
                agent.goal_servo_critic_encoder, agent.action_projector))
        fields = ('normalized_absolute_goal_MAE', 'actual_next_body_servo_command_MAE',
                  'greedy_both_closed_fraction_on_recorded_both_closed_rows')
        values = {key: sum(p[key] for p in paths if p[key] is not None)
                       /sum(p[key] is not None for p in paths)
                  for key in fields}
        values['close_probability_on_recorded_closed_rows'] = [
            sum(p['hands'][h]['policy_close_probability_on_recorded_closed_rows'] for p in paths)
            /len(paths) for h in range(2)]
        values['paths'] = len(paths)
        result[region] = values
    return result


def fit(initial, experience, *, steps, batch_size, learning_rate, seed):
    if type(steps) is not int or not 1 <= steps <= 10000 \
            or type(batch_size) is not int or not 64 <= batch_size <= 1024 or batch_size % 4 \
            or not 1e-6 <= learning_rate <= 3e-4 or type(seed) is not int or seed < 0:
        raise ValueError('Bounded steps/batch/learning-rate/seed required')
    require_pristine_learning_state(initial, experience,
        artifact_types=(ActorMemoryServoRetentionSACPilot.artifact_type,))
    if initial.get('actor_memory_initialization') is not None:
        raise ValueError('Actor initialization must start from the unfitted pristine source')
    validate_conservative_actor_state(initial, artifact_type=initial['artifact_type'])
    if structure_sha256(initial['actor_training_memory']) != structure_sha256(experience['actor_training_memory']):
        raise ValueError('Checkpoint and empty experience memory identity differs')
    bank = ActorTrainMemory(initial['actor_training_memory'],
        compatibility=compatibility_contract(initial['goal_contract']),
        frozen_anchor_SHA256=structure_sha256(initial['body_anchor_state']))
    agent, _ = restored_agent(initial)
    agent.actor.requires_grad_(True)
    if initial.get('success_jaw_balance') is not None:
        agent.success_jaw_balance_config = initial['success_jaw_balance']
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    optimizer = torch.optim.Adam(agent.actor.parameters(), lr=learning_rate)
    config = initial['successful_train_transitions']['config']
    goal_weight = config['actor_goal_mse_weight']/initial['goal_contract']['body_correction_radius']**2
    jaw_weight = config['actor_jaw_nll_weight']
    milestones = [dict(offline_actor_steps=0, command_summary=command_summary(agent, bank))]
    torch.manual_seed(seed)
    for step in range(1, steps+1):
        batch, stats = bank.sample_actor(batch_size, 'cpu', None)
        raw, labels = batch['actor_obs'], batch['action']
        normalized = agent.actor_normalizer(agent.actor_features(raw))
        mean, _, logits = agent.parameters_at(normalized)
        body = agent.success_body_loss(raw, mean.tanh(), labels)
        jaw, _ = agent.successful_jaw_loss(raw, logits, labels)
        loss = goal_weight*body+jaw_weight*jaw
        optimize(optimizer, loss, agent.actor.parameters())
        if step in (100, 250, 500, 1000) or step == steps:
            report = dict(offline_actor_steps=step, body_loss=float(body.detach()),
                jaw_loss=float(jaw.detach()), total_loss=float(loss.detach()),
                command_summary=command_summary(agent, bank), actor_memory_sampling=stats)
            milestones.append(report)
            print(json.dumps(report), flush=True)
    state = deepcopy(initial)
    state['model'] = {k:v.detach().cpu().clone() for k,v in agent.state_dict().items()}
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    assert all(torch.equal(v, state['model'][k]) for k,v in before.items() if not k.startswith('actor.'))
    assert identical(state['optimizers'], initial['optimizers'])
    assert state['goal_contract'] == initial['goal_contract']
    assert identical(state['body_anchor_state'], initial['body_anchor_state'])
    assert identical(state['actor_training_memory'], initial['actor_training_memory'])
    require_pristine_learning_state(state, experience,
        artifact_types=(ActorMemoryServoRetentionSACPilot.artifact_type,))
    return state, dict(steps=steps, batch_size=batch_size, actor_fit_learning_rate=learning_rate,
        seed=seed, configured_body_goal_servo_and_jaw_weights=[goal_weight,jaw_weight],
        milestones=milestones, all_non_actor_model_and_normalizer_tensors_unchanged=True,
        original_frozen_body_anchor_unchanged=True, SAC_all_four_optimizer_states_still_empty=True,
        SAC_actor_updates_and_Q_updates_still0=True, Q_replay_success_reward_and_return_banks_still_empty=True,
        old_actor_memory_only_no_Q_reward_or_evaluation_import=True,
        own_training_command_agreement_NOT_new_physical_success=True,
        actual_randomized_DEV_and_later_SAC_training_REQUIRED=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint', 'actor-memory', 'matching-memory-proof',
                 'training-manifest', 'waypoints', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--steps', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=227107)
    a = p.parse_args(); torch.set_num_threads(1)
    replay_path = a.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources = (a.initial_checkpoint, replay_path, a.actor_memory, a.matching_memory_proof,
               a.training_manifest, a.waypoints)
    data = {str(path): owned_stable_bytes(path) for path in sources}
    hashes = {path:hashlib.sha256(blob).hexdigest() for path,blob in data.items()}
    audit = json.loads(data[str(a.matching_memory_proof)])
    if hashes[str(a.actor_memory)] != audit['actor_only_dataset_SHA256'] or not all(audit.get(k) is True for k in (
            'all27_successful_TRAIN_paths_and12476_rows_own_HDF_and_manifest_exact',
            'actor_observation_and_goal_and_frozen_anchor_and_physics_success_control_compatible',
            'current_stricter_reset_relative_drop10cm_checked_entire_attempt',
            'source_TRAIN_and_new_TRAIN_and_all_DEV_FINAL_seeds_disjoint', 'no_active_HDF_or_replay_read')):
        raise ValueError('Full closed physical successful TRAIN audit is required')
    initial = torch.load(io.BytesIO(data[str(a.initial_checkpoint)]), map_location='cpu', weights_only=True)
    experience = torch.load(io.BytesIO(data[str(replay_path)]), map_location='cpu', weights_only=True)
    memory = torch.load(io.BytesIO(data[str(a.actor_memory)]), map_location='cpu', weights_only=True)
    if structure_sha256(memory) != structure_sha256(initial['actor_training_memory']):
        raise ValueError('Pristine actor memory differs from physical audit dataset')
    physical = json.loads(data[str(a.training_manifest)])
    expected = initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in expected} != expected:
        raise ValueError('Physical manifest differs from original actor coordinates')
    state, fit_proof = fit(initial, experience, steps=a.steps, batch_size=a.batch_size,
        learning_rate=a.learning_rate, seed=a.seed)
    origin = dict(kind='offline_actor_commands_from_own_successful_TRAIN_v1',
        source_initial_checkpoint_SHA256=hashes[str(a.initial_checkpoint)],
        actor_memory_structure_SHA256=structure_sha256(memory),
        actor_memory_file_SHA256=hashes[str(a.actor_memory)],
        frozen_body_anchor_SHA256=structure_sha256(state['body_anchor_state']),
        fitted_actor_model_SHA256=structure_sha256({k:v for k,v in state['model'].items() if k.startswith('actor.')}),
        offline_actor_steps=a.steps, fitting_seed=a.seed,
        Q_or_evaluation_imported=False, physical_generalization_NOT_proven=True)
    state['actor_memory_initialization'] = origin
    experience = deepcopy(experience)
    experience['actor_memory_initialization'] = deepcopy(origin)
    a.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, a.output_dir/'checkpoint_00000000.pt')
    torch.save(experience, a.output_dir/'staged_goal_experience.pt')
    assert identical(state, torch.load(a.output_dir/'checkpoint_00000000.pt', map_location='cpu', weights_only=True))
    assert hashes == {str(path):hashlib.sha256(owned_stable_bytes(path)).hexdigest() for path in sources}
    proof = dict(recorded_UTC=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        actor_memory_fit=fit_proof, artifact_type=state['artifact_type'],
        actor_memory_structure_SHA256=structure_sha256(memory),
        fitted_actor_model_SHA256=structure_sha256({k:v for k,v in state['model'].items() if k.startswith('actor.')}),
        original_reward_success_safety_control_and_box_base_dynamic_flap_randomization_preserved=True,
        offline_actor_initialization_NOT_SAC_training=True, physical_training_NOT_started=True,
        independent_FINAL_unused=True, goal_not_complete=True)
    for name, value in [('training_manifest.json', physical), ('waypoints.json', json.loads(data[str(a.waypoints)])),
            ('initialization_verification.json', proof), ('manifest.json', dict(artifact_type=state['artifact_type'],
                goal_contract=state['goal_contract'],initialization_verification=proof))]:
        (a.output_dir/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(a.output_dir),offline_actor_fit_only=True,
        fitted_actor_steps=a.steps,SAC_actor_Q_counters_and_reward_banks0=True,
        physical_generalization_NOT_verified=True)),flush=True)


if __name__ == '__main__':
    main()
