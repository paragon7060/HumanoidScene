#!/usr/bin/env python3
"""Seed fresh servo-retained SAC from a matched policy, with real return credit."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    EPISODE_RETURN_VARIANT, MeasuredTrainCreditBank, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetentionGentleSACPilot
from kuavo_isaaclab_scene.rl.multi_box.geometry.box_drop import with_reset_drop_profile
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import KEYS
from prepare_actual_success_actor_tail import identical
from export_eval_q_videos import restored_agent


def prepare(initial, experience, source, *, reset_drop=False):
    """Import actor behavior only; neither past Q nor past reward rows enter."""
    if (initial.get('artifact_type') != ServoRetentionGentleSACPilot.artifact_type
            or source.get('artifact_type') != initial['artifact_type']
            or source['goal_contract'] != initial['goal_contract']
            or not identical(source['body_anchor_state'], initial['body_anchor_state'])
            or source['config'] != initial['config']):
        raise ValueError('Actor needs matching servo, goals, body anchor and Gaussian configuration')
    if (initial['actor_updates'] or initial['critic_updates'] or initial['model']['critic_normalizer.count']
            or len(initial['optimizers']) != 4 or any(o['state'] for o in initial['optimizers'])
            or any(initial['successful_train_transitions']['episodes'].values())
            or experience['goal_contract'] != initial['goal_contract']
            or set(experience.get('executed_goal_transitions', {})) != set(KEYS)
            or any(len(v) for v in experience['executed_goal_transitions'].values())
            or not identical(initial['successful_train_transitions'], experience['successful_train_transitions'])
            or experience.get('measured_train_credit') != initial.get('measured_train_credit')
            or experience['measured_train_credit_bank']['config'] != initial.get('measured_train_credit')
            or any(experience['measured_train_credit_bank']['episodes'].values())):
        raise ValueError('Fresh Q, four empty optimizers, replay and all reward-bearing banks are required')
    if any(not torch.isfinite(v).all() for value in (initial, source) for v in value['model'].values()):
        raise ValueError('Initial and source models must be finite')
    if not identical(source['frozen_actor_prior'], initial['frozen_actor_prior']):
        raise ValueError('Matching frozen gripper prior is required')
    state = deepcopy(initial)
    replay = deepcopy(experience)
    for key, value in source['model'].items():
        if key.startswith(('actor.', 'actor_normalizer.')):
            if key not in state['model'] or value.shape != state['model'][key].shape or not torch.isfinite(value).all():
                raise ValueError('Matching finite actor tensors are required')
            state['model'][key] = value.clone()
    state['frozen_actor_prior'] = deepcopy(source['frozen_actor_prior'])
    credit = measured_credit_config(EPISODE_RETURN_VARIANT)
    state['measured_train_credit'] = deepcopy(credit)
    replay['measured_train_credit'] = deepcopy(credit)
    replay['measured_train_credit_bank']['config'] = deepcopy(credit)
    if reset_drop:
        state['goal_contract']['physical_contract'] = with_reset_drop_profile(
            state['goal_contract']['physical_contract'])
        replay['goal_contract'] = deepcopy(state['goal_contract'])
    state['measured_train_credit_bank_report'] = MeasuredTrainCreditBank(
        state['goal_contract']['actor_dim'], state['goal_contract']['critic_dim'],
        state['config']['gamma'], credit).report()
    changed = {k for k in state['model'] if k.startswith(('actor.', 'actor_normalizer.'))}
    assert all(torch.equal(v, initial['model'][k]) for k, v in state['model'].items() if k not in changed)
    assert identical(state['optimizers'], initial['optimizers'])
    return state, replay


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint', 'actor-checkpoint', 'actor-capture-proof',
            'source-evaluation', 'training-manifest', 'waypoints', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reset-drop10', action='store_true', help='New MDP with reviewed reset-relative10cm guard')
    args = p.parse_args(); torch.set_num_threads(1)
    exp = args.initial_checkpoint.parent/'staged_goal_experience.pt'
    files = (args.initial_checkpoint, exp, args.actor_checkpoint, args.actor_capture_proof,
        args.source_evaluation, args.training_manifest, args.waypoints)
    if any(f.is_symlink() or not f.is_file() or f.stat().st_uid != os.getuid() for f in files):
        raise ValueError('Owned regular pristine and protected matching inputs are required')
    hashes = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    capture = json.loads(args.actor_capture_proof.read_text())
    evaluation = json.loads(args.source_evaluation.read_text())
    if (capture['checkpoint_SHA256'] != hashes[str(args.actor_checkpoint)]
            or evaluation['matching_checkpoint_SHA256'] != capture['checkpoint_SHA256']
            or not evaluation['full_original_DEV128_per_region32']
            or evaluation['summary']['supported_successes'] < 1):
        raise ValueError('An exact protected model and completed original DEV128 result are required')
    initial = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    source = torch.load(args.actor_checkpoint, map_location='cpu', weights_only=True)
    if (source['actor_updates'], source['critic_updates']) != (capture['actor_updates'], capture['critic_updates']):
        raise ValueError('Protected actor counters differ')
    state, replay = prepare(initial, torch.load(exp, map_location='cpu', weights_only=True), source,
        reset_drop=args.reset_drop10)
    physical = json.loads(args.training_manifest.read_text())
    original = initial['goal_contract']['physical_contract']
    if {k:physical.get(k) for k in original} != original:
        raise ValueError('Source physical manifest differs')
    if args.reset_drop10:
        physical['terminal_contract'] = deepcopy(state['goal_contract']['physical_contract']['terminal_contract'])
    episodes = [e for es in source['successful_train_transitions']['episodes'].values() for e in es]
    if not episodes:
        raise ValueError('Closed actual TRAIN states are required for action identity checks only')
    raw = torch.cat([e['rows']['actor_obs'][::10] for e in episodes])
    before, _ = restored_agent(source)
    after, _ = restored_agent(state)
    with torch.no_grad():
        assert torch.equal(before.act(raw, True), after.act(raw, True))
    assert all(hashes[str(f)] == hashlib.sha256(f.read_bytes()).hexdigest() for f in files)
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay, args.output_dir/'staged_goal_experience.pt')
    assert identical(state, torch.load(args.output_dir/'checkpoint_00000000.pt', map_location='cpu', weights_only=True))
    assert identical(replay, torch.load(args.output_dir/'staged_goal_experience.pt', map_location='cpu', weights_only=True))
    proof = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        source_actor_updates=source['actor_updates'], source_critic_updates_NOT_imported=source['critic_updates'],
        source_original_DEV128_safe_successes=evaluation['summary']['supported_successes'],
        actor_greedy_body_binary_jaws_and_normalizer_identity_actual_TRAIN_states=len(raw),
        initial_Q_targets_critic_normalizer_entropy_and_four_optimizers_unchanged=True,
        actor_Q_updates0_online_replay_success_and_measured_credit_banks0=True,
        old_reward_rows_NOT_imported_or_relabelled=True, measured_train_credit=state['measured_train_credit'],
        reset_relative_drop10cm=args.reset_drop10, original_reward_weights_controller_success_safety_randomization_preserved=True,
        actual_training_NOT_started=True, independent_FINAL_unused=True, goal_not_complete=True)
    for name, value in (('initialization_verification.json', proof), ('training_manifest.json', physical),
            ('waypoints.json', json.loads(args.waypoints.read_text())),
            ('manifest.json', dict(artifact_type=state['artifact_type'], goal_contract=state['goal_contract'], initialization_verification=proof))):
        (args.output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir), actual_identity_rows=len(raw),
        source_DEV_successes=evaluation['summary']['supported_successes'], new_actor_Q_banks0=True,
        critic_credit=EPISODE_RETURN_VARIANT, actual_training_NOT_started=True)), flush=True)


if __name__ == '__main__':
    main()
