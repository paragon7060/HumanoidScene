"""Compare matching SAC critics with recorded, fully closed DEV trajectories.

This is a read-only diagnostic, not a new evaluation or a training input.
Q values are estimates of cumulative reward, not success probabilities.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch

from export_eval_q_videos import episode_values, restored_agent, sha256
from summarize_batched_staged_run import read_snapshot, supported_success


def require_closed_run(run):
    status = read_snapshot(run / 'status.json')
    verification = read_snapshot(run / 'verification.json')
    managed = read_snapshot(run.parent / 'status.json')
    if (status.get('status') != 'complete' or status.get('interrupted')
            or verification.get('training_exit_code') != 0
            or not verification.get('writers_stopped_at')
            or managed.get('training_exit_code') != 0
            or not managed.get('final_upload_verified')):
        raise ValueError('The original run and final checkpoint/log backup must be complete')
    for key in ('training_pid', 'supervisor_pid'):
        pid = managed.get(key)
        if type(pid) is not int or pid <= 0 or Path('/proc', str(pid)).exists():
            raise ValueError('Both original writers must have stopped before reading the HDF')
    if run.stat().st_uid != os.getuid():
        raise ValueError('Only our recorded run may be inspected')


def identity(path):
    stat = path.stat()
    if stat.st_uid != os.getuid() or not path.is_file():
        raise ValueError('A file owned by the current user is required')
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def summarize(records):
    groups = {}
    for record in records:
        key = record['category']
        groups.setdefault(key, []).append(record)
    result = {}
    for key, rows in groups.items():
        result[key] = dict(count=len(rows), **{
            field + '_median': float(np.median([r[field] for r in rows]))
            for field in ('Q_first', 'Q_last', 'observed_return_first',
                          'Q_mean_absolute_return_difference', 'reward_sum')})
    return result


def audit(run, checkpoint, matching, whole, wave):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This diagnostic requires CUDA_VISIBLE_DEVICES to be empty')
    torch.set_num_threads(1)
    require_closed_run(run)
    if (matching.get('wave') != wave or matching.get('split') != 'validation'
            or Path(matching['source_run']).resolve() != run
            or Path(matching['protected_checkpoint']).resolve() != checkpoint):
        raise ValueError('Matching model proof identifies a different DEV wave or source')
    checksum = sha256(checkpoint)
    if (checksum != matching['checkpoint_SHA256']
            or checksum != whole['matching_checkpoint_SHA256']
            or not whole.get('whole_original_DEV128_completed')
            or not whole.get('all6_scope')
            or not whole.get('actual_end_saved_model_and_normalizer_equal_protected_pre_DEV_model')
            or whole['run_directory_name'] != run.name):
        raise ValueError('The protected model must match the completed whole DEV evaluation')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state['actor_updates'], state['critic_updates']) != (
            whole['actor_updates'], whole['critic_updates']):
        raise ValueError('Evaluation and model counters differ')
    if state['goal_contract'] != read_snapshot(run / 'agent.yaml'):
        raise ValueError('Recorded and restored goal contracts differ')
    end_checkpoint = run / f"checkpoint_{state['critic_updates']:08d}.pt"
    if sha256(end_checkpoint) != whole['actual_end_saved_checkpoint_SHA256']:
        raise ValueError('The original end-of-evaluation checkpoint changed')
    end = torch.load(end_checkpoint, map_location='cpu', weights_only=True)
    if any(not torch.equal(v, end['model'][k]) for k, v in state['model'].items()):
        raise ValueError('The end-of-evaluation model differs from the protected model')
    rows = [x for x in read_snapshot(run / 'metrics.json')['outcomes'] if x['wave'] == wave]
    manifest = read_snapshot(run / 'manifest.json')
    if (len(rows) != 128 or manifest['layout_waves'][wave]['split'] != 'validation'
            or any(not x['complete'] and x['initial_layout_valid'] for x in rows)
            or {x['environment'] for x in rows} != set(range(128))):
        raise ValueError('Only a complete, original whole DEV128 wave may be inspected')
    expected = {'shelf_2_left/small': 16, 'shelf_2_right/small': 16,
                'shelf_2_left/medium': 16, 'shelf_2_right/medium': 16,
                'shelf_3_left/small': 32, 'shelf_3_right/small': 32}
    if Counter(x['layout']['target_region'] + '/' + x['layout']['target_box_type']
               for x in rows) != expected:
        raise ValueError('All six original region and size groups must be retained')
    proven = {x['environment']: x for x in whole['cases']}
    agent, prior = restored_agent(state)
    hdf = run / 'executed_transitions.hdf5'
    before = identity(hdf)
    hdf_checksum = sha256(hdf)
    records, traces, skipped = [], {}, []
    with h5py.File(hdf, 'r') as dataset:
        episodes = {(int(e.attrs['wave']), int(e.attrs['environment'])): e
                    for e in dataset['episodes'].values()}
        for row in sorted(rows, key=lambda x: x['environment']):
            index = row['environment']
            actual = row['result']
            if not row['initial_layout_valid']:
                skipped.append(dict(environment=index, reason='initial_invalid'))
                continue
            episode = episodes[wave, index]
            if (json.loads(episode.attrs['layout_json']) != row['layout']
                    or row['layout'] != manifest['layout_waves'][wave]['layouts'][index]['layout']
                    or bool(episode.attrs['success']) != supported_success(row)):
                raise ValueError('Episode and actual whole DEV outcome differ')
            base = actual.get('staged_base', {})
            if base.get('phase') != 'held_grasp':
                skipped.append(dict(environment=index, reason='never_entered_held_grasp',
                                    actual_category=proven[index]['category']))
                continue
            try:
                arrays, start, error = episode_values(episode, actual, state, agent, prior)
            except ValueError as error:
                if not str(error).startswith('Policy goals do not match the executed commands:'):
                    raise
                # Keep the original command verification tolerance. Account
                # for this requested case, but publish no unverified Q values.
                skipped.append(dict(environment=index,
                    reason='restored_command_precision_verification_failed',
                    actual_category=proven[index]['category'], detail=str(error)))
                continue
            q = arrays['q_min'][start:]
            observed = arrays['observed_return'][start:]
            record = dict(environment=index, region=row['layout']['target_region'],
                box_type=row['layout']['target_box_type'], category=proven[index]['category'],
                initial_base_offset=row['layout'].get('base_offset'),
                actual_success=supported_success(row), steps=actual['steps'],
                manipulation_start_control_step=start + 1,
                max_reconstructed_executed_command_error=error,
                Q_first=float(q[0]), Q_last=float(q[-1]),
                Q_range=[float(q.min()), float(q.max())],
                observed_return_first=float(observed[0]),
                Q_mean_absolute_return_difference=float(np.abs(q-observed).mean()),
                reward_sum=float(arrays['reward'].sum()),
                actual_terminal={k: actual.get(k) for k in (
                    'pinching', 'stable_hands', 'opposing_flaps', 'proof_lift',
                    'hold_time_s', 'rack_clearance_m', 'unsafe_causes')})
            records.append(record)
            # Scalar diagnostic traces contain no raw observation/action payload.
            traces[str(index)] = [dict(control_step=i+1,
                q_min=None if i < start else float(arrays['q_min'][i]),
                reward=float(arrays['reward'][i]),
                observed_return=float(arrays['observed_return'][i]))
                for i in range(len(arrays['reward']))]
    if identity(hdf) != before or sha256(checkpoint) != checksum:
        raise ValueError('Closed sources changed during this read-only diagnostic')
    if len(records) + len(skipped) != 128:
        raise ValueError('Every requested condition must be accounted for')
    proof = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='matching_closed_DEV_critic_diagnostic_NOT_new_evaluation',
        source_run_directory_name=run.name, wave=wave,
        actor_updates=state['actor_updates'], critic_updates=state['critic_updates'],
        source_checkpoint_SHA256=checksum, source_HDF_SHA256=hdf_checksum,
        original_full_DEV_counts=whole['counts'], requested_conditions=128,
        analyzed_held_episodes=len(records), skipped=skipped,
        by_outcome=summarize(records), records=records,
        discount=state['config']['gamma'], reward_scale=state['config']['reward_scale'],
        entropy_backup=state['config']['entropy_backup'],
        Q_definition='matching online min(Q1,Q2) at recorded pre-action state and executed deterministic goal',
        Q_is_success_probability=False,
        observed_return_is_retrospective=True,
        realized_return_difference_alone_does_not_prove_Bellman_or_learning_error=True,
        analyzed_source_commands_match_restored_greedy_policy=True,
        command_verification_tolerance_NOT_relaxed=True,
        unverified_commands_excluded_from_Q_diagnostic_NOT_evaluation_denominator=True,
        closed_sources_unchanged=True, no_physics_replay=True, no_GPU_used=True,
        optimizer_updates=0, replay_rows_imported=0,
        raw_observation_action_payload_NOT_exported=True,
        original_full_DEV_score_unchanged=True, goal_not_complete=True)
    return proof, traces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--matching-model-proof', type=Path, required=True)
    parser.add_argument('--whole-eval-proof', type=Path, required=True)
    parser.add_argument('--wave', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise ValueError('A distinct diagnostic output directory is required')
    proof, traces = audit(args.run_dir.resolve(), args.checkpoint.resolve(),
        read_snapshot(args.matching_model_proof), read_snapshot(args.whole_eval_proof), args.wave)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / 'verification.json').write_text(json.dumps(proof, indent=2, allow_nan=False)+'\n')
    (args.output_dir / 'critic_traces.json').write_text(json.dumps(traces, allow_nan=False)+'\n')
    print(json.dumps({k: proof[k] for k in ('analyzed_held_episodes', 'by_outcome',
                                          'original_full_DEV_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
