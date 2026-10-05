#!/usr/bin/env python3
"""Join closed physical TRAIN paths to exact, matching recorded 21-D SAC rows.

Never invert 24-D commands into goals, change rewards, or import DEV/FINAL.
The new input enables an auxiliary learner objective and preserves all source
models, optimizers, physical replay, counters, and success-retention schedules.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import h5py
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    MeasuredTrainCreditBank, VARIANT, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import KEYS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--closed-run', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    source, run = args.checkpoint.resolve(), args.closed_run.resolve()
    managed = json.loads((run.parent/'status.json').read_text())
    if managed['phase'] != 'finished' or not managed.get('final_upload_verified') \
            or any(Path('/proc', str(managed[k])).exists() for k in ('training_pid', 'supervisor_pid')):
        raise ValueError('Native source must be closed and its final Drive backup verified')
    state = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    experience = torch.load(source.parent/'staged_goal_experience.pt',
        map_location='cpu', weights_only=True, mmap=True)
    if state['artifact_type'] != 'staged_actual_flap_bounded_actor_correction_hybrid_sac_v1' \
            or state['goal_contract'] != experience['goal_contract'] \
            or state.get('measured_train_credit') or experience.get('measured_train_credit'):
        raise ValueError('Expected matching original one-step actual-flap SAC and real replay')
    rows = experience['executed_goal_transitions']
    assert set(rows) == set(KEYS) and rows['critic_obs'].shape[1] == 577
    assert rows['actor_obs'].shape[1] == 518
    metrics = json.loads((run/'metrics.json').read_text())
    outcomes = {(r['wave'], r['environment']): r for r in metrics['outcomes']}
    # Fingerprints only narrow candidates. Every relevant recorded tensor,
    # reward, absorbing mask and held clock is checked exactly below.
    def fingerprint(a, b):
        return hashlib.sha256(a[:86].numpy().tobytes() + b[:86].numpy().tobytes()).digest()
    index = {}
    for i in range(len(rows['reward'])):
        index.setdefault(fingerprint(rows['critic_obs'][i], rows['next_critic_obs'][i]), []).append(i)
    config = measured_credit_config(VARIANT)
    bank = MeasuredTrainCreditBank(518, 577, state['config']['gamma'], config)
    accepted, rejected = [], []
    with h5py.File(run/'executed_transitions.hdf5', 'r') as f:
        manifest = json.loads(f.attrs['manifest_json'])
        physical = state['goal_contract']['physical_contract']
        if any(manifest['training_contract'].get(k) != v for k, v in physical.items()):
            raise ValueError('Native source physics/reward/safety differs from real SAC replay')
        for name, episode in f['episodes'].items():
            outcome = outcomes.get((int(episode.attrs['wave']), int(episode.attrs['environment'])))
            if outcome is None or outcome['split'] != 'train':
                continue
            result = outcome['result'] or {}
            start = (result.get('staged_base') or {}).get('manipulation_start')
            if start is None or not outcome['complete'] or result.get('numerical_failure') \
                    or result.get('invalid_reset') or not outcome['initial_layout_valid']:
                continue
            transitions = episode['transitions']
            assert json.loads(episode.attrs['layout_json']) == outcome['layout']
            native = {k: torch.from_numpy(transitions[k][:]) for k in (
                'actor_obs', 'critic_obs', 'next_actor_obs', 'next_critic_obs',
                'actor_supplemental', 'next_actor_supplemental', 'reward', 'terminated', 'truncated')}
            matches = []
            for j in range(start, len(native['reward'])):
                candidates = index.get(fingerprint(native['critic_obs'][j], native['next_critic_obs'][j]), [])
                found = []
                for i in candidates:
                    if rows['reward'][i] != native['reward'][j] \
                            or rows['terminated'][i] != (native['terminated'][j] | native['truncated'][j]):
                        continue
                    clock = min(j - start, state['goal_contract']['source_warm_start']['clock_horizon']) \
                        / state['goal_contract']['source_warm_start']['clock_horizon']
                    if abs(float(rows['critic_obs'][i, 530]) - clock) > 1e-6:
                        continue
                    comparisons = [
                        # Actor rows contain compact pose/clock features,
                        # not the original464-D telemetry. The full raw
                        # actor is present exactly in the recorded critic.
                        (rows['critic_obs'][i, :464], native['actor_obs'][j]),
                        (rows['next_critic_obs'][i, :464], native['next_actor_obs'][j]),
                        (rows['critic_obs'][i, :530], native['critic_obs'][j]),
                        (rows['next_critic_obs'][i, :530], native['next_critic_obs'][j]),
                        (rows['actor_obs'][i, 474:512], native['actor_supplemental'][j]),
                        (rows['next_actor_obs'][i, 474:512], native['next_actor_supplemental'][j]),
                        (rows['critic_obs'][i, 533:571], native['actor_supplemental'][j]),
                        (rows['next_critic_obs'][i, 533:571], native['next_actor_supplemental'][j]),
                    ]
                    if all(torch.equal(a, b) for a, b in comparisons):
                        found.append(i)
                if len(found) != 1:
                    rejected.append(dict(episode=name, reason='ambiguous_or_missing_exact_measured_goal_row',
                        native_row=j, matching_rows=len(found)))
                    break
                matches.append(found[0])
            else:
                if not matches:
                    continue
                path = {k: v[matches] for k, v in rows.items()}
                try:
                    bank.add_episode(path, outcome, source_run=run.name)
                except ValueError as error:
                    rejected.append(dict(episode=name, reason=str(error)))
                else:
                    accepted.append(dict(episode=name, region=outcome['layout']['target_region'],
                        successful=bool(result['success']), rows=len(matches)))
    if not bank.size or not any(not e['successful'] for e in accepted):
        print(json.dumps(dict(matched_paths=len(accepted), rejected_paths=len(rejected),
            first_rejections=rejected[:3], retained_bank=bank.report())), flush=True)
        raise ValueError('Measured credit must include actual failed TRAIN paths')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = args.output_dir/source.name
    torch.save(state | dict(measured_train_credit=config,
        measured_train_credit_bank_report=bank.report()), checkpoint)
    torch.save(experience | dict(measured_train_credit=config, measured_train_credit_bank=bank.state()),
        args.output_dir/'staged_goal_experience.pt')
    audit = dict(created_at=datetime.now().astimezone().isoformat(), source_checkpoint=str(source),
        source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        closed_native_run=str(run), actor_updates=state['actor_updates'], critic_updates=state['critic_updates'],
        actual_one_step_replay_rows=len(rows['reward']), measured_train_credit=config,
        matched_paths=accepted, rejected_paths=rejected, retained_bank=bank.report(),
        original_goal_contract_model_Q_optimizer_counters_and_schedule_preserved=True,
        original_one_step_replay_unchanged=True, actual_absolute_21D_goals_not_inverse_labels=True,
        actual_failed_and_successful_TRAIN_included=True, no_DEV_or_FINAL_imported=True,
        auxiliary_is_uncorrected_behavior_nstep_not_unbiased_current_policy_label=True,
        initialized_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    (args.output_dir/'manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('actor_updates', 'critic_updates',
        'actual_one_step_replay_rows', 'retained_bank')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
