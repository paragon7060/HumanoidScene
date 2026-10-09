#!/usr/bin/env python3
"""Compare actual Adam proposals with the TRAIN-success safeguard on closed data."""
import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import torch

from audit_closed_dev_critics import require_closed_run
from audit_closed_train_actor_gradients import finite, objectives
from export_eval_q_videos import restored_agent
from summarize_batched_staged_run import read_snapshot, supported_success
from kuavo_isaaclab_scene.rl.algorithms.common import optimize
from kuavo_isaaclab_scene.rl.algorithms.success_update_guard import (
    VARIANT, success_update_guard_config, success_metrics, acceptable, guarded_actor_step)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''): digest.update(chunk)
    return digest.hexdigest()


def audit(run, checkpoint, whole_path, output, batches):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists() or not 1 <= batches <= 8:
        raise ValueError('Empty GPU mask, unique output and 1..8 batches required')
    require_closed_run(run)
    whole = read_snapshot(whole_path)
    checksum = sha(checkpoint)
    if whole['run_directory_name'] != run.name or not whole['whole_original_DEV128_completed'] \
            or not whole['all6_scope'] or not whole['actual_end_saved_model_and_normalizer_equal_protected_pre_DEV_model'] \
            or whole['matching_checkpoint_SHA256'] != checksum:
        raise ValueError('Exact matching whole evaluation proof required')
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not finite(state) or state['goal_contract'] != read_snapshot(run/'agent.yaml'):
        raise ValueError('Finite original controller state required')
    experience_path = run/'staged_goal_experience.pt'
    experience_checksum = sha(experience_path)
    experience = torch.load(experience_path, map_location='cpu', weights_only=True, mmap=True)
    if experience['goal_contract'] != state['goal_contract']:
        raise ValueError('TRAIN replay and model contract differ')
    replay = experience['executed_goal_transitions']
    bank = TrainSuccessBank(518, 578, state['successful_train_transitions']['config'])
    bank.restore(state['successful_train_transitions'])
    original = {(o['wave'], o['environment']): o for o in read_snapshot(run/'metrics.json')['outcomes']}
    for episodes in bank.episodes.values():
        for e in episodes:
            o = e['outcome']; actual = original[(o['wave'], o['environment'])]
            if actual['split'] != 'train' or not supported_success(actual) \
                    or any(actual[k] != o[k] for k in ('layout', 'result', 'complete', 'initial_layout_valid')):
                raise ValueError('Only original safe completed TRAIN outcomes may protect updates')
    goal = state['goal_contract']; radius = goal['fixed_prior_radius']
    progress = min(1., max(0., state['actor_updates']-state['success_schedule_actor_origin'])/bank.config['fade_actor_updates'])
    fraction = bank.config['initial_replay_fraction'] + progress*(bank.config['final_replay_fraction']-bank.config['initial_replay_fraction'])
    config = success_update_guard_config(VARIANT)
    results = []
    torch.set_num_threads(1)
    for index in range(batches):
        torch.manual_seed(9250+index)
        selected = torch.randint(len(replay['reward']), (256,))
        batch, _ = bank.mix({k:v[selected] for k,v in replay.items()}, fraction, 'cpu')
        successes = bank.sample_actor(64, 'cpu')
        agents = []
        for _ in range(2):
            a, _ = restored_agent(state)
            a.restore(state, training=True)
            a.actor.requires_grad_(True)
            for field, stored in (('jaw_saturation_config','jaw_saturation'),
                    ('body_saturation_config','body_saturation'),('success_jaw_balance_config','success_jaw_balance')):
                setattr(a, field, deepcopy(state.get(stored)))
            agents.append(a)
        before = success_metrics(agents[0], successes)
        sampling_state = torch.get_rng_state()
        for a in agents:
            torch.set_rng_state(sampling_state)
            losses, _ = objectives(a, batch, successes,
                bank.config['actor_goal_mse_weight']/radius**2, bank.config['actor_jaw_nll_weight'])
            loss = sum(losses.values())
            if a is agents[0]:
                optimize(a.actor_optimizer, loss, a.actor.parameters())
            else:
                a.actor_success_guard_config = config
                accepted, report = guarded_actor_step(a, loss, successes)
        ordinary = success_metrics(agents[0], successes)
        guarded = success_metrics(agents[1], successes)
        if not acceptable(before, guarded, config):
            raise AssertionError('Accepted or restored guard state lost a protected TRAIN metric')
        results.append(dict(seed=9250+index, before=before, ordinary_after=ordinary,
            ordinary_passes_same_protection=acceptable(before, ordinary, config),
            guarded_after=guarded, accepted=accepted, scale=report['actor_success_guard_parameter_scale'],
            projected_constraints=report['actor_success_guard_projected_constraints']))
    if sha(checkpoint) != checksum or sha(experience_path) != experience_checksum:
        raise AssertionError('Original closed input changed')
    result = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='actual_closed_TRAIN_disposable_Adam_guard_comparison_NOT_rollout',
        run_directory_name=run.name, checkpoint_SHA256=checksum, experience_SHA256=experience_checksum,
        actual_completed_TRAIN_success_paths=sum(len(x) for x in bank.episodes.values()),
        guard=config, batches=results,
        ordinary_regressing_batches=sum(not r['ordinary_passes_same_protection'] for r in results),
        guarded_regressing_batches=0, guarded_accepted_batches=sum(r['accepted'] for r in results),
        original_model_optimizer_replay_unchanged=True, DEV_FINAL_training_import=False,
        no_actual_physics_or_success_rate_improvement_claim=True, goal_not_complete=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'batches'}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ('run', 'checkpoint', 'whole-dev', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--batches', type=int, default=4)
    args = parser.parse_args()
    audit(args.run.resolve(), args.checkpoint.resolve(), args.whole_dev, args.output, args.batches)
