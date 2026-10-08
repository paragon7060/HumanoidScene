#!/usr/bin/env python3
"""Inspect goal-servo gradients on successful and failed, closed TRAIN paths.

This queries one protected final policy on past measured states. It does not
reconstruct the historical learner, import replay, or claim physical success.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from audit_servo_actor_gradients import load_owned
from export_eval_q_videos import restored_agent, sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_actions import body_command
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import staged_context


def samples(source, state, prior, *, group_by_box_type=False):
    """Use every valid held TRAIN episode; retain early, middle and late states."""
    outcomes = json.loads(load_owned(source / 'metrics.json'))['outcomes']
    train = {(v['wave'], v['environment']): v for v in outcomes if v['split'] == 'train'}
    groups = defaultdict(list)
    coverage = Counter(requested_train_episodes=len(train))
    with h5py.File(source / 'executed_transitions.hdf5', 'r') as h:
        episodes = {(int(e.attrs['wave']), int(e.attrs['environment'])): e
                    for e in h['episodes'].values()}
        for key, outcome in train.items():
            if not outcome['initial_layout_valid']:
                coverage['initial_invalid'] += 1
                continue
            result = outcome['result']
            base = result.get('staged_base', {})
            start = base.get('manipulation_start')
            if base.get('phase') != 'held_grasp' or not isinstance(start, int):
                coverage['never_entered_held_phase'] += 1
                continue
            if not outcome['complete'] or key not in episodes:
                raise ValueError('Completed valid TRAIN episode is missing')
            episode = episodes[key]
            layout = json.loads(episode.attrs['layout_json'])
            rows = episode['transitions']
            n = len(rows['actor_obs'])
            if (layout != outcome['layout'] or layout['split'] != 'train'
                    or n != outcome['executed_transition_rows'] or n != result['steps']
                    or n <= start):
                raise ValueError('Closed TRAIN identity or transition count differs')
            region = layout['target_region']
            ending = ('success' if result.get('success') else 'unsafe' if result.get('unsafe')
                      else 'timeout' if result.get('time_out') else 'other')
            coverage['included_train_episodes'] += 1
            coverage['episodes_' + ending] += 1
            stage = SimpleNamespace(phase='held_grasp', manipulation_start=start,
                target_xy=torch.tensor([base['base_target_xy_rack_m']]),
                target_yaw=base['base_target_yaw_rack_rad'])
            initial = torch.from_numpy(episode['initial_state/observations/policy'][:])[None]
            anchor = prior.coordinates.box_anchor(initial)
            spans = {'early': np.arange(start, min(n, start + 8)),
                     'middle': np.unique(np.linspace(start, n - 1, min(n - start, 8)).astype(int)),
                     'late': np.arange(max(start, n - 8), n)}
            for window, indices in spans.items():
                raw = torch.from_numpy(rows['actor_obs'][indices])
                extra = torch.from_numpy(rows['actor_supplemental'][indices])
                clock = torch.from_numpy(indices - start)
                config = prior.state
                features = prior.coordinates.observations(raw, clock,
                    config.get('time_harmonics', 0), config.get('clock_horizon', 410),
                    condition_on_shelf=config.get('shelf_conditioned_clock_fit', False),
                    clock_limit=config.get('actor_clock_limit'))
                context = staged_context(raw, stage, state['goal_contract']['fixed_prior_radius'])
                ao = torch.cat((features, anchor.expand(len(raw), -1), extra, context), -1)
                physical = body_command(torch.from_numpy(rows['action'][indices]))
                if ao.shape[1] != 518 or not torch.isfinite(ao).all():
                    raise ValueError('Measured actor feature reconstruction differs')
                identity = ((region, layout['target_box_type'], ending, window)
                            if group_by_box_type else (region, ending, window))
                groups[identity].append((ao, physical))
    return groups, dict(coverage)


def diagnose(raw, recorded, agent, generator):
    normal = agent.actor_normalizer(agent.actor_features(raw))
    with torch.no_grad():
        mean, log_std, _ = agent.continuous_parameters(normal, raw)
        _, radius = agent.anchor_and_scale(raw)
        greedy = agent.act(raw, True)
        delta = agent.goal_servo_critic_encoder.unclipped_body(raw, greedy)
        command = agent.critic_action_features(raw, greedy)[:, :19]
    latent = mean.detach().requires_grad_(True)
    body = agent.body_from_latent(normal, latent.tanh(), raw)
    goals = agent.action_projector(raw, torch.cat((body, greedy[:, 19:]), -1))
    encoded = agent.critic_action_features(raw, goals)[:, :19]
    if not torch.allclose(goals, greedy, atol=1e-7, rtol=1e-6):
        raise ValueError('Diagnostic mean differs from the actual deterministic sampler')
    jacobian = torch.autograd.grad(encoded.sum(), latent)[0]
    encoder = agent.goal_servo_critic_encoder
    steps = raw.new_tensor(encoder.coordinates.joints.scales + [.1 / 30, .1 / 30])
    expected = encoder.scale[:19].to(raw) / steps * radius * (1 - mean.tanh().square())
    expected *= delta.abs() <= 1
    if not torch.allclose(jacobian, expected, atol=2e-5, rtol=2e-5):
        raise ValueError('Actual autograd differs from the independent servo Jacobian')
    masks = {'body': radius > 1e-8, 'arms': radius > 1e-8}
    masks['arms'] = masks['arms'].clone()
    masks['arms'][:, [0, 15, 16, 17, 18]] = False
    noises = {k: [] for k in ('Gaussian', 'full_ramp_arm_bias')}
    with torch.no_grad():
        for _ in range(8):
            draw = torch.randn(mean.shape, generator=generator)
            bias = (draw * .8).clamp(-1.6, 1.6)
            bias[:, [0, 15, 16, 17, 18]] = 0
            for name in ('Gaussian', 'full_ramp_arm_bias'):
                noisy_body, _, _ = agent.continuous_sample(normal, raw=raw,
                    noise=draw if name == 'Gaussian' else None,
                    deterministic=name == 'full_ramp_arm_bias',
                    body_latent_offset=bias if name == 'full_ramp_arm_bias' else None)
                noisy_goals = agent.action_projector(raw, torch.cat((noisy_body, greedy[:, 19:]), -1))
                noises[name].append((agent.critic_action_features(raw, noisy_goals)[:, :19] - command).abs())
    result = {'states': len(raw), 'all_jacobians_finite': bool(torch.isfinite(jacobian).all()),
        'gradient_coordinate': 'actual_state_aware_pre_tanh_Gaussian_mean',
        'network_residual_mean_local_scaling_NOT_included_in_this_Jacobian': True,
        'deterministic_body_mean_matches_actual_sampler': True,
        'noise_and_episode_bias_use_actual_continuous_sampler': True,
        'greedy_command_MAE_against_past_executed_body': float((command - recorded[:, :19]).abs().mean()),
        'greedy_jaw_pair_match_past_executed_fraction': float((greedy[:, 19:] == recorded[:, 19:]).all(-1).float().mean()),
        'Gaussian_std_min': float(log_std.exp().min()), 'Gaussian_std_max': float(log_std.exp().max())}
    for name, active in masks.items():
        denom = int(active.sum())
        if not denom:
            result[name] = {'active_coordinate_samples': 0}
            continue
        blocked = (jacobian.abs() < 1e-8) & active
        record = {'active_coordinate_samples': denom,
            'servo_clipped_active_fraction': float(((delta.abs() > 1) & active).sum()) / denom,
            'zero_servo_gradient_active_fraction': float(blocked.sum()) / denom,
            'all_available_gradients_zero_states': int(((active.sum(-1) > 0) & (blocked.sum(-1) == active.sum(-1))).sum()),
            'active_gradient_abs_median': float(jacobian.abs()[active].median()),
            'tanh_mean_abs_over3_active_fraction': float(((mean.abs() > 3) & active).sum()) / denom}
        for noise_name, values in noises.items():
            v = torch.stack(values)[:, active]
            record[noise_name] = {'command_delta_median': float(v.median()),
                'command_delta_p95': float(torch.quantile(v.flatten(), .95)),
                'changes_over0p05_fraction': float((v > .05).float().mean())}
        result[name] = record
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-pointer', type=Path, required=True)
    p.add_argument('--completed-evaluation', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--by-box-type', action='store_true',
        help='Keep small and medium separately within each rack region')
    a = p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('This read-only diagnostic requires CUDA_VISIBLE_DEVICES empty')
    torch.set_num_threads(1)
    pointer_blob = load_owned(a.checkpoint_pointer)
    evaluation_blob = load_owned(a.completed_evaluation)
    pointer = json.loads(pointer_blob)
    evaluation = json.loads(evaluation_blob)
    source = Path(pointer['source_run'])
    supervisor = json.loads(load_owned(source.parent / 'status.json'))
    status_blob = load_owned(source / 'status.json')
    if (supervisor.get('training_exit_code') != 0
            or Path('/proc', str(supervisor['training_pid'])).exists()
            or json.loads(status_blob).get('status') != 'complete'):
        raise ValueError('The original TRAIN writer must have normally exited')
    checkpoint = Path(pointer['protected_checkpoint'])
    blob = load_owned(checkpoint)
    digest = sha256(checkpoint)
    if (digest != pointer['checkpoint_SHA256'] or digest != evaluation['matching_checkpoint_SHA256']
            or not evaluation['whole_original_DEV128_completed']):
        raise ValueError('Protected final model requires matching whole DEV evidence')
    state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if (state['actor_updates'], state['critic_updates']) != (pointer['actor_updates'], pointer['critic_updates']):
        raise ValueError('Protected actor/critic counters differ')
    hdf = source / 'executed_transitions.hdf5'
    if hdf.is_symlink() or hdf.stat().st_uid != os.getuid():
        raise ValueError('An owned regular closed HDF is required')
    hdf_before = sha256(hdf)
    agent, prior = restored_agent(state)
    groups, coverage = samples(source, state, prior, group_by_box_type=a.by_box_type)
    generator = torch.Generator().manual_seed(237081400)
    results = []
    for group, rows in sorted(groups.items()):
        region, *middle, ending, window = group
        raw = torch.cat([v[0] for v in rows])
        labels = torch.cat([v[1] for v in rows])
        result = dict(region=region, ending=ending, window=window, episodes=len(rows),
            **diagnose(raw, labels, agent, generator))
        if middle:
            result['box_type'] = middle[0]
        results.append(result)
        print(json.dumps({k: result[k] for k in ('region', 'ending', 'window', 'states', 'arms')}), flush=True)
    if not results or not all(torch.equal(v, agent.state_dict()[k]) for k, v in state['model'].items()):
        raise ValueError('No TRAIN states or model/normalizer changed')
    if (hdf_before != sha256(hdf) or blob != load_owned(checkpoint)
            or pointer_blob != load_owned(a.checkpoint_pointer)
            or evaluation_blob != load_owned(a.completed_evaluation)
            or status_blob != load_owned(source / 'status.json')):
        raise ValueError('Closed input or protected model changed during the diagnostic')
    proof = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        role='protected_final_policy_on_closed_successful_and_failed_TRAIN_states',
        matching_checkpoint_SHA256=digest, closed_HDF_SHA256=hdf_before,
        source_run_name=source.name, actor_updates=state['actor_updates'], critic_updates=state['critic_updates'],
        coverage=coverage, groups=results,
        actual_autograd_matches_independent_servo_Jacobian=True,
        historical_collection_actions_compared_but_historical_policy_NOT_reconstructed=True,
        Gaussian_and_full_ramp_arm_bias_are_one_state_probes_NOT_full_AR1_collection=True,
        no_current_optimizer_step_no_replay_import_no_GPU_or_Isaac=True,
        no_live_HDF_or_replay_read=True, no_DEV_or_FINAL_states_used=True,
        protected_model_normalizer_and_closed_inputs_unchanged=True,
        physical_success_or_causal_learning_improvement_NOT_claimed=True, goal_not_complete=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(a.output), 'coverage': coverage, 'groups': len(results)}), flush=True)


if __name__ == '__main__':
    main()
