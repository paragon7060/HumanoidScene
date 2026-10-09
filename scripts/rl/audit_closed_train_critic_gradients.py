"""Compare TD and recorded-return critic updates on one closed TRAIN run.

All updates are disposable CPU copies. This is neither a physical evaluation
nor evidence of improved success. Recorded behavior returns are off-policy.
"""

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from audit_closed_dev_critics import identity, require_closed_run
from export_eval_q_videos import restored_agent, sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import (
    BALANCED_RETURN_VARIANT, EPISODE_RETURN_VARIANT, MeasuredTrainCreditBank, measured_credit_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_train_credit import completed_episode_returns
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank


def features(agent, batch):
    return torch.cat((agent.critic_normalizer(batch['critic_obs']),
        agent.critic_action_features(batch['actor_obs'], batch['action'])), -1)


def loss(agent, batch, *, episode_return=False):
    if episode_return:
        target = agent.config.reward_scale * batch['reward']
    else:
        target, _ = agent.critic_target(batch, agent.log_alpha.exp().detach(),
                                      agent.log_alpha_discrete.exp().detach())
    encoded = features(agent, batch)
    return (F.mse_loss(agent.q1(encoded).squeeze(-1), target)
            + F.mse_loss(agent.q2(encoded).squeeze(-1), target))


def gradient(agent, batch, parameters, *, episode_return=False):
    objective = loss(agent, batch, episode_return=episode_return)
    values = torch.autograd.grad(objective, parameters)
    return objective.item(), torch.cat([g.detach().flatten() for g in values])


def fixed_train_queries(success_bank, measured_bank, gamma):
    """Use evenly spaced rows of retained TRAIN episodes, never DEV trajectories."""
    queries = []
    seen = set()
    for bank, provenance in ((success_bank, 'successful_TRAIN'),
                             (measured_bank, 'measured_TRAIN')):
        for episodes in bank.episodes.values():
            for episode in episodes:
                if episode['identity'] in seen:
                    continue
                seen.add(episode['identity'])
                rows = completed_episode_returns(episode['rows'], gamma=gamma)
                ids = torch.linspace(0, len(rows['reward'])-1,
                                     min(32, len(rows['reward']))).round().long().unique()
                result = episode['outcome']['result']
                category = ('success' if result['success'] else 'unsafe' if result.get('unsafe')
                            else 'time_out' if result.get('time_out') else 'other_failure')
                queries.append(dict(category=category, source=provenance,
                    batch={k: v[ids] for k, v in rows.items()}))
    return queries


@torch.no_grad()
def calibration(agent, queries):
    categories = {}
    for query in queries:
        batch = query['batch']
        encoded = features(agent, batch)
        q = torch.minimum(agent.q1(encoded), agent.q2(encoded)).squeeze(-1)
        target = agent.config.reward_scale * batch['reward']
        values = dict(Q_mean=q.mean().item(), return_mean=target.mean().item(),
                      Q_minus_recorded_return=(q-target).mean().item(),
                      absolute_difference=(q-target).abs().mean().item())
        categories.setdefault(query['category'], []).append(values)
    return {key: dict(episodes=len(values), **{
        name+'_episode_median': float(np.median([v[name] for v in values]))
        for name in values[0]}) for key, values in categories.items()}


def audit(run, checkpoint, matching, output, steps, seed):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CUDA_VISIBLE_DEVICES must be empty')
    torch.set_num_threads(1)
    require_closed_run(run)
    if (Path(matching['source_run']).resolve() != run
            or Path(matching['protected_checkpoint']).resolve() != checkpoint
            or sha256(checkpoint) != matching['checkpoint_SHA256']):
        raise ValueError('Closed source and exact protected model proof differ')
    payload = run/'staged_goal_experience.pt'
    before = identity(payload)
    payload_checksum = sha256(payload)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    experience = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
    config = measured_credit_config(EPISODE_RETURN_VARIANT)
    if (experience['goal_contract'] != state['goal_contract']
            or experience['measured_train_credit'] != config
            or state['measured_train_credit'] != config
            or [state['actor_updates'], state['critic_updates']] !=
               [matching['actor_updates'], matching['critic_updates']]):
        raise ValueError('Matching controller, critic counters and return bank are required')
    replay = experience['executed_goal_transitions']
    n = len(replay['reward'])
    if not n or any(not bool((replay[k][:, -6] == 1).all()) for k in
                   ('actor_obs', 'critic_obs', 'next_actor_obs', 'next_critic_obs')):
        raise ValueError('Replay must contain confirmed held-phase TRAIN only')
    success = TrainSuccessBank(state['actor_obs_dim'], state['critic_obs_dim'],
                              state['successful_train_transitions']['config'])
    success.restore(state['successful_train_transitions'])
    measured = MeasuredTrainCreditBank(state['actor_obs_dim'], state['critic_obs_dim'],
                                      state['config']['gamma'], config)
    measured.restore(experience['measured_train_credit_bank'])
    if measured.report() != state['measured_train_credit_bank_report']:
        raise ValueError('Saved final TRAIN bank and model report differ')
    origin = state['success_schedule_actor_origin']
    progress = max(0., min(1., (state['actor_updates']-origin)/success.config['fade_actor_updates']))
    fraction = (success.config['initial_replay_fraction'] + progress *
                (success.config['final_replay_fraction']-success.config['initial_replay_fraction']))
    torch.manual_seed(seed)
    batches = []
    for _ in range(steps):
        ids = torch.randint(n, (256,))
        actual, retained = success.mix({k: v[ids] for k, v in replay.items()}, fraction, 'cpu')
        batches.append((actual, measured.sample(config['batch_size'], 'cpu'), retained))
    balanced = MeasuredTrainCreditBank(state['actor_obs_dim'], state['critic_obs_dim'],
        state['config']['gamma'], measured_credit_config(BALANCED_RETURN_VARIANT))
    for episodes in measured.episodes.values():
        for episode in episodes:
            balanced.add_episode(episode['rows'], episode['outcome'],
                                 source_run=episode['identity'].split('/wave')[0])
    torch.manual_seed(seed)
    balanced_batches = [(actual, balanced.sample(config['batch_size'], 'cpu'), retained)
                        for actual, _, retained in batches]
    queries = fixed_train_queries(success, measured, state['config']['gamma'])
    agent, _ = restored_agent(state)
    agent.q1.requires_grad_(True)
    agent.q2.requires_grad_(True)
    parameters = [*agent.q1.parameters(), *agent.q2.parameters()]
    agent.measured_train_credit_enabled = True
    agent.measured_train_credit_config = config
    baseline = calibration(agent, queries)
    directions = []
    for sampling, trial_batches in ((EPISODE_RETURN_VARIANT, batches),
                                   (BALANCED_RETURN_VARIANT, balanced_batches)):
        for i, (batch, auxiliary, retained) in enumerate(trial_batches[:8]):
            torch.manual_seed(seed+1000+i)
            td_loss, td = gradient(agent, batch, parameters)
            mc_loss, mc = gradient(agent, auxiliary, parameters, episode_return=True)
            td_norm, mc_norm = td.norm(), mc.norm()
            directions.append(dict(batch=i, return_sampling=sampling,
                one_step_loss=td_loss, return_loss=mc_loss,
                TD_gradient_norm=td_norm.item(), return_gradient_norm=mc_norm.item(),
                gradient_cosine=(torch.dot(td, mc)/(td_norm*mc_norm).clamp_min(1e-12)).item(),
                weighted_return_to_TD_gradient_norm_at_0p1=(.1*mc_norm/td_norm.clamp_min(1e-12)).item(),
                retained_success_rows=retained))
    comparisons = []
    for weight, sampling, trial_batches in (
            (.1, EPISODE_RETURN_VARIANT, batches),
            (.3, EPISODE_RETURN_VARIANT, batches),
            (1., EPISODE_RETURN_VARIANT, batches),
            (.1, BALANCED_RETURN_VARIANT, balanced_batches)):
        # Each trial starts with the same model and mature Adam moments. Never
        # share saved optimizer tensors between disposable trials.
        agent.restore(deepcopy(state), training=True)
        agent.q1.requires_grad_(True)
        agent.q2.requires_grad_(True)
        agent.measured_train_credit_config = measured_credit_config(sampling)
        reports = []
        for i, (batch, auxiliary, _) in enumerate(trial_batches):
            torch.manual_seed(seed+1000+i)
            report = agent.update(batch, update_actor=False,
                                 critic_auxiliary=auxiliary, critic_auxiliary_weight=weight)
            reports.append({k: report[k] for k in ('q_loss', 'one_step_q_loss',
                'measured_episode_return_q_loss', 'measured_episode_return_weight')})
        unchanged = all(torch.equal(v, agent.state_dict()[k]) for k, v in state['model'].items()
                        if not k.startswith(('q1.', 'q2.', 'target1.', 'target2.')))
        if not unchanged:
            raise ValueError('The actor, normalizers or other model state changed')
        comparisons.append(dict(return_weight=weight, return_sampling=sampling,
            disposable_critic_updates=steps,
            final_calibration=calibration(agent, queries), first_update=reports[0],
            last_update=reports[-1], actor_and_normalizers_unchanged=True))
    if identity(payload) != before or sha256(checkpoint) != matching['checkpoint_SHA256']:
        raise ValueError('Closed originals changed')
    proof = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_actual_TRAIN_disposable_CPU_critic_gradient_diagnostic_NOT_evaluation',
        source_run_directory_name=run.name, source_model_SHA256=matching['checkpoint_SHA256'],
        source_experience_SHA256=payload_checksum, actor_updates=state['actor_updates'],
        critic_updates=state['critic_updates'], seed=seed, replay_rows=n,
        success_bank=success.report(), measured_bank=measured.report(),
        successful_TRAIN_Q_replay_fraction=fraction, actual_batch_size=256,
        return_batch_size=config['batch_size'], return_weight_in_existing_training=.1,
        gradient_batches=directions, baseline_calibration=baseline, comparisons=comparisons,
        same_one_step_batches_target_sampling_seeds_and_mature_Adam=True,
        original_sampling_weight_trials_same_return_batches=True,
        balanced_trial_changes_only_return_sampling=True,
        normalizers_frozen_for_controlled_diagnostic=True, actor_updates_per_trial=0,
        target_network_Polyak_updates_retained=True, no_actual_training_files_changed=True,
        closed_originals_unchanged=True, no_GPU_or_physics=True, no_DEV_or_FINAL_rows=True,
        raw_observations_actions_NOT_exported=True,
        recorded_behavior_return_is_off_policy_NOT_unbiased_current_policy_target=True,
        no_new_physical_success_claim=True, goal_not_complete=True)
    output.mkdir(parents=True, exist_ok=False)
    (output/'verification.json').write_text(json.dumps(proof, indent=2)+'\n')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--matching-model-proof', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261009)
    args = parser.parse_args()
    if not 1 <= args.steps <= 64:
        parser.error('--steps must be between 1 and 64')
    proof = audit(args.run_dir.resolve(), args.checkpoint.resolve(),
                  json.loads(args.matching_model_proof.read_text()),
                  args.output_dir.resolve(), args.steps, args.seed)
    print(json.dumps(dict(baseline=proof['baseline_calibration'],
        gradient_batches=proof['gradient_batches'], comparisons=proof['comparisons'])))


if __name__ == '__main__':
    main()
