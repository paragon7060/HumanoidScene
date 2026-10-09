"""Compare exact SAC action values on completed successful TRAIN paths.

Queries never execute alternate commands, import evaluation rows or update
models. A successful behavior return is not an unbiased current-policy value.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import io
import json
import os
from pathlib import Path

import numpy as np
import torch

from audit_closed_dev_critics import identity, require_closed_run
from compare_actor_train_memory import owned_stable_bytes
from export_eval_q_videos import restored_agent, sha256
from summarize_batched_staged_run import read_snapshot, supported_success
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_train_credit import completed_episode_returns
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS, TrainSuccessBank


def action_values(agent, actor, critic, goals):
    features = torch.cat((agent.critic_normalizer(critic),
                          agent.critic_action_features(actor, goals)), -1)
    return torch.minimum(agent.q1(features), agent.q2(features)).squeeze(-1)


def phase_summary(indices, mask, recorded, greedy, expected, returns,
                  command_difference, goal_difference, jaw_difference):
    if not bool(mask.any()):
        return dict(queried_rows=0)
    def median(value):return float(value[mask].median())
    return dict(queried_rows=int(mask.sum()),
        first_actual_episode_row=int(indices[mask].min()),
        last_actual_episode_row=int(indices[mask].max()),
        recorded_action_Q_median=median(recorded), greedy_action_Q_median=median(greedy),
        current_policy_expected_Q_median=median(expected),
        recorded_behavior_return_median=median(returns),
        greedy_minus_recorded_action_Q_median=median(greedy-recorded),
        expected_policy_minus_recorded_action_Q_median=median(expected-recorded),
        greedy_Q_exceeds_recorded_fraction=float((greedy[mask]>recorded[mask]+1e-6).float().mean()),
        recorded_return_minus_recorded_action_Q_median=median(returns-recorded),
        decoded_command_max_abs_difference_median=median(command_difference),
        absolute_goal_max_abs_difference_median=median(goal_difference),
        different_greedy_jaw_choices_fraction=float(jaw_difference[mask].float().mean()))


def audit(run, checkpoint, matching_path, whole_path, output, *, samples=16, seed=9049):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Read-only TRAIN queries require an empty CUDA mask')
    if output.exists() or not 1 <= samples <= 64:
        raise ValueError('Use a new output and1..64 policy samples')
    torch.set_num_threads(1)
    require_closed_run(run)
    matching, whole = read_snapshot(matching_path), read_snapshot(whole_path)
    checkpoint_identity = identity(checkpoint)
    checksum = sha256(checkpoint)
    if (Path(matching['source_run']).resolve()!=run
            or Path(matching['protected_checkpoint']).resolve()!=checkpoint
            or checksum!=matching['checkpoint_SHA256']
            or checksum!=whole['matching_checkpoint_SHA256']
            or whole['run_directory_name']!=run.name
            or not whole.get('whole_original_DEV128_completed')
            or not whole.get('all6_scope')
            or not whole.get('actual_end_saved_model_and_normalizer_equal_protected_pre_DEV_model')):
        raise ValueError('Exact original whole evaluation and protected model required')
    state = torch.load(io.BytesIO(owned_stable_bytes(checkpoint)), map_location='cpu', weights_only=True)
    def finite(value):
        if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        return True
    if not finite(state):
        raise ValueError('Protected checkpoint contains nonfinite model or optimizer tensors')
    if ((state['actor_updates'],state['critic_updates'])!=(whole['actor_updates'],whole['critic_updates'])
            or state['goal_contract']!=read_snapshot(run/'agent.yaml')):
        raise ValueError('Original model counters or physical goal contract differs')
    end_checkpoint = run/f"checkpoint_{state['critic_updates']:08d}.pt"
    if sha256(end_checkpoint)!=whole['actual_end_saved_checkpoint_SHA256']:
        raise ValueError('Actual saved end model changed')
    bank = TrainSuccessBank(state['actor_obs_dim'],state['critic_obs_dim'],
                            state['successful_train_transitions']['config'])
    bank.restore(state['successful_train_transitions'])
    outcomes = {(r['wave'],r['environment']):r for r in read_snapshot(run/'metrics.json')['outcomes']}
    agent, _ = restored_agent(state)
    agent.eval().requires_grad_(False)
    model_before = structure_sha256(agent.state_dict())
    records=[]
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for region in REGIONS:
            for episode in bank.episodes[region]:
                outcome=episode['outcome'];key=(outcome['wave'],outcome['environment'])
                original=outcomes[key]
                if (not supported_success(original) or original['split']!='train'
                        or any(original[k]!=outcome[k] for k in ('layout','result','complete','initial_layout_valid'))
                        or episode['identity'].split('/wave')[0] not in (str(run),run.name)):
                    raise ValueError('Successful bank must match the actual closed TRAIN terminal')
                rows=episode['rows'];n=len(rows['reward'])
                full=torch.linspace(0,n-1,min(32,n)).round().long()
                tail=torch.linspace(max(0,n-64),n-1,min(16,n)).round().long()
                indices=torch.cat((full,tail)).unique(sorted=True)
                actor,critic,goals=[rows[k][indices] for k in ('actor_obs','critic_obs','action')]
                tokens=actor[:,94:98]
                if not ((tokens.argmax(-1)==REGIONS.index(region))&(tokens[:,REGIONS.index(region)]>.5)).all():
                    raise ValueError('Actual one-hot region differs from the successful terminal')
                returns=completed_episode_returns(rows,gamma=state['config']['gamma'])['reward'][indices]
                recorded=action_values(agent,actor,critic,goals)
                greedy_goals=agent.act(actor,deterministic=True)
                greedy=action_values(agent,actor,critic,greedy_goals)
                normalized=agent.actor_normalizer(agent.actor_features(actor))
                expectations=[]
                for _ in range(samples):
                    body,_,logits=agent.continuous_sample(normalized,raw=actor)
                    actions,weights,_,_=agent.enumerate_jaws(actor,body,logits)
                    expectations.append((agent.branch_values(agent.critic_normalizer(critic),actions,
                                                            raw=actor)*weights).sum(-1))
                expected=torch.stack(expectations).mean(0)
                actual_command=agent.critic_action_features(actor,goals)
                current_command=agent.critic_action_features(actor,greedy_goals)
                command_difference=(current_command-actual_command).abs().amax(-1)
                goal_difference=(greedy_goals-goals).abs().amax(-1)
                jaw_difference=(greedy_goals[:,19:21]!=goals[:,19:21]).any(-1)
                if not torch.isfinite(torch.stack((recorded,greedy,expected,returns))).all():
                    raise ValueError('Nonfinite actual TRAIN query; preserve source and reject')
                all_rows=torch.ones(len(indices),dtype=torch.bool)
                masks={'all_queried':all_rows,'last64_actual_rows':indices>=max(0,n-64),
                       'last16_actual_rows':indices>=max(0,n-16),'first_actual_held_row':indices==0}
                phases={name:phase_summary(indices,mask,recorded,greedy,expected,returns,
                    command_difference,goal_difference,jaw_difference) for name,mask in masks.items()}
                records.append(dict(wave=outcome['wave'],environment=outcome['environment'],
                    seed=outcome['layout']['seed'],region=region,box_type=outcome['layout']['target_box_type'],
                    actual_successful_held_rows=n,phases=phases))
    if identity(checkpoint)!=checkpoint_identity or structure_sha256(agent.state_dict())!=model_before:
        raise ValueError('Protected file or restored parameters/normalizer changed')
    grouped=defaultdict(list)
    for record in records:grouped[record['region']+'/'+record['box_type']].append(record)
    summaries={}
    for name,cases in grouped.items():
        phases={}
        for phase in ('all_queried','last64_actual_rows','last16_actual_rows','first_actual_held_row'):
            values=[r['phases'][phase] for r in cases if r['phases'][phase]['queried_rows']]
            phases[phase]=dict(episodes=len(values),**{
                key+'_episode_median':float(np.median([v[key] for v in values]))
                for key in values[0] if key not in ('queried_rows','first_actual_episode_row','last_actual_episode_row')})
        summaries[name]=dict(actual_successful_TRAIN_episodes=len(cases),phases=phases)
    result=dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_actual_successful_TRAIN_action_ranking_NOT_new_rollout',run_directory_name=run.name,
        actual_model_SHA256=checksum,actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],
        original_observation_region_names_by_column=list(REGIONS),actual_success_bank=bank.report(),
        policy_body_samples=samples,all_four_exact_jaw_branches=True,CPU_query_seed=seed,
        gamma=state['config']['gamma'],cases=records,by_region_size=summaries,
        actual_success_terminals_and_original_TRAIN_labels_verified=True,
        exact_actual_goal_servo_action_encoding_used=True,
        protected_model_finite_and_strict_artifact_restoration=True,
        behavior_return_NOT_unbiased_current_policy_value=True,
        Q_ranking_or_hypothetical_commands_NOT_physical_success_or_counterfactual_evidence=True,
        no_actor_critic_optimizer_normalizer_or_original_file_changes=True,
        no_live_payload_GPU_physics_replay_import_or_DEV_FINAL_queries=True,
        raw_observations_joint_actions_NOT_exported=True,independent_FINAL_unused=True,goal_not_complete=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    pending=output.with_suffix('.pending.json');pending.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');pending.replace(output)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run-dir','checkpoint','matching-model-proof','whole-eval-proof','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--policy-samples',type=int,default=16)
    parser.add_argument('--seed',type=int,default=9049)
    a=parser.parse_args()
    result=audit(a.run_dir.resolve(),a.checkpoint.resolve(),a.matching_model_proof,
                 a.whole_eval_proof,a.output,samples=a.policy_samples,seed=a.seed)
    print(json.dumps({k:result[k] for k in ('actor_updates','critic_updates','actual_success_bank','by_region_size')}))


if __name__=='__main__':main()
