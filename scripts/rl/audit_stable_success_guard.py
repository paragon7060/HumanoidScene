"""Disposable CPU Adam checks on closed TRAIN data, never an evaluation input.

The new numeric contract is recalibrated on exactly the saved TRAIN cohort.
This is diagnostic migration only; production starts with an empty cohort.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from audit_closed_dev_critics import require_closed_run, identity
from export_eval_q_videos import restored_agent, sha256
from summarize_batched_staged_run import supported_success
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms.success_update_guard import success_update_guard_config
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank


def audit(run, checkpoint, output, batches=2):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists() or not 1 <= batches <= 4:
        raise ValueError('Empty CUDA mask, unique output and 1..4 batches required')
    torch.set_num_threads(1)
    require_closed_run(run)
    checksum = sha256(checkpoint)
    original = identity(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    experience_path = run/'staged_goal_experience.pt'
    replay_identity = identity(experience_path)
    experience = torch.load(experience_path, map_location='cpu', weights_only=True, mmap=True)
    if experience['goal_contract'] != state['goal_contract']:
        raise ValueError('Matching actual TRAIN replay required')
    replay = experience['executed_goal_transitions']
    bank = TrainSuccessBank(518, 578, state['successful_train_transitions']['config'])
    bank.restore(state['successful_train_transitions'])
    outcomes = json.loads((run/'metrics.json').read_text())['outcomes']
    actual = {(o['wave'],o['environment']):o for o in outcomes}
    entries = state['success_guard_memory']['entries']
    for e in entries:
        o=e['outcome']; measured=actual[o['wave'],o['environment']]
        if measured['split'] != 'train' or not supported_success(measured) or any(
                measured[k] != o[k] for k in ('layout','result','complete','initial_layout_valid')):
            raise ValueError('Persistent cohort requires original safe completed TRAIN outcomes')
    results=[]
    for index in range(batches):
        torch.manual_seed(10450+index)
        rows=torch.randint(len(replay['reward']),(256,))
        batch, mixed=bank.mix({k:v[rows].clone() for k,v in replay.items()},.2,'cpu')
        success=bank.sample_actor(64,'cpu')
        rng=torch.get_rng_state()
        for variant in (cohort.VARIANT,cohort.STABLE_VARIANT):
            agent,_=restored_agent(state)
            agent.restore(state,training=True)
            agent.actor.requires_grad_(True)
            agent.q1.requires_grad_(True);agent.q2.requires_grad_(True)
            agent.log_alpha.requires_grad_(True);agent.log_alpha_discrete.requires_grad_(True)
            for name in ('jaw_saturation','body_saturation','success_jaw_balance'):
                setattr(agent,name+'_config',deepcopy(state.get(name)))
            agent.actor_success_guard_config=success_update_guard_config(variant)
            if variant==cohort.STABLE_VARIANT:
                # Same original safe TRAIN paths; no DEV/FINAL labels used.
                paths={}
                for e in entries:
                    paths.setdefault(e['outcome']['layout']['target_region'],[]).append(dict(
                        identity=e['identity'],outcome=e['outcome'],rows={
                            'actor_obs':e['actor_obs'],'action':e['action']}))
                agent.success_guard_memory=cohort.empty_memory()
                cohort.synchronize(agent,SimpleNamespace(episodes=paths))
            guard_batch,groups=cohort.cohort_batch(agent)
            values,correct=cohort.metrics(agent,guard_batch,groups)
            baseline_allowed=cohort.within_ceilings(values,correct,agent.success_guard_memory,
                                                  agent.actor_success_guard_config)
            parameters=[p.detach().clone() for p in agent.actor.parameters()]
            with torch.no_grad():
                obs=batch['actor_obs']; norm=agent.actor_normalizer(agent.actor_features(obs))
                old_mean,old_std,old_logits=agent.continuous_parameters(norm,obs)
                old_goals=agent.act(obs,True).clone()
            old_entropy=[float(p.detach()) for p in (agent.log_alpha,agent.log_alpha_discrete)]
            torch.set_rng_state(rng)
            report=agent.update(batch,successful_train=success,
                success_goal_weight=bank.config['actor_goal_mse_weight']/state['goal_contract']['fixed_prior_radius']**2,
                success_jaw_weight=bank.config['actor_jaw_nll_weight'])
            with torch.no_grad():
                mean,std,logits=agent.continuous_parameters(norm,obs)
                delta=torch.cat([(p.detach()-old).flatten() for p,old in zip(agent.actor.parameters(),parameters)])
                new_goals=agent.act(obs,True)
            results.append(dict(batch=index,variant=variant,baseline_allowed=baseline_allowed,
                success_Q_rows=mixed,actual_parameter_delta_norm=float(delta.norm()),
                actual_parameter_delta_max=float(delta.abs().max()),
                deterministic_goal_change_max=float((new_goals-old_goals).abs().max()),
                body_mean_change_max=float((mean-old_mean).abs().max()),
                std_change_max=float((std-old_std).abs().max()),jaw_logit_change_max=float((logits-old_logits).abs().max()),
                entropy_log_parameters_before=old_entropy,
                entropy_log_parameters_after=[float(p.detach()) for p in (agent.log_alpha,agent.log_alpha_discrete)],
                **{k:v for k,v in report.items() if k.startswith('actor_success_guard') or k in
                   ('actor_updated','alpha','alpha_discrete','q_loss','success_goal_loss','success_jaw_loss')}))
    if identity(checkpoint)!=original or sha256(checkpoint)!=checksum or identity(experience_path)!=replay_identity:
        raise AssertionError('Original input changed')
    result=dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_TRAIN_disposable_full_SAC_step_NUMERIC_diagnostic_NOT_rollout',
        run_directory_name=run.name,checkpoint_SHA256=checksum,
        replay_rows=len(replay['reward']),persistent_TRAIN_paths=len(entries),
        no_DEV_FINAL_rows_used=True,no_original_file_updates=True,
        critic_updated_in_disposable_copy_without_episode_return_auxiliary=True,
        no_rollout_success_improvement_claim=True,results=results)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(batch=r['batch'],variant=r['variant'],baseline_allowed=r['baseline_allowed'],
        updated=r['actor_updated'],scale=r['actor_success_guard_parameter_scale'],
        parameter_delta=r['actual_parameter_delta_norm'],body_change=r['body_mean_change_max'],
        jaw_change=r['jaw_logit_change_max'],region_scales=r.get('actor_success_guard_region_parameter_scales')) for r in results]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run','checkpoint','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batches',type=int,default=2)
    a=p.parse_args();audit(a.run.resolve(),a.checkpoint.resolve(),a.output,a.batches)
