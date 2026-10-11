"""Two disposable SAC proposals on a saved completed TRAIN success bank.

The Q minibatch is success-only for this diagnostic, not production replay.
CPU is the default. Optional cuda:0 is isolated to physical GPU3 before
CUDA initialization. No rollout, evaluation labels, current replay/HDF,
optimizer import into a live learner, or physical improvement is claimed.
"""
import argparse
from copy import deepcopy
from datetime import datetime
import importlib.util
import json
import os
from pathlib import Path

import torch

from audit_closed_dev_critics import identity
from export_eval_q_videos import restored_agent, sha256
from summarize_batched_staged_run import read_snapshot, supported_success
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms import success_update_guard as update_guard
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank, REGIONS


def audit(checkpoint, metrics, output, device='cpu', baseline_source_root=None):
    if output.exists():
        raise ValueError('Unique output required')
    if device=='cpu':
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
            raise ValueError('CPU diagnostic requires empty CUDA mask')
        isolation=None
    else:
        from single_gpu_runtime import ensure_single_gpu_namespace
        isolation=ensure_single_gpu_namespace(3)
    torch.set_num_threads(1)
    original, checksum = identity(checkpoint), sha256(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['actor_success_guard']['name'] != cohort.TUBE_VARIANT:
        raise ValueError('Saved fixed tube checkpoint required')
    actual = {(x['wave'],x['environment']):x for x in read_snapshot(metrics)['outcomes']}
    bank = TrainSuccessBank(518,578,state['successful_train_transitions']['config'])
    bank.restore(state['successful_train_transitions'])
    for episodes in bank.episodes.values():
        for entry in episodes:
            o=entry['outcome'];recorded=actual[o['wave'],o['environment']]
            if recorded['split']!='train' or not supported_success(recorded) or any(
                    recorded[k]!=o[k] for k in ('layout','result','complete','initial_layout_valid')):
                raise ValueError('Exact completed actual safe TRAIN outcomes required')
    repair = cohort.violation_correction
    current_guard = update_guard.guarded_actor_step
    baseline_guard = baseline_identity = None
    if baseline_source_root is not None:
        file = baseline_source_root/'src/kuavo_isaaclab_scene/rl/algorithms/success_update_guard.py'
        specification = importlib.util.spec_from_file_location(
            'kuavo_isaaclab_scene.rl.algorithms._disposable_baseline_guard', file)
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        if module.success_update_guard_config(cohort.TUBE_VARIANT) != state['actor_success_guard']:
            raise ValueError('Baseline optimizer must preserve the same fixed guard contract')
        baseline_guard = module.guarded_actor_step
        baseline_identity = dict(file=str(file),SHA256=sha256(file))
    records=[]
    try:
        for index in range(2):
            torch.manual_seed(27012+index)
            batch=bank.sample(256,device);success=bank.sample_actor(64,device)
            rng=torch.get_rng_state()
            cuda_rng=torch.cuda.get_rng_state() if device!='cpu' else None
            for enabled in (False,True):
                agent,_=restored_agent(state,device)
                agent.restore(state,training=True)
                agent.actor.requires_grad_(True);agent.q1.requires_grad_(True);agent.q2.requires_grad_(True)
                agent.log_alpha.requires_grad_(True);agent.log_alpha_discrete.requires_grad_(True)
                for name in ('jaw_saturation','body_saturation','success_jaw_balance'):
                    setattr(agent,name+'_config',deepcopy(state.get(name)))
                raw=batch['actor_obs'];norm=agent.actor_normalizer(agent.actor_features(raw))
                parameters={k:p.detach().clone() for k,p in agent.actor.named_parameters()}
                with torch.no_grad():
                    mean0,_,jaw0=agent.continuous_parameters(norm,raw)
                    alpha0=[float(p.detach()) for p in (agent.log_alpha,agent.log_alpha_discrete)]
                if baseline_guard is None:
                    cohort.violation_correction=repair if enabled else lambda *args:(None,dict(body=0,jaw_loss=0,jaw_label=0))
                else:
                    cohort.violation_correction=repair
                    update_guard.guarded_actor_step=current_guard if enabled else baseline_guard
                torch.set_rng_state(rng)
                if cuda_rng is not None:torch.cuda.set_rng_state(cuda_rng)
                report=agent.update(batch,successful_train=success,
                    success_goal_weight=bank.config['actor_goal_mse_weight']/state['goal_contract']['fixed_prior_radius']**2,
                    success_jaw_weight=bank.config['actor_jaw_nll_weight'])
                with torch.no_grad():
                    mean,_,jaw=agent.continuous_parameters(norm,raw)
                deltas={k:p.detach()-parameters[k] for k,p in agent.actor.named_parameters()}
                movement={}
                for region_id,region in enumerate(REGIONS):
                    values=[v.flatten() for k,v in deltas.items() if k.startswith('network.heads.'+str(region_id)+'.')]
                    mask=raw[:,94:98].argmax(-1)==region_id
                    movement[region]=dict(parameter_delta_norm=float(torch.cat(values).norm()),
                        body_mean_change_max=float((mean-mean0)[mask].abs().max()) if mask.any() else None,
                        jaw_logit_change_max=float((jaw-jaw0)[mask].abs().max()) if mask.any() else None)
                guard_batch,groups=cohort.cohort_batch(agent)
                values,correct=cohort.metrics(agent,guard_batch,groups)
                within=cohort.within_ceilings(values,correct,agent.success_guard_memory,agent.actor_success_guard_config)
                if not within or agent.success_guard_memory['ceilings']!=state['success_guard_memory']['ceilings']:
                    raise AssertionError('Fixed whole cohort protection was changed or violated')
                if not all(torch.isfinite(v).all() for v in agent.state_dict().values()):
                    raise FloatingPointError('Nonfinite disposable model')
                records.append(dict(batch=index,candidate_repair_enabled=enabled if baseline_guard is None else True,
                    body_restoration_search_enabled=enabled if baseline_guard is not None else None,
                    Q_minibatch_actual_successful_TRAIN_rows=256,
                    actor_retention_actual_successful_TRAIN_rows=64,
                    movement_by_region=movement,
                    alpha_log_before=alpha0,
                    alpha_log_after=[float(p.detach()) for p in (agent.log_alpha,agent.log_alpha_discrete)],
                    fixed_whole_cohort_within_ceilings=True,
                    **{k:v for k,v in report.items() if k in ('actor_updated','q_loss','alpha','alpha_discrete')
                       or k.startswith('actor_success_guard')}))
    finally:
        cohort.violation_correction=repair
        update_guard.guarded_actor_step=current_guard
    if identity(checkpoint)!=original or sha256(checkpoint)!=checksum:
        raise AssertionError('Saved original model changed')
    result=dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='completed_TRAIN_success_only_disposable_SAC_feasibility_repair_NOT_rollout',
        device=device,single_GPU_isolation=isolation,
        CUDA_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        checkpoint_SHA256=checksum,source_actor_updates=state['actor_updates'],source_critic_updates=state['critic_updates'],
        baseline_optimizer=baseline_identity,
        actual_TRAIN_bank_rows=bank.size,records=records,
        no_production_replay_distribution_or_actual_GPU_proposal_reconstruction=True,
        teacher_and_episode_return_auxiliary_not_reconstructed=True,
        no_optimizer_mutation_or_import_into_live_training=True,
        current_raw_replay_HDF_and_DEV_FINAL_not_read=True,
        no_Isaac_or_new_physics=True,
        no_physical_success_or_future_success_guarantee=True,goal_not_complete=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{k:r[k] for k in ('batch','candidate_repair_enabled','actor_updated','movement_by_region')}
                      for r in records]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('checkpoint','metrics','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--baseline-source-root',type=Path,
        help='Trusted frozen source of the original optimizer; same fixed guard contract required')
    a=p.parse_args();audit(a.checkpoint.resolve(),a.metrics.resolve(),a.output,a.device,
        a.baseline_source_root.resolve() if a.baseline_source_root else None)
