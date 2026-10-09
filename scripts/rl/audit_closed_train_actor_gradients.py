"""Measure competing actor objectives on a closed, matching real TRAIN run.

No optimizer, normalizer, model or physical state is updated. Gradient norms
and directions are local evidence; they do not establish rollout improvement.
"""
import argparse
from copy import deepcopy
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
from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS, TrainSuccessBank


def finite(value):
    if isinstance(value, torch.Tensor):return bool(torch.isfinite(value).all())
    if isinstance(value, dict):return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):return all(finite(v) for v in value)
    return True


def objectives(agent, batch, successes, goal_weight, jaw_weight):
    raw = batch['actor_obs']
    normalized = agent.actor_normalizer(agent.actor_features(raw))
    body, body_logp, logits = agent.continuous_sample(normalized, raw=raw)
    actions, probabilities, jaw_logp, near = agent.enumerate_jaws(raw, body, logits)
    q = agent.branch_values(agent.critic_normalizer(batch['critic_obs']), actions, raw=raw)
    scale = q.detach().abs().mean().clamp_min(1).reciprocal() if agent.config.actor_q_normalize else 1.
    losses = dict(Q=-(probabilities*scale*q).sum(-1).mean(),
        continuous_entropy=agent.log_alpha.exp().detach()*body_logp.mean(),
        discrete_entropy=(probabilities*agent.log_alpha_discrete.exp().detach()*jaw_logp).sum(-1).mean())
    sr = successes['actor_obs']
    sn = agent.actor_normalizer(agent.actor_features(sr))
    mean, _, sl = agent.continuous_parameters(sn, sr)
    losses['successful_goal_retention'] = goal_weight*agent.success_body_loss(sr, mean.tanh(), successes['action'])
    jaw_loss, _ = agent.successful_jaw_loss(sr, sl, successes['action'])
    losses['successful_jaw_retention'] = jaw_weight*jaw_loss
    for name, term in (
            ('jaw_saturation', agent.actor_jaw_regularization(logits, near)),
            ('body_saturation', agent.actor_body_regularization(normalized, raw))):
        if term is not None:losses[name] = term[0]
    # Actor features omit masks/history; use the critic's original perceived
    # observation prefix for target selection, never privileged contact fields.
    tokens, valid = target_token(batch['critic_obs'][:,:464])
    if not bool(valid.all()):raise ValueError('Actual TRAIN target box must be perceived')
    if not torch.equal(tokens[:,8:12].argmax(-1),raw[:,94:98].argmax(-1)):
        raise ValueError('Perceived target region and actual actor routing differ')
    contribution = -(probabilities*scale*q).sum(-1)
    # Keep each class's actual fraction of the mixed batch. These objectives
    # sum to the original Q objective, without equalizing rare classes.
    class_losses = {}
    for region_id, region in enumerate(REGIONS):
        for size_id, size in enumerate(('small','medium')):
            mask = (tokens[:,8:12].argmax(-1)==region_id)&(tokens[:,3:5].argmax(-1)==size_id)
            if bool(mask.any()):
                class_losses[region+'/'+size] = (contribution[mask].sum()/len(raw),int(mask.sum()))
    return losses, class_losses


def gradient(loss, parameters):
    if not loss.requires_grad:
        return torch.zeros(sum(p.numel() for p in parameters))
    parts = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    return torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten()
                      for p, v in zip(parameters, parts)])


def directions(vectors):
    def cosine(a, b):
        divisor = a.norm()*b.norm()
        return float(torch.dot(a, b)/divisor) if float(divisor)>1e-15 else None
    total = sum(vectors.values())
    retained = vectors['successful_goal_retention']+vectors['successful_jaw_retention']
    return dict(component_norms={k:float(v.norm()) for k,v in vectors.items()},
        total_gradient_norm=float(total.norm()),
        Q_vs_success_retention_cosine=cosine(vectors['Q'], retained),
        total_vs_success_retention_cosine=cosine(total, retained),
        successful_retention_to_Q_norm=(float(retained.norm()/vectors['Q'].norm())
                                      if float(vectors['Q'].norm())>1e-15 else None),
        descent_alignment_with_each_objective={k:cosine(total,v) for k,v in vectors.items()})


def copied_adam_direction(parameters, vectors, saved_optimizer):
    """One prospective step in disposable parameters and mature Adam moments."""
    copies = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    before = [p.detach().clone() for p in copies]
    optimizer = torch.optim.Adam(copies)
    optimizer.load_state_dict(deepcopy(saved_optimizer))
    total = sum(vectors.values())
    offset = 0
    for p in copies:
        p.grad = total[offset:offset+p.numel()].reshape_as(p).clone()
        offset += p.numel()
    norm = torch.nn.utils.clip_grad_norm_(copies,1.,error_if_nonfinite=True)
    optimizer.step()
    delta = torch.cat([(p.detach()-old).flatten() for p,old in zip(copies,before)])
    if not torch.isfinite(delta).all():raise ValueError('Nonfinite copied Adam step')
    return delta, float(norm)


def audit(run, checkpoint, matching_path, whole_path, output, *, batches=8, seed=9150):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or output.exists() or not 1<=batches<=32:
        raise ValueError('Use an empty CUDA mask, unique output and1..32 batches')
    torch.set_num_threads(1)
    require_closed_run(run)
    matching, whole = read_snapshot(matching_path), read_snapshot(whole_path)
    checksum = sha256(checkpoint)
    if (Path(matching['source_run']).resolve()!=run
            or Path(matching['protected_checkpoint']).resolve()!=checkpoint
            or matching['checkpoint_SHA256']!=checksum or whole['matching_checkpoint_SHA256']!=checksum
            or whole['run_directory_name']!=run.name or not whole['whole_original_DEV128_completed']
            or not whole['all6_scope']
            or not whole['actual_end_saved_model_and_normalizer_equal_protected_pre_DEV_model']):
        raise ValueError('Matching actual whole evaluation and protected model required')
    checkpoint_before = identity(checkpoint)
    state = torch.load(io.BytesIO(owned_stable_bytes(checkpoint)), map_location='cpu', weights_only=True)
    if not finite(state) or (state['actor_updates'],state['critic_updates'])!=(whole['actor_updates'],whole['critic_updates']):
        raise ValueError('Finite model/optimizer and exact update counters required')
    if state['goal_contract']!=read_snapshot(run/'agent.yaml'):
        raise ValueError('Original controller contract differs')
    end = run/f"checkpoint_{state['critic_updates']:08d}.pt"
    if sha256(end)!=whole['actual_end_saved_checkpoint_SHA256']:
        raise ValueError('Actual saved end model differs')
    payload = run/'staged_goal_experience.pt'
    payload_before, payload_checksum = identity(payload), sha256(payload)
    experience = torch.load(payload, map_location='cpu', weights_only=True, mmap=True)
    replay = experience['executed_goal_transitions']
    n = len(replay['reward'])
    if experience['goal_contract']!=state['goal_contract'] or not n or not finite(replay) \
            or any(not bool((replay[k][:,-6]==1).all()) for k in
                   ('actor_obs','next_actor_obs','critic_obs','next_critic_obs')):
        raise ValueError('Matching finite held TRAIN replay required')
    bank = TrainSuccessBank(state['actor_obs_dim'],state['critic_obs_dim'],state['successful_train_transitions']['config'])
    bank.restore(state['successful_train_transitions'])
    original = {(r['wave'],r['environment']):r for r in read_snapshot(run/'metrics.json')['outcomes']}
    for episodes in bank.episodes.values():
        for episode in episodes:
            outcome = episode['outcome']
            actual = original[(outcome['wave'],outcome['environment'])]
            if (not supported_success(actual) or actual['split']!='train'
                    or any(actual[k]!=outcome[k] for k in ('layout','result','complete','initial_layout_valid'))
                    or episode['identity'].split('/wave')[0] not in (str(run),run.name)):
                raise ValueError('Success bank must match original safe TRAIN outcomes')
    contract = state['goal_contract']
    if contract.get('fixed_prior_radius') is None or contract.get('prior_actor_loss_disabled') is not True:
        raise ValueError('This diagnostic requires the fixed-radius actor with disabled teacher loss')
    radius = contract['fixed_prior_radius']
    goal_weight = bank.config['actor_goal_mse_weight']/radius**2
    jaw_weight = bank.config['actor_jaw_nll_weight']
    progress = min(1.,max(0.,state['actor_updates']-state['success_schedule_actor_origin'])/bank.config['fade_actor_updates'])
    fraction = bank.config['initial_replay_fraction']+progress*(bank.config['final_replay_fraction']-bank.config['initial_replay_fraction'])
    agent, _ = restored_agent(state)
    # The inference exporter intentionally omits actor-only TRAIN losses.
    # Restore all three from their validated saved contracts for this audit.
    from kuavo_isaaclab_scene.rl.multi_box.experiments.body_saturation import body_saturation_config
    from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_saturation import jaw_saturation_config
    from kuavo_isaaclab_scene.rl.multi_box.experiments.success_jaw_balance import success_jaw_balance_config
    for key, factory in (('body_saturation',body_saturation_config),
                         ('jaw_saturation',jaw_saturation_config),
                         ('success_jaw_balance',success_jaw_balance_config)):
        config = state.get(key)
        if config is not None and config!=factory(config['variant']):
            raise ValueError('Saved actor-only TRAIN loss configuration differs')
        setattr(agent,key+'_config',deepcopy(config))
    agent.requires_grad_(False)
    agent.actor.requires_grad_(True)
    model_before = structure_sha256(agent.state_dict())
    named = list(agent.actor.named_parameters())
    parameters = [p for _,p in named]
    masks = {}
    offset = 0
    for name, parameter in named:
        head = name.split('network.heads.')[1].split('.')[0] if name.startswith('network.heads.') else None
        if head is None:raise ValueError('Independent actual region heads required')
        masks.setdefault(int(head),[]).append((offset,offset+parameter.numel()))
        offset += parameter.numel()
    records = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for index in range(batches):
            ids = torch.randint(n,(256,))
            batch, retained = bank.mix({k:v[ids] for k,v in replay.items()},fraction,'cpu')
            successes = bank.sample_actor(64,'cpu')
            present = torch.cat((batch['actor_obs'][:,94:98],successes['actor_obs'][:,94:98])).argmax(-1).unique()
            if not torch.equal(present,torch.arange(len(REGIONS))):
                raise ValueError('Copied Adam comparison requires every actual head present; absent heads must retain grad=None')
            losses, class_losses = objectives(agent,batch,successes,goal_weight,jaw_weight)
            vectors = {k:gradient(loss,parameters) for k,loss in losses.items()}
            delta, pre_clip_norm = copied_adam_direction(parameters,vectors,state['optimizers'][0])
            class_gradients = {k:gradient(v[0],parameters) for k,v in class_losses.items()}
            if not torch.allclose(sum(class_gradients.values()),vectors['Q'],atol=2e-6,rtol=2e-5):
                raise ValueError('Actual region-size Q contributions do not sum to the original objective')
            if not finite(vectors):raise ValueError('Nonfinite actual actor gradient')
            by_region = {}
            for head, region in enumerate(REGIONS):
                pieces = masks[head]
                local = {k:torch.cat([v[a:b] for a,b in pieces]) for k,v in vectors.items()}
                local_delta = torch.cat([delta[a:b] for a,b in pieces])
                retention = local['successful_goal_retention']+local['successful_jaw_retention']
                divisor = local_delta.norm()*retention.norm()
                by_region[region] = directions(local) | dict(
                    copied_Adam_parameter_delta_norm=float(local_delta.norm()),
                    copied_Adam_descent_vs_retention_cosine=(float(torch.dot(-local_delta,retention)/divisor)
                        if float(divisor)>1e-15 else None),
                    copied_Adam_retention_loss_first_order_change=float(torch.dot(local_delta,retention)))
            class_reports = {}
            for key,v in class_gradients.items():
                region = key.split('/')[0]
                head = REGIONS.index(region)
                pieces = masks[head]
                local_q = torch.cat([v[a:b] for a,b in pieces])
                retained_vector = vectors['successful_goal_retention']+vectors['successful_jaw_retention']
                local_retention = torch.cat([retained_vector[a:b] for a,b in pieces])
                class_reports[key] = dict(actual_rows=class_losses[key][1],
                    global_batch_weighted_Q_gradient_norm=float(local_q.norm()),
                    Q_contribution_vs_same_region_retention_cosine=(float(torch.dot(local_q,local_retention)/(local_q.norm()*local_retention.norm()))
                        if float(local_q.norm()*local_retention.norm())>1e-15 else None))
            records.append(dict(batch=index,retained_success_rows_in_main_batch=retained,
                copied_Adam_production_max_gradient_norm=1.,gradient_norm_before_clipping=pre_clip_norm,
                weighted_objective_values={k:float(v.detach()) for k,v in losses.items()},
                **directions(vectors),by_region=by_region,Q_contributions_by_actual_region_size=class_reports))
    if (structure_sha256(agent.state_dict())!=model_before or identity(checkpoint)!=checkpoint_before
            or identity(payload)!=payload_before or sha256(checkpoint)!=checksum):
        raise ValueError('Original file or restored model changed')
    summaries = {}
    for region in REGIONS:
        values = [r['by_region'][region] for r in records]
        summaries[region] = dict(component_gradient_norm_medians={k:float(np.median([v['component_norms'][k] for v in values]))
            for k in values[0]['component_norms']},
            Q_vs_retention_cosines=[v['Q_vs_success_retention_cosine'] for v in values],
            total_vs_retention_cosines=[v['total_vs_success_retention_cosine'] for v in values],
            copied_Adam_descent_vs_retention_cosines=[v['copied_Adam_descent_vs_retention_cosine'] for v in values],
            copied_Adam_retention_loss_first_order_changes=[v['copied_Adam_retention_loss_first_order_change'] for v in values],
            actual_success_bank_episodes=len(bank.episodes[region]))
    result = dict(recorded_at=datetime.now().astimezone().isoformat(),
        role='closed_actual_TRAIN_actor_objective_gradient_diagnostic_NOT_training_or_rollout',
        run_directory_name=run.name,model_SHA256=checksum,experience_SHA256=payload_checksum,
        actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],replay_rows=n,
        actual_success_bank=bank.report(),actual_actor_goal_weight=goal_weight,actual_actor_jaw_weight=jaw_weight,
        actual_success_Q_replay_fraction=fraction,CPU_query_seed=seed,batches=records,by_region=summaries,
        original_Q_gradient_equals_sum_of_actual_region_size_contributions=True,
        exact_measured_goal_servo_encoding_and_four_binary_jaw_branches=True,
        all_saved_actor_only_TRAIN_loss_configs_restored={k:state.get(k) for k in
            ('body_saturation','jaw_saturation','success_jaw_balance')},
        actor_normalizer_frozen_and_current_critic_normalizer_held_for_diagnostic=True,
        raw_observations_and_joint_actions_NOT_exported=True,
        mature_Adam_and_production_norm1_clipping_simulated_in_disposable_parameter_copies=True,
        future_normalizer_and_critic_updates_NOT_simulated=True,
        no_original_actor_critic_optimizer_normalizer_physics_or_file_updates=True,
        no_live_payload_or_DEV_FINAL_training_queries=True,goal_not_complete=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    for name in ('run-dir','checkpoint','matching-model-proof','whole-eval-proof','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batches',type=int,default=8)
    parser.add_argument('--seed',type=int,default=9150)
    a=parser.parse_args()
    result=audit(a.run_dir.resolve(),a.checkpoint.resolve(),a.matching_model_proof,
                 a.whole_eval_proof,a.output,batches=a.batches,seed=a.seed)
    print(json.dumps({k:result[k] for k in ('actor_updates','critic_updates','actual_actor_goal_weight','by_region')}))


if __name__=='__main__':main()
