#!/usr/bin/env python3
"""Read-only CPU comparison of saved actual-flap jaw policies on fixed TRAIN.

This measures actor probabilities, not physical success or learned Q values.
Never pass an active experience file: use an immutable Drive-verified input.
"""
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
import argparse, hashlib, io, json, os

import torch

from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig, SquashedActor
from kuavo_isaaclab_scene.rl.algorithms.common import ObservationNormalizer
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import (
    ActualFlapResidualSACPilot, BoundedCorrectionHybridSAC, AbsoluteGoalJawProjector)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalProjector, CONTEXT_DIM
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import (
    SUPPLEMENTAL_DIM, supplemental_perception_contract)


def cpu_agent(state):
    contract=state['goal_contract']
    if (state['artifact_type']!=ActualFlapResidualSACPilot.artifact_type
            or contract['actor_dim']!=518 or contract['critic_dim']!=577
            or contract['supplemental_perception']!=supplemental_perception_contract()
            or state['config']['actor_feature_mode']!='flat'):
        raise ValueError('Expected the actual-flap518D/577D flat correction controller')
    nominal_dim=contract['actor_dim']-SUPPLEMENTAL_DIM
    prefix=nominal_dim-CONTEXT_DIM
    anchor_state=state['body_anchor_state']
    if anchor_state['source_goal_contract']['actor_dim']!=nominal_dim:
        raise ValueError('Frozen anchor dimension differs')
    anchor=torch.nn.ModuleDict(dict(actor=SquashedActor(nominal_dim,21,state['config']['hidden']),
        actor_normalizer=ObservationNormalizer(nominal_dim)))
    anchor.load_state_dict(anchor_state['model']);anchor.requires_grad_(False)
    prior_state=state['frozen_warm_start'].get('bc_prior',state['frozen_warm_start'])
    prior=PoseStudent(prior_state,'cpu')
    projector=StagedGoalProjector(prior,prior.agent.actor_obs_dim,True,.05)

    @torch.no_grad()
    def body_anchor(raw):
        nominal=torch.cat((raw[:,:prefix],raw[:,-CONTEXT_DIM:]),-1).clone()
        nominal[:,-1]=.05
        request=anchor['actor'](anchor['actor_normalizer'](nominal),deterministic=True)[0]
        return projector(nominal,request)[:,:19]

    agent=BoundedCorrectionHybridSAC(518,577,21,SACConfig(**state['config']),'cpu',
        action_projector=AbsoluteGoalJawProjector(),
        validated_jaw_prior_confidence=contract['validated_jaw_prior_confidence'],
        jaw_prior_residual_gain=contract['jaw_prior_residual_gain'])
    agent.correction_radius=contract['body_correction_radius'];agent.executed_body_anchor=body_anchor
    reference=torch.nn.ModuleDict(dict(actor=deepcopy(anchor['actor']),
        actor_normalizer=deepcopy(anchor['actor_normalizer'])))
    reference.load_state_dict(state['frozen_actor_prior']);reference.requires_grad_(False)
    agent.validated_jaw_prior=lambda normalized:reference['actor'].network(
        torch.cat((normalized[:,:prefix],normalized[:,-CONTEXT_DIM:]),-1)).chunk(2,-1)[0][:,19:21]
    agent.restore(state,training=False);agent.requires_grad_(False)
    restored=agent.state_dict()
    if set(restored)!=set(state['model']) or any(not torch.equal(v,restored[k]) for k,v in state['model'].items()):
        raise ValueError('CPU restored model differs from checkpoint')
    if any(not torch.isfinite(v).all() for v in restored.values()):
        raise ValueError('Checkpoint contains a nonfinite model tensor')
    return agent


def describe(values):
    return dict(minimum=float(values.min()),p10=float(values.quantile(.1)),
        median=float(values.median()),mean=float(values.mean()),p90=float(values.quantile(.9)),maximum=float(values.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experience',type=Path,required=True)
    parser.add_argument('--verified-receipt',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,action='append',required=True,
        help='Saved complete checkpoint; first is the reference policy')
    parser.add_argument('--output-json',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=1024)
    args=parser.parse_args()
    if not 1<=args.batch_size<=8192:parser.error('Batch size must be in[1,8192]')
    torch.set_num_threads(1)
    source=args.experience.resolve();receipt=json.loads(args.verified_receipt.read_text())
    entry=next(e for e in receipt['files'] if e['file']==source.name)
    if (not receipt['all_files_size_MD5_verified'] or not entry['size_MD5_verified']
            or Path(receipt['input_dir']).resolve()!=source.parent or source.stat().st_size!=entry['bytes']):
        raise ValueError('Immutable experience does not match the verified input receipt')
    signature=source.stat().st_size,source.stat().st_mtime_ns
    digest=hashlib.md5()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024**2),b''):digest.update(chunk)
    if digest.hexdigest()!=entry['MD5']:raise ValueError('Immutable TRAIN input MD5 differs')
    data=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    episodes={e['identity']:e for es in data['measured_train_credit_bank']['episodes'].values() for e in es}
    for es in data['successful_train_transitions']['episodes'].values():
        for e in es:
            if e['identity'] in episodes:
                if any(not torch.equal(v,episodes[e['identity']]['rows'][k]) for k,v in e['rows'].items()):
                    raise ValueError('Repeated TRAIN identity contains different data')
            episodes[e['identity']]=e
    gate=AbsoluteGoalJawProjector().gate;groups=defaultdict(list);identities=defaultdict(list)
    for identity,e in sorted(episodes.items()):
        outcome=e['outcome'];rows=e['rows']
        if not (outcome['complete'] and outcome['initial_layout_valid'] and outcome['split']=='train'):
            raise ValueError('Only complete valid actual TRAIN paths are allowed')
        raw=rows['actor_obs']
        safe=~(rows['critic_obs'][:,523:530]>.5).any(-1)
        valid=raw[:,510:512].sum(-1)>.5
        mask=safe&valid&gate.near(raw).all(-1)
        if not mask.any():continue
        key=(outcome['layout']['target_region'],bool(outcome['result']['success']))
        groups[key].append(raw[mask].clone());identities[key].append(identity)
    if not groups:raise ValueError('No safe valid both-near TRAIN states')
    inputs={key:torch.cat(v) for key,v in groups.items()}
    results=[];reference_logits={}
    for index,path in enumerate(args.checkpoint):
        path=path.resolve()
        if path.name.startswith('.') or '.pending' in path.name or not path.name.startswith('checkpoint_') or path.suffix!='.pt':
            raise ValueError('Pass a saved checkpoint_<iteration>.pt, not a pending file')
        stat=path.stat()
        if stat.st_uid!=os.getuid():raise ValueError('Checkpoint must belong to the current OS user')
        payload=path.read_bytes()
        if (stat.st_size,stat.st_mtime_ns)!=(path.stat().st_size,path.stat().st_mtime_ns):
            raise ValueError('Checkpoint changed while taking its CPU snapshot')
        state=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
        if state['goal_contract']!=data['goal_contract']:raise ValueError('Checkpoint and actual TRAIN contract differ')
        agent=cpu_agent(state);summaries=[]
        with torch.no_grad():
            for key,raw in sorted(inputs.items()):
                logits=torch.cat([agent.parameters_at(agent.actor_normalizer(agent.actor_features(part)))[2]
                    for part in raw.split(args.batch_size)])
                if not torch.isfinite(logits).all():raise ValueError('Nonfinite effective jaw logits')
                if not index:reference_logits[key]=logits.clone()
                base=reference_logits[key];probability=logits.sigmoid();both=probability.prod(-1)
                summaries.append(dict(region=key[0],retained_episode_success=key[1],rows=len(raw),
                    contributing_paths=identities[key],effective_logit_L_R=[describe(logits[:,h]) for h in range(2)],
                    close_probability_L_R=[describe(probability[:,h]) for h in range(2)],
                    both_close_probability=describe(both),both_greedy_close_rows=int((logits>0).all(-1).sum()),
                    negative_logit_below_minus4_L_R=[int((logits[:,h]<-4).sum()) for h in range(2)],
                    abs_logit_over4_L_R=[int((logits[:,h].abs()>4).sum()) for h in range(2)],
                    changed_greedy_jaw_sign_rows_L_R=[int(((logits[:,h]>0)!=(base[:,h]>0)).sum()) for h in range(2)],
                    logit_delta_from_reference_L_R=[describe(logits[:,h]-base[:,h]) for h in range(2)]))
        results.append(dict(checkpoint_basename=path.name,checkpoint_bytes=len(payload),
            checkpoint_SHA256=hashlib.sha256(payload).hexdigest(),actor_updates=state['actor_updates'],
            critic_updates=state['critic_updates'],jaw_saturation=state.get('jaw_saturation'),
            success_jaw_balance=state.get('success_jaw_balance'),
            jaw_behavior=state.get('jaw_behavior'),
            latest_actor_update_metrics=state.get('latest_actor_metrics',{}),
            latest_actor_metrics_may_predate_checkpoint_Q_update=True,
            all_restored_model_tensors_match_checkpoint=True,groups=summaries))
    if signature!=(source.stat().st_size,source.stat().st_mtime_ns):raise ValueError('TRAIN source changed')
    report=dict(recorded_at=datetime.now().astimezone().isoformat(),source_basename=source.name,
        source_bytes=entry['bytes'],fresh_source_MD5_verified=entry['MD5'],retained_TRAIN_paths=len(episodes),
        scope='same_saved_safe_valid_both_nominal_near_TRAIN_states',
        biased_retained_correlated_states_not_population_rate=True,
        pure_learned_policy_probabilities_without_collection_U4=True,
        comparison_is_not_physical_counterfactual_or_DEV=True,CPU_only=True,optimizer_updates=0,
        physical_rollouts=0,active_experience_or_HDF_read=False,DEV_or_FINAL_imported=False,
        source_and_live_policy_unmodified=True,goal_not_complete=True,models=results)
    args.output_json.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output_json),retained_TRAIN_paths=len(episodes),
        models=[dict(checkpoint=m['checkpoint_basename'],actor=m['actor_updates'],Q=m['critic_updates'],
            failed_groups=[dict(region=g['region'],rows=g['rows'],both_greedy_close=g['both_greedy_close_rows'],
                p_both_close=g['both_close_probability']['mean']) for g in m['groups'] if not g['retained_episode_success']])
            for m in results]),ensure_ascii=False))


if __name__=='__main__':main()
