#!/usr/bin/env python3
"""Bound a candidate capture score from immutable TRAIN's saved hand mean.

The two hand errors and individual driver angles are not recoverable from
their stored means. This tool computes intervals, never reconstructed hand
states, new rewards, replay labels or physical-success predictions.
"""
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import argparse
import hashlib
import json
import math

import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import supplemental_perception_contract


def score_bounds_from_saved_mean(mean, old_scale=.1, candidate_scale=.025):
    """Tight intervals for narrow mean and .25*sum+.5*minimum scores.

    If sL+sR=2m with sL,sR in[0,1], a narrower exponential gives sL**p
    and sR**p, p=old_scale/candidate_scale>1. For the weak-hand blend,
    sorted t=s_min lies in[max(2m-1,0),m]; its convex objective is
    .25*(2m-t)**p+.75*t**p. Its minimum includes the stationary point.
    """
    if not isinstance(mean,torch.Tensor) or not mean.is_floating_point():
        raise TypeError('Saved mean must be a floating tensor')
    if not math.isfinite(old_scale) or not math.isfinite(candidate_scale) \
            or not 0<candidate_scale<old_scale:
        raise ValueError('Candidate scale must be finite, positive and narrower')
    if not torch.isfinite(mean).all() or not ((mean>=0)&(mean<=1)).all():
        raise ValueError('Saved capture mean must lie in[0,1]')
    m=mean.to(torch.float64);power=old_scale/candidate_scale
    if not math.isfinite(power):raise ValueError('Scale ratio must be finite')
    total=2*m;low=(total-1).clamp_min(0);high=total-low
    narrow_mean_lower=m.pow(power)
    narrow_mean_upper=(low.pow(power)+high.pow(power))*.5
    ratio=math.exp(-math.log(3)/(power-1))
    stationary=total*(ratio/(1+ratio))
    t=torch.minimum(torch.maximum(stationary,low),m)
    weak_lower=.25*(total-t).pow(power)+.75*t.pow(power)
    weak_extreme=.25*high.pow(power)+.75*low.pow(power)
    weak_upper=torch.maximum(weak_extreme,m.pow(power))
    return dict(narrow_mean_lower=narrow_mean_lower,narrow_mean_upper=narrow_mean_upper,
        weak_blend_lower=weak_lower,weak_blend_upper=weak_upper)


def describe(values):
    if not len(values):return dict(count=0)
    return dict(count=len(values),minimum=float(values.min()),p10=float(values.quantile(.1)),
        median=float(values.median()),mean=float(values.mean()),p90=float(values.quantile(.9)),maximum=float(values.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experience',type=Path,required=True)
    parser.add_argument('--verified-receipt',type=Path,required=True)
    parser.add_argument('--candidate-scale',type=float,default=.025)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(1)
    source=args.experience.resolve();receipt=json.loads(args.verified_receipt.read_text())
    entry=next(e for e in receipt['files'] if e['file']==source.name)
    assert receipt['all_files_size_MD5_verified'] and entry['size_MD5_verified']
    assert Path(receipt['input_dir']).resolve()==source.parent and source.stat().st_size==entry['bytes']
    signature=source.stat().st_size,source.stat().st_mtime_ns
    digest=hashlib.md5()
    with source.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*2**20),b''):digest.update(chunk)
    assert digest.hexdigest()==entry['MD5']
    state=torch.load(source,map_location='cpu',weights_only=True,mmap=True)
    contract=state['goal_contract']
    assert contract['actor_dim']==518 and contract['critic_dim']==577
    assert contract['supplemental_perception']==supplemental_perception_contract()
    scale=contract['physical_contract']['reward_profile']['capture_scale_m'];assert scale==.1
    episodes={e['identity']:e for values in state['measured_train_credit_bank']['episodes'].values() for e in values}
    for values in state['successful_train_transitions']['episodes'].values():
        for episode in values:
            if episode['identity'] in episodes:
                assert all(torch.equal(value,episodes[episode['identity']]['rows'][key]) for key,value in episode['rows'].items())
            episodes[episode['identity']]=episode
    gate=GoalGripperProjector();groups=defaultdict(list);paths=[]
    for identity,episode in sorted(episodes.items()):
        outcome=episode['outcome'];rows=episode['rows']
        assert outcome['complete'] and outcome['initial_layout_valid'] and outcome['split']=='train'
        pre=rows['critic_obs'];post=rows['next_critic_obs'];raw=rows['next_actor_obs'];priv=post[:,464:530]
        safe=~(pre[:,523:530]>.5).any(-1)&~(post[:,523:530]>.5).any(-1)
        near=gate.near(rows['actor_obs']);closed=rows['action'][:,19:21]>0
        assert not (closed&~near).any()
        counts=torch.zeros(2,dtype=torch.long);sustained=[]
        for row in closed&near:
            counts=torch.where(row,counts+1,0);sustained.append(counts.clone())
        valid=raw[:,510:512].sum(-1)>.5
        eligible=(closed&near&(torch.stack(sustained)>=4)&(post[:,46:48]>=.95)).all(-1)&safe&valid
        force=priv[:,15:23].reshape(-1,2,2,2)*50
        in_region=priv[:,23:31].reshape(-1,2,2,2)>.5;opposed=priv[:,31:35].reshape(-1,2,2)>.5
        qualified=(force>=5).all(-1)&in_region.all(-1)&opposed;unique=qualified.sum(-1)==1
        assert torch.equal(unique,priv[:,35:37]>.5)
        bilateral=unique.all(-1)&(qualified[:,0].to(torch.long).argmax(-1)!=qualified[:,1].to(torch.long).argmax(-1))
        assert torch.equal(bilateral,(priv[:,49]>.5)&(priv[:,50]>.5))
        contact=(force>0).flatten(1).any(-1)
        categories={'neither_hand_filtered_contact':~contact,'bilateral_opposing_unique_pinch':bilateral,
            'contact_without_bilateral_opposing_pinch':contact&~bilateral}
        for category,category_mask in categories.items():
            mask=eligible&category_mask
            key=(outcome['layout']['target_region'],bool(outcome['result']['success']),category)
            capture=priv[mask,47].clone();groups[key].append(capture)
            if len(capture):paths.append(dict(identity=identity,region=key[0],episode_success=key[1],category=category,rows=len(capture)))
    summaries=[]
    for (region,success,category),parts in sorted(groups.items()):
        capture=torch.cat(parts)
        # Stored mean/exp operations were float32. Keep a numerical margin
        # rather than claiming the unknown real per-hand scores are exact.
        roundoff_margin=1e-6
        low_bounds=score_bounds_from_saved_mean((capture.to(torch.float64)-roundoff_margin).clamp(0,1),scale,args.candidate_scale)
        high_bounds=score_bounds_from_saved_mean((capture.to(torch.float64)+roundoff_margin).clamp(0,1),scale,args.candidate_scale)
        bounds={key:(low_bounds if key.endswith('_lower') else high_bounds)[key] for key in low_bounds}
        summaries.append(dict(region=region,episode_success=success,category=category,rows=len(capture),
            old_saved_mean=describe(capture),candidate_bound_distributions={key:describe(value) for key,value in bounds.items()},
            old_mean_at_least0p8=int((capture>=.8).sum()),
            candidate_weak_score_guaranteed_below0p8=int((bounds['weak_blend_upper']<.8).sum()),
            candidate_weak_score_guaranteed_at_least0p8=int((bounds['weak_blend_lower']>=.8).sum()),
            candidate_weak_score_interval_straddles0p8=int(((bounds['weak_blend_lower']<.8)&(bounds['weak_blend_upper']>=.8)).sum())))
    assert signature==(source.stat().st_size,source.stat().st_mtime_ns)
    report=dict(recorded_at=datetime.now().astimezone().isoformat(),source_bytes=entry['bytes'],fresh_source_MD5=entry['MD5'],
        source_scale_m=scale,candidate_scale_m=args.candidate_scale,power=scale/args.candidate_scale,
        mathematical_tight_intervals_not_reconstructed_hand_errors=True,
        saved_mean_roundoff_margin=1e-6,interval_guarantees_conditional_on_the_saved_exponential_mean_definition=True,
        candidate_weak_formula='.25*(sL+sR)+.5*min(sL,sR)',
        stored_actual_driver_angles_not_recoverable_from_one_clipped_closure_per_hand=True,
        CPU_only=True,physical_rollouts=0,optimizer_updates=0,reward_relabeling=False,
        candidate_score_is_not_reward_or_proven_contact_classification=True,
        diagnostic0p8_is_not_success_or_safety_gate=True,retained_actual_TRAIN_paths=len(episodes),
        correlated_retained_states_not_population_success_rate=True,
        no_active_HDF_or_replay_read=True,no_DEV_or_FINAL_imported=True,
        live_physics_reward_and_policy_unchanged=True,goal_not_complete=True,groups=summaries,paths=paths)
    args.output_json.write_text(json.dumps(report,indent=2)+'\n')
    selected=[g for g in summaries if not g['episode_success'] and g['category']=='neither_hand_filtered_contact']
    print(json.dumps(dict(output=str(args.output_json),failed_closed_no_contact_states=sum(g['rows'] for g in selected),
        old_capture_at_least0p8=sum(g['old_mean_at_least0p8'] for g in selected),
        candidate_guaranteed_below0p8=sum(g['candidate_weak_score_guaranteed_below0p8'] for g in selected),
        candidate_guaranteed_at_least0p8=sum(g['candidate_weak_score_guaranteed_at_least0p8'] for g in selected))))


if __name__=='__main__':main()
