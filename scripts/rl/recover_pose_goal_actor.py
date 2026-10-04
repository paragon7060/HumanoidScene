#!/usr/bin/env python3
"""Rollback only a goal-SAC actor; retain current Q, counters and real replay."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def recover(checkpoint,best_checkpoint,output_dir,*,replay_capacity=None,
            normalize_prior_loss_by_radius=False,actor_min_replay_rows=None,
            anchor_prior_to_validated_policy=False):
    checkpoint,best_checkpoint,output_dir=map(Path,(checkpoint,best_checkpoint,output_dir))
    current=torch.load(checkpoint,map_location='cpu',weights_only=True)
    best=torch.load(best_checkpoint,map_location='cpu',weights_only=True)
    if (current.get('artifact_type') not in {'pose_goal_sac_no_live_reference','staged_base_hold_remaining_goal_sac_v1'}
            or best.get('artifact_type')!=current['artifact_type']
            or current.get('goal_contract')!=best.get('goal_contract')
            or current['config']!=best['config']):
        raise ValueError('Actor recovery requires the same goal-SAC policy/physical contract')
    # The input normalizer is frozen; different normalization would invalidate
    # the best network. Leave all critic normalization and optimizer state live.
    for key,value in best['model'].items():
        if key.startswith('actor_normalizer.') and not torch.equal(value,current['model'][key]):
            raise ValueError('The validated actor observation normalization differs')
        if key.startswith('actor.'):
            current['model'][key]=value
    current['optimizers'][0]['state']={}
    experience_name=('staged_goal_experience.pt' if current['artifact_type']=='staged_base_hold_remaining_goal_sac_v1'
                     else 'pose_goal_experience.pt')
    actual=torch.load(checkpoint.parent/experience_name,map_location='cpu',weights_only=True)
    if actual.get('goal_contract')!=current['goal_contract']:
        raise ValueError('Latest measured replay differs from recovered actor contract')
    audit=dict(current_checkpoint=str(checkpoint.resolve()),best_checkpoint=str(best_checkpoint.resolve()),
               current_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               best_sha256=hashlib.sha256(best_checkpoint.read_bytes()).hexdigest(),
               actor_restored=True,actor_optimizer_moments_reset=True,
               latest_critic_optimizer_and_replay_preserved=True,
               actor_updates_not_reset=True,actual_rows=len(actual['executed_goal_transitions']['action']))
    if replay_capacity is not None or normalize_prior_loss_by_radius or actor_min_replay_rows is not None or anchor_prior_to_validated_policy:
        if current['artifact_type']!='staged_base_hold_remaining_goal_sac_v1':
            raise ValueError('Training migration options require the staged21-goal controller')
        contract=dict(current['goal_contract'])
        if type(anchor_prior_to_validated_policy) is not bool or (
                anchor_prior_to_validated_policy and not normalize_prior_loss_by_radius):
            raise ValueError('Validated actor anchoring requires normalized prior loss')
        capacity=contract.get('replay_capacity',20000) if replay_capacity is None else replay_capacity
        minimum=contract.get('actor_min_replay_rows',64) if actor_min_replay_rows is None else actor_min_replay_rows
        if type(capacity) is not int or not max(1024,audit['actual_rows'])<=capacity<=2000000 \
                or type(minimum) is not int or not 64<=minimum<=capacity:
            raise ValueError('Migration cannot discard actual replay; actor warmup must fit the buffer')
        if capacity!=20000:contract['replay_capacity']=capacity
        else:contract.pop('replay_capacity',None)
        if minimum!=64:contract['actor_min_replay_rows']=minimum
        else:contract.pop('actor_min_replay_rows',None)
        if normalize_prior_loss_by_radius:
            best_contract=best['goal_contract']
            if best_contract.get('normalize_prior_loss_by_radius',False):
                progress=max(0,best['actor_updates']-best.get('prior_schedule_actor_origin',0.))/best_contract['fade_actor_updates']
            else:
                progress=max(0,best['critic_updates']-best_contract['initial_critic_warmup'])/best_contract['fade_critic_updates']
            progress=min(1.,progress)
            # Match the validated actor's projection radius. Actual optimizer
            # counters and all measured Q/replay remain unchanged.
            fade=contract['fade_critic_updates']//4
            current['prior_schedule_actor_origin']=current['actor_updates']-progress*fade
            contract.update(normalize_prior_loss_by_radius=True,
                prior_loss_units='mean_squared_normalized_goal_error_divided_by_current_radius_squared',
                prior_fade_units='actor_updates_after_schedule_origin',fade_actor_updates=fade)
            audit.update(validated_policy_radius_preserved=.05+.10*progress,
                         actor_prior_schedule_origin=current['prior_schedule_actor_origin'])
        if anchor_prior_to_validated_policy:
            current['frozen_actor_prior']={k:v.clone() for k,v in best['model'].items()
                if k.startswith(('actor.','actor_normalizer.'))}
            contract['actor_prior_source']='frozen_validated_remaining_goal_actor'
            audit['regularization_anchor_is_validated_actor']=True
        current['goal_contract']=actual['goal_contract']=contract
        audit.update(replay_capacity=capacity,actor_min_replay_rows=minimum,
                     normalized_prior_loss=normalize_prior_loss_by_radius,
                     actual_transition_tensors_unchanged=True,
                     physical_control_and_observation_contract_unchanged=True)
        current['latest_actor_metrics']={}
    current['actor_recovery']=audit
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())|dict(
        actor_recovery=audit,goal_contract=current['goal_contract'])
    output_dir.mkdir(parents=True,exist_ok=False)
    index=current['critic_updates'] if current['artifact_type']=='staged_base_hold_remaining_goal_sac_v1' else current['actor_updates']
    destination=save_checkpoint(output_dir,current,index,keep=None)
    torch.save(actual,output_dir/experience_name)
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output_dir/'status.json').write_text(json.dumps(dict(status='complete',actor_recovery=audit))+'\n')
    return destination,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','best-checkpoint','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--replay-capacity',type=int)
    parser.add_argument('--normalize-prior-loss-by-radius',action='store_true')
    parser.add_argument('--actor-min-replay-rows',type=int)
    parser.add_argument('--anchor-prior-to-validated-policy',action='store_true')
    args=parser.parse_args()
    try:_,audit=recover(args.checkpoint,args.best_checkpoint,args.output_dir,
        replay_capacity=args.replay_capacity,normalize_prior_loss_by_radius=args.normalize_prior_loss_by_radius,
        actor_min_replay_rows=args.actor_min_replay_rows,
        anchor_prior_to_validated_policy=args.anchor_prior_to_validated_policy)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(audit))


if __name__=='__main__':main()
