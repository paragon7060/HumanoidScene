#!/usr/bin/env python3
"""Rollback only a goal-SAC actor; retain current Q, counters and real replay."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint

STAGED_TYPES={'staged_base_hold_remaining_goal_sac_v1','staged_base_hold_remaining_hybrid_sac_v1'}

def recover(checkpoint,best_checkpoint,output_dir,*,replay_capacity=None,
            normalize_prior_loss_by_radius=False,actor_min_replay_rows=None,
            anchor_prior_to_validated_policy=False,fixed_prior_radius=None,
            validated_jaw_prior_confidence=0.,jaw_prior_residual_gain=1.,
            body_policy_std=None,body_policy_std_cap=None):
    checkpoint,best_checkpoint,output_dir=map(Path,(checkpoint,best_checkpoint,output_dir))
    current=torch.load(checkpoint,map_location='cpu',weights_only=True)
    best=torch.load(best_checkpoint,map_location='cpu',weights_only=True)
    if (current.get('artifact_type') not in {'pose_goal_sac_no_live_reference',*STAGED_TYPES}
            or best.get('artifact_type')!=current['artifact_type']
            or current.get('goal_contract')!=best.get('goal_contract')
            or current['config']!=best['config']):
        raise ValueError('Actor recovery requires the same goal-SAC policy/physical contract')
    staged=current['artifact_type'] in STAGED_TYPES
    if current['artifact_type']=='staged_base_hold_remaining_hybrid_sac_v1' and (
            current.get('algorithm')!='hybrid_goal_sac' or best.get('algorithm')!='hybrid_goal_sac'
            or current.get('hybrid_contract')!=best.get('hybrid_contract')
            or len(current.get('optimizers',[]))!=4 or len(best.get('optimizers',[]))!=4):
        raise ValueError('Hybrid actor recovery requires matching binary Q and both entropy optimizers')
    # The input normalizer is frozen; different normalization would invalidate
    # the best network. Leave all critic normalization and optimizer state live.
    for key,value in best['model'].items():
        if key.startswith('actor_normalizer.') and not torch.equal(value,current['model'][key]):
            raise ValueError('The validated actor observation normalization differs')
        if key.startswith('actor.'):
            current['model'][key]=value
    current['optimizers'][0]['state']={}
    experience_name=('staged_goal_experience.pt' if staged
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
    policy_migration=(fixed_prior_radius is not None or validated_jaw_prior_confidence
        or body_policy_std is not None or body_policy_std_cap is not None)
    if policy_migration and not staged:
        raise ValueError('Policy exploration migration requires staged held-goal SAC')
    if fixed_prior_radius is not None and (not math.isfinite(fixed_prior_radius)
            or not .05<=fixed_prior_radius<=.15):
        raise ValueError('Fixed goal radius must be within0.05..0.15')
    if validated_jaw_prior_confidence and (current['artifact_type']!='staged_base_hold_remaining_hybrid_sac_v1'
            or not anchor_prior_to_validated_policy or not .5<validated_jaw_prior_confidence<1
            or not math.isfinite(jaw_prior_residual_gain) or jaw_prior_residual_gain<=0):
        raise ValueError('Confident binary jaws need a validated hybrid actor and finite residual gain')
    if (body_policy_std is None)!=(body_policy_std_cap is None):
        raise ValueError('Specify both body standard deviation and its cap')
    if body_policy_std is not None and not (math.isfinite(body_policy_std) and math.isfinite(body_policy_std_cap)
            and current['config']['min_policy_std']<=body_policy_std<=body_policy_std_cap<=.02):
        raise ValueError('Body exploration must fit the existing minimum and0.02 cap')
    if replay_capacity is not None or normalize_prior_loss_by_radius or actor_min_replay_rows is not None or anchor_prior_to_validated_policy or policy_migration:
        if not staged:
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
        if fixed_prior_radius is not None:
            contract['fixed_prior_radius']=fixed_prior_radius
            audit['fixed_projection_radius']=fixed_prior_radius
        if validated_jaw_prior_confidence:
            policy=dict(validated_jaw_prior_confidence=validated_jaw_prior_confidence,
                jaw_prior_residual_gain=jaw_prior_residual_gain)
            contract.update(**policy,imitation_jaw_targets='confident_frozen_validated_binary_prior')
            current['hybrid_contract'].update(**policy,
                jaw_policy='confident_validated_binary_prior_plus_trainable_logit_residual_v1')
            audit.update(**policy,binary_jaws_remain_trainable=True)
        if body_policy_std is not None:
            current['config'].update(initial_policy_std=body_policy_std,max_policy_std=body_policy_std_cap)
            contract.update(exploration_std_initial=body_policy_std,exploration_std_cap=body_policy_std_cap)
            last=max(int(k.split('.')[2]) for k in current['model']
                if k.startswith('actor.network.') and k.endswith('.weight'))
            # Preserve all21 goal means, reset only continuous log std outputs.
            continuous=19 if current['artifact_type']=='staged_base_hold_remaining_hybrid_sac_v1' else 21
            current['model'][f'actor.network.{last}.weight'][21:21+continuous].zero_()
            current['model'][f'actor.network.{last}.bias'][21:21+continuous].fill_(math.log(body_policy_std))
            audit.update(body_policy_std=body_policy_std,body_policy_std_cap=body_policy_std_cap,
                actor_goal_means_preserved=True)
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
    index=current['critic_updates'] if staged else current['actor_updates']
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
    parser.add_argument('--fixed-prior-radius',type=float)
    parser.add_argument('--validated-jaw-prior-confidence',type=float,default=0.)
    parser.add_argument('--jaw-prior-residual-gain',type=float,default=1.)
    parser.add_argument('--body-policy-std',type=float)
    parser.add_argument('--body-policy-std-cap',type=float)
    args=parser.parse_args()
    try:_,audit=recover(args.checkpoint,args.best_checkpoint,args.output_dir,
        replay_capacity=args.replay_capacity,normalize_prior_loss_by_radius=args.normalize_prior_loss_by_radius,
        actor_min_replay_rows=args.actor_min_replay_rows,
        anchor_prior_to_validated_policy=args.anchor_prior_to_validated_policy,
        fixed_prior_radius=args.fixed_prior_radius,
        validated_jaw_prior_confidence=args.validated_jaw_prior_confidence,
        jaw_prior_residual_gain=args.jaw_prior_residual_gain,
        body_policy_std=args.body_policy_std,body_policy_std_cap=args.body_policy_std_cap)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(audit))


if __name__=='__main__':main()
