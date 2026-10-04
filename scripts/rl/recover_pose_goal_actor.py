#!/usr/bin/env python3
"""Rollback only a goal-SAC actor; retain current Q, counters and real replay."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def recover(checkpoint,best_checkpoint,output_dir):
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
    current['actor_recovery']=audit
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())|dict(actor_recovery=audit)
    output_dir.mkdir(parents=True,exist_ok=False)
    destination=save_checkpoint(output_dir,current,current['actor_updates'],keep=None)
    torch.save(actual,output_dir/experience_name)
    (output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (output_dir/'status.json').write_text(json.dumps(dict(status='complete',actor_recovery=audit))+'\n')
    return destination,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','best-checkpoint','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    try:_,audit=recover(args.checkpoint,args.best_checkpoint,args.output_dir)
    except ValueError as error:parser.error(str(error))
    print(json.dumps(audit))


if __name__=='__main__':main()
