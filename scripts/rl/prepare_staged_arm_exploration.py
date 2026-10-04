#!/usr/bin/env python3
"""Prepare a new experiment from immutable, fully backed-up staged inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.episode_arm_exploration import enable_episode_arm_exploration
from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--verified-backup-receipt',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    source=args.checkpoint.resolve().parent
    receipt=json.loads(args.verified_backup_receipt.read_text())
    status=json.loads((source/'status.json').read_text())
    manifest=json.loads((source/'manifest.json').read_text())
    if (Path(receipt['directory']).resolve()!=source or receipt.get('final_size_md5_verified') is not True
            or status.get('status')!='complete'
            or not (status.get('initialized_not_trained') is True
                    or status.get('initialized_not_new_training') is True)
            or args.output_dir.exists()):
        parser.error('Require immutable initialized inputs, matching full backup receipt and new output')
    checkpoint=args.checkpoint.resolve();replay=source/'staged_goal_experience.pt'
    before={str(p):digest(p) for p in (checkpoint,replay)}
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    experience=torch.load(replay,map_location='cpu',weights_only=True,mmap=True)
    if manifest['goal_contract']!=state['goal_contract']:
        parser.error('Source manifest/checkpoint contract differs')
    if not all(bool(torch.isfinite(v).all()) for v in state['model'].values()):
        parser.error('Source model contains nonfinite tensors')
    new_state,new_experience,audit=enable_episode_arm_exploration(state,experience)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    save_checkpoint(args.output_dir,new_state,state['critic_updates'],keep=None)
    torch.save(new_experience,args.output_dir/'staged_goal_experience.pt')
    after={str(p):digest(p) for p in (checkpoint,replay)}
    if before!=after:raise ValueError('Immutable source inputs changed during preparation')
    prepared=torch.load(args.output_dir/'staged_goal_experience.pt',map_location='cpu',weights_only=True,mmap=True)
    if any(not torch.equal(v,prepared['executed_goal_transitions'][k])
           for k,v in experience['executed_goal_transitions'].items()):
        raise ValueError('Saved actual replay differs from the source')
    audit.update(source_checkpoint_sha256=before[str(checkpoint)],source_replay_sha256=before[str(replay)],
        immutable_source_size_md5_backup_verified=True, source_inputs_unchanged_after_preparation=True,
        saved_actual_replay_tensor_equality_verified=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest|dict(goal_contract=new_state['goal_contract'],
        episode_arm_behavior_initialization=audit,initialized_not_trained=True),indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',initialized_not_trained=True,
        actor_updates=state['actor_updates'],critic_updates=state['critic_updates'],new_optimizer_updates=0))+'\n')
    (args.output_dir/'episode_arm_initialization_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),**audit)))


if __name__=='__main__':main()
