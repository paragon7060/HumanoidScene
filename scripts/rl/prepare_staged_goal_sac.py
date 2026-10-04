#!/usr/bin/env python3
"""Initialize a held-base 21-goal actor and fresh critics; does not train."""
import argparse
import json
from pathlib import Path

import h5py
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalSACPilot


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--free-grippers',action='store_true')
    parser.add_argument('--gripper-logit-scale',type=float,default=1.)
    args=parser.parse_args()
    if not all(p.is_file() for p in (args.checkpoint,args.waypoints,*args.native_seed)):
        parser.error('Existing matching files are required')
    with h5py.File(args.native_seed[0],'r') as source:
        meta=json.loads(source.attrs['manifest_json'])
        contract=meta['training_contract']
        episode=next(iter(source['episodes'].values()))
        raw=torch.tensor(episode['transitions/actor_obs'][0:1])
    warm=PoseGoalSACPilot(args.checkpoint,args.native_seed,contract,args.output_dir,
                         training=False,device='cpu')
    templates=json.loads(args.waypoints.read_text())
    if templates.get('physical_action_contract')!=contract['action_contract']:
        parser.error('Physical travel contract differs')
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    staged=StagedGoalSACPilot(warm,contract,args.output_dir,stage,training=True,device='cpu',
                            free_grippers=args.free_grippers,gripper_logit_scale=args.gripper_logit_scale)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    staged.save(final=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(contract|dict(
        artifact_type=staged.artifact_type,goal_contract=staged.contract,
        initialized_not_trained=True,old_Q_or_replay_imported=False),indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',
        initialized_not_trained=True,actor_updates=0,critic_updates=0))+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),actor_dim=staged.actor_dim,
        critic_dim=staged.critic_dim,action_dim=21,actor_updates=0,critic_updates=0,
        old_Q_or_replay_imported=False)))


if __name__=='__main__':main()
