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
    parser.add_argument('--actor-checkpoint',type=Path,
        help='Matching held-goal actor/normalizer only; Q/replay/all optimizers remain fresh')
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--waypoints',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--free-grippers',action='store_true')
    parser.add_argument('--gripper-logit-scale',type=float,default=1.)
    parser.add_argument('--replay-capacity',type=int,default=20000)
    parser.add_argument('--normalize-prior-loss-by-radius',action='store_true')
    parser.add_argument('--actor-min-replay-rows',type=int,default=64)
    parser.add_argument('--anchor-prior-to-initial-policy',action='store_true')
    parser.add_argument('--exploration-correlation',type=float,default=0.)
    parser.add_argument('--hybrid-grippers',action='store_true',
        help='Fresh Q with19 continuous goals and exact two-Bernoulli jaw policy')
    parser.add_argument('--physics-solver',choices=('TGS','PGS'),
        help='Explicit fresh-Q dynamics identity; the source is used as actor prior only')
    args=parser.parse_args()
    if not all(p.is_file() for p in (args.checkpoint,args.waypoints,*args.native_seed,
                                   *((args.actor_checkpoint,) if args.actor_checkpoint else ()))):
        parser.error('Existing matching files are required')
    with h5py.File(args.native_seed[0],'r') as source:
        meta=json.loads(source.attrs['manifest_json'])
        contract=meta['training_contract']
        episode=next(iter(source['episodes'].values()))
        raw=torch.tensor(episode['transitions/actor_obs'][0:1])
    from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import grasp_lift_terminal_contract
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_prior_lift_contract
    warm=PoseGoalSACPilot(args.checkpoint,args.native_seed,frozen_prior_lift_contract(contract),args.output_dir,
                         training=False,device='cpu')
    contract=contract|dict(terminal_contract=contract['terminal_contract']|grasp_lift_terminal_contract())
    if args.physics_solver:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import staged_solver_contract
        contract=contract|dict(physics_dynamics=staged_solver_contract(args.physics_solver))
    templates=json.loads(args.waypoints.read_text())
    if templates.get('physical_action_contract')!=contract['action_contract']:
        parser.error('Physical travel contract differs')
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    pilot_class=StagedGoalSACPilot
    if args.hybrid_grippers:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_hybrid_goal_sac import StagedHybridGoalSACPilot
        pilot_class=StagedHybridGoalSACPilot
        if not args.free_grippers:parser.error('Hybrid jaw decisions require --free-grippers')
    staged=pilot_class(warm,contract,args.output_dir,stage,training=True,device='cpu',
                            free_grippers=args.free_grippers,gripper_logit_scale=args.gripper_logit_scale,
                            replay_capacity=args.replay_capacity,
                            normalize_prior_loss_by_radius=args.normalize_prior_loss_by_radius,
                            actor_min_replay_rows=args.actor_min_replay_rows,
                            anchor_prior_to_initial_policy=args.anchor_prior_to_initial_policy,
                            exploration_correlation=args.exploration_correlation)
    migration={}
    if args.actor_checkpoint:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
        migration=initialize_staged_actor_only(staged,torch.load(args.actor_checkpoint,map_location='cpu',weights_only=True))
    args.output_dir.mkdir(parents=True,exist_ok=False)
    staged.save(final=True)
    (args.output_dir/'manifest.json').write_text(json.dumps(contract|dict(
        artifact_type=staged.artifact_type,goal_contract=staged.contract,
        initialized_not_trained=True,old_Q_or_replay_imported=False,
        actor_only_initialization=migration),indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',
        initialized_not_trained=True,actor_updates=0,critic_updates=0))+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),actor_dim=staged.actor_dim,
        critic_dim=staged.critic_dim,action_dim=21,actor_updates=0,critic_updates=0,
        old_Q_or_replay_imported=False)))


if __name__=='__main__':main()
