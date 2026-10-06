#!/usr/bin/env python3
"""Initialize regional CPU workplace SAC from actor tensors, with fresh Q/replay."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','training-manifest','source-waypoints','workplace-results','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--selection',action='append',required=True,help='REGION=CANDIDATE, once for each of four regions')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('A unique initialization directory is required')
    selections=dict(item.split('=',1) for item in args.selection)
    if len(selections)!=len(args.selection):parser.error('Duplicate regional selections')
    import h5py,torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import build_region_workplaces
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
        CPU_PHYSICS_BACKEND,staged_solver_contract,frozen_prior_lift_contract,require_current_lift_contract)
    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import frozen_actor_reward_contract
    torch.set_num_threads(1);torch.manual_seed(20261006)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    physical=json.loads(args.training_manifest.read_text());require_current_lift_contract(physical)
    if source.get('artifact_type')!=ActualFlapResidualSACPilot.artifact_type \
            or physical.get('physics_dynamics')!=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND):
        raise ValueError('Regional initialization requires the existing actual-flap CPU PhysX actor')
    results_SHA=hashlib.sha256(args.workplace_results.read_bytes()).hexdigest()
    waypoints=build_region_workplaces(json.loads(args.source_waypoints.read_text()),
        json.loads(args.workplace_results.read_text()),selections,results_SHA256=results_SHA)
    warm=PoseGoalSACPilot(source['frozen_warm_start'],args.native_seed,
        frozen_prior_lift_contract(frozen_actor_reward_contract(physical)),args.output_dir,training=False,device='cpu')
    with h5py.File(args.native_seed[0],'r') as h:
        raw=torch.tensor(next(iter(h['episodes'].values()))['transitions/actor_obs'][:1])
    stage=StagedBaseHoldDiagnostic(warm.coordinates,waypoints,raw);goal=source['goal_contract']
    pilot=ActualFlapResidualSACPilot(warm,physical,args.output_dir,stage,body_anchor_state=source['body_anchor_state'],
        replay_capacity=goal['replay_capacity'],actor_min_replay_rows=goal['actor_min_replay_rows'],
        exploration_correlation=goal['exploration_correlation'],train_success_retention=True,device='cpu')
    proof=initialize_staged_actor_only(pilot,source,allow_workplace_change=True)
    names=[k for k in pilot.agent.state_dict() if k.startswith(('actor.','actor_normalizer.'))]
    if any(not torch.equal(v,source['model'][k]) for k,v in pilot.agent.state_dict().items() if k in names) \
            or pilot.replay.size or pilot.success_bank.size or pilot.actor_updates or pilot.critic_updates \
            or any(opt.state for opt in pilot.agent.optimizers) or pilot.agent.critic_normalizer.count:
        raise ValueError('Regional initialization changed source actor or imported incompatible learning state')
    proof.update(source_actor_tensors_bit_identical=True,actor_tensor_count=len(names),
        fresh_Q_replay_success_bank_and_four_optimizer_states=True,unproven_upper_right_grasp=True,
        source_results_SHA256=results_SHA,source_checkpoint_SHA256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True);pilot.save(final=True)
    for filename,value in [('training_manifest.json',physical),('waypoints.json',waypoints),
        ('initialization_verification.json',proof),('manifest.json',dict(goal_contract=pilot.contract,
            initialization=proof,initialized_not_trained=True,all_four_region_success_unproven=True))]:
        (args.output_dir/filename).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),verification=proof)))


if __name__=='__main__':main()
