#!/usr/bin/env python3
"""Initialize CPU PhysX SAC from the actual-flap actor only, with fresh Q/replay."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','training-manifest','waypoints','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--initialization-seed',type=int,default=20261006)
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('A unique output directory is required')
    if not 0<=args.initialization_seed<2**32:parser.error('Initialization seed must be within0..2**32-1')
    import h5py
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
        CPU_PHYSICS_BACKEND,staged_solver_contract,frozen_prior_lift_contract,require_current_lift_contract)
    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import frozen_actor_reward_contract
    torch.set_num_threads(2);torch.manual_seed(args.initialization_seed)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if source.get('artifact_type')!=ActualFlapResidualSACPilot.artifact_type:
        raise ValueError('CPU initialization requires the measured actual-flap actor')
    physical=json.loads(args.training_manifest.read_text());require_current_lift_contract(physical)
    if physical.get('physics_dynamics')!=staged_solver_contract('PGS'):
        raise ValueError('CPU actor initialization requires the reviewed GPU PGS source contract')
    from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import firm_flap_dynamics_contract
    if physical.get('flap_dynamics')!=firm_flap_dynamics_contract():
        raise ValueError('CPU experiment must preserve reviewed firm flap randomization')
    current=deepcopy(physical)
    current['physics_dynamics']=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND)
    warm=PoseGoalSACPilot(source['frozen_warm_start'],args.native_seed,
        frozen_prior_lift_contract(frozen_actor_reward_contract(current)),args.output_dir,training=False,device='cpu')
    templates=json.loads(args.waypoints.read_text())
    if templates['physical_action_contract']!=current['action_contract']:raise ValueError('Waypoint travel differs')
    with h5py.File(args.native_seed[0],'r') as stream:
        raw=torch.tensor(next(iter(stream['episodes'].values()))['transitions/actor_obs'][:1])
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    goal=source['goal_contract']
    pilot=ActualFlapResidualSACPilot(warm,current,args.output_dir,stage,
        body_anchor_state=source['body_anchor_state'],device='cpu',
        replay_capacity=goal['replay_capacity'],actor_min_replay_rows=goal['actor_min_replay_rows'],
        exploration_correlation=goal['exploration_correlation'],train_success_retention=True)
    migration=initialize_staged_actor_only(pilot,source)
    tensor_names=[k for k in pilot.agent.state_dict() if k.startswith(('actor.','actor_normalizer.'))]
    if any(not torch.equal(pilot.agent.state_dict()[k],source['model'][k]) for k in tensor_names):
        raise ValueError('CPU initialization changed the source actor tensors')
    if pilot.replay.size or pilot.actor_updates or pilot.critic_updates or pilot.success_bank.size \
            or any(opt.state for opt in pilot.agent.optimizers) or pilot.agent.critic_normalizer.count:
        raise ValueError('CPU initialization imported incompatible Q/replay/learning state')
    if not all(bool(torch.isfinite(v).all()) for v in pilot.agent.state_dict().values()):
        raise ValueError('Nonfinite initialized model')
    proof=migration|dict(source_actor_tensors_bit_identical=True,compared_actor_tensor_count=len(tensor_names),
        source_body_anchor_preserved=True,physics_backend=CPU_PHYSICS_BACKEND,
        new_actor_Q_updates_zero=True,new_replay_and_success_bank_empty=True,
        four_optimizer_states_empty=True,critic_normalizer_count_zero=True,
        source_checkpoint_SHA256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        no_DEV_or_FINAL_transitions_imported=True,initialized_not_trained=True)
    args.output_dir.mkdir(parents=True,exist_ok=False);pilot.save(final=True)
    (args.output_dir/'training_manifest.json').write_text(json.dumps(current,indent=2)+'\n')
    (args.output_dir/'manifest.json').write_text(json.dumps(dict(artifact_type=pilot.artifact_type,
        training_contract=current,goal_contract=pilot.contract,initialization_verification=proof,
        source_checkpoint=str(args.checkpoint.resolve()),initialized_not_trained=True,
        initialization_seed=args.initialization_seed,created_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n')
    (args.output_dir/'initialization_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',initialized_not_trained=True))+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),verification=proof)))


if __name__=='__main__':main()
