#!/usr/bin/env python3
"""Initialize fresh physical-command SAC from a frozen actor and closed TRAIN successes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):result.update(block)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('frozen-actor','native-successes','success-outcomes','training-manifest','waypoints','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--source-run',required=True,help='Original TRAIN run basename; no DEV/FINAL data')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('Use a unique, new output directory')
    import h5py
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_prior_lift_contract
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_sac import PhysicalBodySACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_native_seed import seed_physical_training_successes
    state=torch.load(args.frozen_actor,map_location='cpu',weights_only=True)
    physical=json.loads(args.training_manifest.read_text())
    templates=json.loads(args.waypoints.read_text())
    if templates['physical_action_contract']!=physical['action_contract']:
        raise ValueError('Held waypoints and physical travel contract differ')
    before=(args.native_successes.stat().st_size,args.native_successes.stat().st_mtime_ns)
    with h5py.File(args.native_successes,'r') as stream:
        meta=json.loads(stream.attrs['manifest_json'])
        if meta.get('source_run_basename',args.source_run)!=args.source_run:
            raise ValueError('Native corpus provenance differs from declared source run')
        raw=torch.tensor(next(iter(stream['episodes'].values()))['transitions/actor_obs'][:1])
    warm=PoseGoalSACPilot(state['frozen_warm_start'],args.native_seed,
        frozen_prior_lift_contract(physical),args.output_dir,training=False)
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    pilot=PhysicalBodySACPilot(warm,physical,args.output_dir,stage,frozen_goal_state=state)
    outcomes=json.loads(args.success_outcomes.read_text())
    provenance=seed_physical_training_successes(pilot,args.native_successes,outcomes,source_run=args.source_run)
    if before!=(args.native_successes.stat().st_size,args.native_successes.stat().st_mtime_ns):
        raise ValueError('Native source changed during reading; only closed immutable data may seed Q')
    if pilot.actor_updates or pilot.critic_updates or pilot.agent.q_optimizer.state or pilot.agent.actor_optimizer.state:
        raise ValueError('Initialization unexpectedly performed optimization')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    pilot.save(final=True)
    manifest=dict(artifact_type=pilot.artifact_type,physical_body_contract=pilot.contract,
        training_contract=physical,native_TRAIN_seed_provenance=provenance,
        initialized_utc=datetime.now(timezone.utc).isoformat(),new_training_updates=0,
        inputs={name:dict(path=str(path.resolve()),sha256=digest(path)) for name,path in
            [('frozen_actor',args.frozen_actor),('native_successes',args.native_successes),
             ('success_outcomes',args.success_outcomes),('waypoints',args.waypoints),
             ('training_manifest',args.training_manifest)]})
    audit=dict(actor_updates=0,critic_updates=0,fresh_Q_and_optimizer=True,
        actual_native_rows=pilot.replay.size,actual_success_bank=pilot.success_bank.report(),
        source_actor_updates=pilot.frozen_goal_actor['source_actor_updates'],
        initialization_is_not_a_new_physical_success=True)
    for name,value in [('manifest.json',manifest),('physical_initialization_audit.json',audit),
                       ('status.json',dict(status='complete',initialized_not_trained=True))]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),**audit)),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
