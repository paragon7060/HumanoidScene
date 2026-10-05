#!/usr/bin/env python3
"""Actor-only initialization for articulated perception and bounded SAC correction."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def verify_initial_actor(pilot,source):
    """Literal TRAIN states test control identity; no transition enters new Q."""
    import math
    import torch
    rows=[e['rows']['actor_obs'][::20] for region in
        source.get('successful_train_transitions',{}).get('episodes',{}).values() for e in region]
    if not rows:raise ValueError('Measured source TRAIN states required for initialization proof')
    nominal=torch.cat(rows)
    augmented=torch.cat((nominal[:,:-6],torch.randn(len(nominal),38),nominal[:,-6:]),-1)
    augmented[:,-1]=pilot.radius
    old_norm=pilot.body_anchor['actor_normalizer'](nominal)
    with torch.no_grad():
        mean=pilot.body_anchor['actor'].network(old_norm).chunk(2,-1)[0]
        reference=pilot.frozen_actor_prior['actor'].network(old_norm).chunk(2,-1)[0][:,19:21]
        confidence=pilot.validated_jaw_prior_confidence
        logits=torch.where(reference>0,math.log(confidence/(1-confidence)),-math.log(confidence/(1-confidence))) \
            +pilot.jaw_prior_residual_gain*(mean[:,19:21]-reference)
        old=pilot.source_projector(nominal,torch.cat((mean[:,:19].tanh(),(logits>0).to(mean)*2-1),-1))
        new=pilot.agent.act(augmented,deterministic=True)
        body_error=float((old[:,:19]-new[:,:19]).abs().max())
        changed=int((old[:,19:21]!=new[:,19:21]).sum())
        actual_logits=pilot.agent.parameters_at(pilot.agent.actor_normalizer(augmented))[2]
        jaw_logit_error=float((logits-actual_logits).abs().max())
    if body_error>1e-6 or changed or jaw_logit_error>1e-3:
        raise ValueError(f'Initial source executed policy changed: body{body_error},jaws{changed},logits{jaw_logit_error}')
    return dict(actual_TRAIN_input_rows=len(nominal),executed_body_error_max=body_error,
        executed_jaw_disagreement_count=changed,jaw_logit_error_max=jaw_logit_error,
        arbitrary_actual_extra_features_cannot_change_initial_zero_body_correction=True,
        source_radius_preserved=.05,destination_radius=pilot.radius,
        comparison_only_no_source_rewards_or_replay_imported=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','training-manifest','waypoints','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--firm-flaps',action='store_true')
    args=parser.parse_args()
    if args.output_dir.exists():parser.error('A unique output directory is required')
    import h5py
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import (
        ActualFlapResidualSACPilot,actor_anchor_state)
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_prior_lift_contract,require_current_lift_contract
    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import frozen_actor_reward_contract
    from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import supplemental_perception_contract
    from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import firm_flap_dynamics_contract
    torch.set_num_threads(2)
    source=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    physical=json.loads(args.training_manifest.read_text());require_current_lift_contract(physical)
    if physical['reward_profile'].get('contact_shaping',{}).get('name')!='opposing_pad_contact_progress_v1':
        raise ValueError('Bounded contact correction requires the reviewed contact reward')
    warm=PoseGoalSACPilot(source['frozen_warm_start'],args.native_seed,
        frozen_prior_lift_contract(frozen_actor_reward_contract(physical)),args.output_dir,training=False,device='cpu')
    physical=physical|dict(supplemental_perception=supplemental_perception_contract())
    if args.firm_flaps:physical['flap_dynamics']=firm_flap_dynamics_contract()
    templates=json.loads(args.waypoints.read_text())
    if templates['physical_action_contract']!=physical['action_contract']:
        raise ValueError('Waypoint travel differs')
    with h5py.File(args.native_seed[0],'r') as stream:
        raw=torch.tensor(next(iter(stream['episodes'].values()))['transitions/actor_obs'][:1])
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    pilot=ActualFlapResidualSACPilot(warm,physical,args.output_dir,stage,body_anchor_state=actor_anchor_state(source),
        device='cpu',replay_capacity=500000,actor_min_replay_rows=32768,
        exploration_correlation=.99,train_success_retention=True)
    identity=verify_initial_actor(pilot,source)
    if pilot.replay.size or pilot.actor_updates or pilot.critic_updates or pilot.success_bank.size \
            or any(opt.state for opt in pilot.agent.optimizers) or pilot.agent.critic_normalizer.count:
        raise ValueError('New perception/dynamics imported incompatible learning state')
    args.output_dir.mkdir(parents=True,exist_ok=False);pilot.save(final=True)
    (args.output_dir/'training_manifest.json').write_text(json.dumps(physical,indent=2)+'\n')
    manifest=dict(artifact_type=pilot.artifact_type,training_contract=physical,goal_contract=pilot.contract,
        initialized_not_trained=True,initial_actor_identity=identity,fresh_Q_replay_and_all_optimizers=True,
        source_actor_updates=source['actor_updates'],source_critic_updates_not_imported=source['critic_updates'],
        source_checkpoint_SHA256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        source_checkpoint=str(args.checkpoint.resolve()),prior_loss_reactivated=False,
        created_utc=datetime.now(timezone.utc).isoformat())
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (args.output_dir/'initialization_verification.json').write_text(json.dumps(identity|dict(
        all_model_tensors_finite=all(bool(torch.isfinite(v).all()) for v in pilot.agent.state_dict().values()),
        fresh_Q_replay_optimizers_and_success_bank=True),indent=2)+'\n')
    (args.output_dir/'status.json').write_text(json.dumps(dict(status='complete',initialized_not_trained=True))+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),actor_dim=pilot.actor_dim,critic_dim=pilot.critic_dim,
        fresh_actual_replay_rows=0,source_actor_updates=source['actor_updates'],identity=identity,firm_flaps=args.firm_flaps)))


if __name__=='__main__':main()
