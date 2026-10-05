#!/usr/bin/env python3
"""Reuse a nominal grasp actor while starting contact-reward Q/replay afresh."""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path


def compare_source_actor_behavior(pilot,state):
    """Read-only comparison on literal TRAIN actor inputs; never seed Q."""
    from copy import deepcopy
    import torch
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    batches=[e['rows']['actor_obs'][::20] for es in
        state.get('successful_train_transitions',{}).get('episodes',{}).values() for e in es]
    if not batches:raise ValueError('Actual source TRAIN actor inputs required to verify control identity')
    inputs=torch.cat(batches)
    frozen=deepcopy(pilot.frozen_actor_prior)
    frozen.load_state_dict(state['frozen_actor_prior'])
    original=pilot.agent_class(pilot.actor_dim,pilot.critic_dim,21,SACConfig(**state['config']),
        'cpu',action_projector=pilot.agent.action_projector,
        validated_jaw_prior_confidence=pilot.validated_jaw_prior_confidence,
        jaw_prior_residual_gain=pilot.jaw_prior_residual_gain)
    original.validated_jaw_prior=lambda normalized:frozen['actor'].network(normalized).chunk(2,-1)[0][:,19:21]
    original.restore(state,training=False)
    with torch.no_grad():
        old=original.act(inputs,deterministic=True);new=pilot.agent.act(inputs,deterministic=True)
    return dict(actual_TRAIN_input_rows=len(inputs),greedy_goals_exactly_equal=torch.equal(old,new),
        greedy_body_error_max=float((old[:,:19]-new[:,:19]).abs().max()),
        executed_jaw_disagreement_count=int((old[:,19:21]!=new[:,19:21]).sum()),
        comparison_only_no_rewards_or_replay_imported=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','training-manifest','waypoints','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    args=parser.parse_args()
    if args.output_dir.exists() or not all(p.is_file() for p in (args.checkpoint,args.training_manifest,args.waypoints,*args.native_seed)):
        parser.error('Existing matching inputs and a unique new output directory required')
    import h5py
    import torch
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_hybrid_goal_sac import StagedHybridGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import require_current_lift_contract,frozen_prior_lift_contract
    from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import PHYSICAL_KEYS
    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import with_contact_reward_profile
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if state.get('artifact_type')!=StagedHybridGoalSACPilot.artifact_type:
        raise ValueError('Contact initialization requires the existing held-goal hybrid SAC actor')
    base=json.loads(args.training_manifest.read_text());require_current_lift_contract(base)
    expected={k:base.get(k) for k in (*PHYSICAL_KEYS,'flap_pose_source')}
    if 'physics_dynamics' in base:expected['physics_dynamics']=base['physics_dynamics']
    source=state['goal_contract']
    if source['physical_contract']!=expected:
        raise ValueError('Source actor must match base observations, actions, dynamics, reward and safety')
    physical=with_contact_reward_profile(base)
    warm=PoseGoalSACPilot(state['frozen_warm_start'],args.native_seed,
        frozen_prior_lift_contract(base),args.output_dir,training=False,device='cpu')
    templates=json.loads(args.waypoints.read_text())
    if templates['physical_action_contract']!=physical['action_contract']:
        raise ValueError('Waypoint action contract differs')
    with h5py.File(args.native_seed[0],'r') as stream:
        raw=torch.tensor(next(iter(stream['episodes'].values()))['transitions/actor_obs'][:1])
    stage=StagedBaseHoldDiagnostic(warm.coordinates,templates,raw)
    options=dict(free_grippers=source['gripper_prior_bound'] is False,
        gripper_logit_scale=source.get('gripper_logit_scale',1.),replay_capacity=source.get('replay_capacity',20000),
        normalize_prior_loss_by_radius=source.get('normalize_prior_loss_by_radius',False),
        actor_min_replay_rows=source.get('actor_min_replay_rows',64),
        anchor_prior_to_initial_policy=source.get('actor_prior_source')=='frozen_validated_remaining_goal_actor',
        exploration_correlation=source.get('exploration_correlation',0.),
        fixed_prior_radius=source.get('fixed_prior_radius'),
        validated_jaw_prior_confidence=source.get('validated_jaw_prior_confidence',0.),
        jaw_prior_residual_gain=source.get('jaw_prior_residual_gain',1.),
        train_success_retention='train_success_retention' in source,
        episode_arm_exploration=source.get('episode_arm_exploration'))
    pilot=StagedHybridGoalSACPilot(warm,physical,args.output_dir,stage,training=True,device='cpu',**options)
    # Preserve the reviewed exploration bounds; construction still initializes
    # fresh Q, target Q, entropy, critic normalization and all optimizers.
    pilot.agent.config=replace(pilot.agent.config,initial_policy_std=state['config']['initial_policy_std'],
        max_policy_std=state['config']['max_policy_std'])
    if asdict(pilot.agent.config)!=state['config']:
        raise ValueError('Fresh contact learner must preserve the source SAC exploration configuration')
    q_before={k:v.clone() for k,v in pilot.agent.q1.state_dict().items()}
    migration=initialize_staged_actor_only(pilot,state,allow_contact_reward_change=True)
    source_prior_progress=max(0.,state['actor_updates']-state.get('prior_schedule_actor_origin',0.))
    pilot.prior_schedule_actor_origin=-source_prior_progress
    behavior=compare_source_actor_behavior(pilot,state)
    if not behavior['greedy_goals_exactly_equal']:
        raise ValueError('Fresh contact actor initialization changed the source executed policy')
    if any(not torch.equal(v,pilot.agent.q1.state_dict()[k]) for k,v in q_before.items()) \
            or pilot.actor_updates or pilot.critic_updates or pilot.replay.size \
            or any(opt.state for opt in pilot.agent.optimizers) or pilot.success_bank and pilot.success_bank.size:
        raise ValueError('Contact initialization imported old learning state')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    pilot.save(final=True)
    (args.output_dir/'training_manifest.json').write_text(json.dumps(physical,indent=2)+'\n')
    inputs={name:dict(file=str(path.resolve()),SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        for name,path in [('actor_checkpoint',args.checkpoint),('original_manifest',args.training_manifest),('waypoints',args.waypoints)]}
    manifest=dict(artifact_type=pilot.artifact_type,training_contract=physical,
        goal_contract=pilot.contract,initialized_not_trained=True,fresh_Q_replay_and_all_optimizers=True,
        actor_only_initialization=migration,successful_old_reward_rows_imported=0,
        source_prior_fade_progress_preserved=source_prior_progress,
        old_actor_prior_penalty_not_reactivated=True,initial_effective_prior_weight=pilot.prior_weight,
        exploration_preserved=True,greedy_actor_identity=behavior,inputs=inputs,created_utc=datetime.now(timezone.utc).isoformat())
    for name,value in [('manifest.json',manifest),('initialization_verification.json',migration|dict(
            fresh_Q_replay_and_all_optimizers=True,greedy_actor_identity=behavior,
            all_model_tensors_finite=all(bool(torch.isfinite(v).all()) for v in pilot.agent.state_dict().values()))),
            ('status.json',dict(status='complete',initialized_not_trained=True))]:
        (args.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(directory=str(args.output_dir.resolve()),actor_updates=0,critic_updates=0,
        replay_rows=0,source_actor_updates=state['actor_updates'],reward_profile=physical['reward_profile']['contact_shaping'])),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
