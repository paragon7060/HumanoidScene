#!/usr/bin/env python3
"""Collect balanced physical layout waves with one held-base SAC learner.

Only active, physically settled held-phase transitions train Q. Finished
environments are excluded until the next neutral whole-wave reset. Evaluation
waves never update models or enter replay. No live reference/IK controller.
"""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import signal
import time


def main():
    from isaaclab.app import AppLauncher
    from kuavo_isaaclab_scene.robots.base_drive import add_base_drive_cli_args,export_base_drive_cli
    from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args,export_gripper_cli
    from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args,export_robot_model_cli
    from kuavo_isaaclab_scene.workcell.rack_rollers import add_rack_roller_cli_args,export_rack_roller_cli
    parser=argparse.ArgumentParser(description=__doc__)
    AppLauncher.add_app_launcher_args(parser)
    for name in ('checkpoint','demo-dataset','training-manifest','waypoints','waves-json','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--native-seed',type=Path,action='append',required=True)
    parser.add_argument('--training',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--stop-on-validation-regression',action='store_true')
    parser.add_argument('--minimum-validation-region-success-rate',type=float,default=0.)
    parser.add_argument('--contact-stability-probe',action='store_true',
        help='Frozen-only finer physics integration/box solver diagnostic; cannot seed or train Q')
    parser.add_argument('--steps',type=int,default=900)
    add_robot_model_cli_args(parser);add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser);add_base_drive_cli_args(parser)
    parser.set_defaults(headless=True,robot_model='s63',gripper='leju-twofinger',rack_rollers=True)
    args=parser.parse_args()
    if not 1<=args.steps<=900 or args.output_dir.exists():parser.error('New output and 1..900 steps required')
    if args.contact_stability_probe and args.training:
        parser.error('Contact stability probe changes solver dynamics and is frozen-only')
    waves=json.loads(args.waves_json.read_text())
    n=len(waves[0]['layouts']) if waves else 0
    if not 1<=n<=128 or any(len(w['layouts'])!=n for w in waves):
        parser.error('Each wave must contain the same 1..128 independent layouts')
    for w in waves:
        if w['split'] not in ('train','validation','holdout'):parser.error('Unknown wave split')
        if w['split']=='train' and not args.training:parser.error('Frozen runs cannot contain TRAIN waves')
        expected='train' if w['split']=='train' else 'holdout'
        if any(row['layout']['split']!=expected for row in w['layouts']):parser.error('Mixed wave splits')
    training_seeds={r['layout']['seed'] for w in waves if w['split']=='train' for r in w['layouts']}
    final_seeds={r['layout']['seed'] for w in waves if w['split']!='train' for r in w['layouts']}
    if training_seeds&final_seeds:parser.error('TRAIN and FINAL seeds overlap')
    validation_seeds={r['layout']['seed'] for w in waves if w['split']=='validation' for r in w['layouts']}
    heldout_seeds={r['layout']['seed'] for w in waves if w['split']=='holdout' for r in w['layouts']}
    if validation_seeds&heldout_seeds:parser.error('Development and independent final seeds overlap')
    if args.stop_on_validation_regression and (not args.training or waves[0]['split']!='validation'):
        parser.error('Regression monitoring requires TRAIN and an initial development wave')
    if not 0<=args.minimum_validation_region_success_rate<=1 or (
            args.minimum_validation_region_success_rate and not args.stop_on_validation_regression):
        parser.error('A development floor within0..1 requires the regression guard')
    if any(len({r['layout']['seed'] for r in w['layouts']})!=n for w in waves):
        parser.error('Each parallel wave needs distinct initial cases')
    if args.robot_model!='s63' or args.gripper!='leju-twofinger' or not args.rack_rollers:
        parser.error('Current held-base checkpoint requires S63/Leju/rack rollers')
    export_robot_model_cli(args);export_gripper_cli(args);export_rack_roller_cli(args);export_base_drive_cli(args)
    app=AppLauncher(args).app
    stopped={'value':False};signal.signal(signal.SIGTERM,lambda *_:stopped.update(value=True))
    env=recorder=None;output=args.output_dir.resolve()
    try:
        import torch
        import numpy as np
        from isaaclab.envs import ManagerBasedRLEnv
        from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
        from kuavo_isaaclab_scene.rl.multi_box.training_env_cfg import MultiBoxGraspAssemblyEnvCfg
        from kuavo_isaaclab_scene.rl.multi_box.rewards import MultiBoxRewardWeights
        from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import GRASP_ASSIGNMENT_SCALE_M
        from kuavo_isaaclab_scene.rl.multi_box.state.isaac_privileged_grasp import GRASP_APPROACH_REWARD_SCALE_M,GRASP_CAPTURE_REWARD_SCALE_M
        from kuavo_isaaclab_scene.rl.multi_box.metrics.potentials import FRONT_STAGE_CLEARANCE_M,FRONT_STAGE_LANE_TOLERANCE_M,FRONT_STAGE_REWARD_SCALE_M
        from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import configure_vr_torso_up_diagnostic,select_reference_episode
        from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
        from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import GraspLayout,layout_reset_observation
        from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalSACPilot
        from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages,settle_batched_layouts,DevelopmentSuccessGuard
        from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import GraspActionProjector
        from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
        from kuavo_isaaclab_scene.rl.multi_box.debug.contact_sensors import V2_RACK_SENSOR_NAMES,V2_COLLISION_BODY_NAMES
        from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import filtered_force_by_body
        from kuavo_isaaclab_scene.workcell.rack_rollers import resolve_rack_roller_settings
        from kuavo_isaaclab_scene.recording.rl_initial_state import capture_rl_initial_state
        from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder
        contract=json.loads(args.training_manifest.read_text())
        if contract.get('reward_profile',{}).get('weights')!=asdict(MultiBoxRewardWeights()):
            raise ValueError('Current physical reward weights differ from checkpoint input manifest')
        cfg=MultiBoxGraspAssemblyEnvCfg(num_envs=n);cfg.episode_length_s=30.
        cfg.multi_box=replace(cfg.multi_box,self_collision_enabled=contract['self_collision']['enabled'])
        cfg.sim.device=args.device or 'cuda:0'
        solver_probe=None
        if args.contact_stability_probe:
            from isaaclab.sim import RigidBodyPropertiesCfg
            from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
            cfg.sim.dt=1/240;cfg.decimation=8;cfg.sim.render_interval=8
            for name in physical_asset_names():
                spawn=getattr(cfg.scene,name).spawn
                spawn.rigid_props=RigidBodyPropertiesCfg(max_linear_velocity=25.,
                    max_angular_velocity=10000.,max_depenetration_velocity=2.)
                spawn.articulation_props.solver_position_iteration_count=64
                spawn.articulation_props.solver_velocity_iteration_count=16
            solver_probe=dict(frozen_only=True,physics_dt_s=1/240,control_dt_s=1/30,
                box_position_iterations=64,box_velocity_iterations=16,
                box_max_linear_velocity_mps=25.,box_max_angular_velocity_degps=10000.,
                box_max_depenetration_velocity_mps=2.,success_and_safety_unchanged=True,
                Q_import_eligible=False)
        profile=dict(weights=asdict(MultiBoxRewardWeights()),approach_scale_m=GRASP_APPROACH_REWARD_SCALE_M,
            assignment_scale_m=GRASP_ASSIGNMENT_SCALE_M,capture_scale_m=GRASP_CAPTURE_REWARD_SCALE_M,
            front_stage_clearance_m=FRONT_STAGE_CLEARANCE_M,front_stage_lane_tolerance_m=FRONT_STAGE_LANE_TOLERANCE_M,
            front_stage_scale_m=FRONT_STAGE_REWARD_SCALE_M,geometry_profile='rack_front_lane_then_opposing_flap_reach_v3')
        thresholds=dict(rack_contact_force_n=float(cfg.multi_box.rack_contact_force),
            obstacle_contact_force_n=float(cfg.task.obstacle_contact_force),workspace_radius_m=float(cfg.multi_box.workspace_radius),
            max_box_lift_height_m=float(cfg.multi_box.max_box_lift_height),
            max_box_linear_speed_mps=float(cfg.multi_box.max_box_linear_speed),
            max_box_angular_speed_radps=float(cfg.multi_box.max_box_angular_speed))
        if profile!=contract['reward_profile'] or thresholds!=contract['terminal_contract']['safety_thresholds']:
            raise ValueError('Current reward geometry or physical termination differs')
        contract=configure_vr_torso_up_diagnostic(cfg,contract,.06)|{'num_envs':n}
        if contract.get('flap_pose_source','nominal')!='nominal':
            raise ValueError('Batched prototype requires its explicitly nominal observation checkpoint')
        class WaveEnv(TerminalObservationMixin,ManagerBasedRLEnv):pass
        env=WaveEnv(cfg);env.enable_numerical_dynamics_recovery()
        dims={k:list(v) for k,v in env.observation_manager.group_obs_dim.items()}
        actions={k:env.action_manager.get_term(k).action_dim for k in env.action_manager.active_terms}
        if dims!=contract['observations'] or actions!=contract['actions']:
            raise ValueError('Batched physical observation/action widths differ')
        base=env.action_manager.get_term('base');upper=env.action_manager.get_term('upper_body')
        head=env.action_manager.get_term('head');height=env.action_manager.get_term('height')
        validate_goal_feedback_rates(base._scale,upper._scale,head._scale,height.cfg.speed_m_s,env.step_dt)
        batch,_=load_v2_grasp_demonstrations(args.demo_dataset,self_collision_enabled=cfg.multi_box.self_collision_enabled)
        sources={i:select_reference_episode(batch,i)['actor_obs'][0] for i in
                 {row['episode_index'] for w in waves for row in w['layouts']}}
        state=torch.load(args.checkpoint,map_location=env.device,weights_only=True)
        if state.get('artifact_type')!=StagedGoalSACPilot.artifact_type:
            raise ValueError('Batched learner requires the separately initialized staged checkpoint')
        warm=PoseGoalSACPilot(state['frozen_warm_start'],args.native_seed,contract,output,training=False,device=env.device)
        templates=json.loads(args.waypoints.read_text())
        if templates['physical_action_contract']!=contract['action_contract']:raise ValueError('Waypoint travel differs')
        projection=GraspActionProjector(list(actions.items()))
        # Match the single-env reference lifecycle: the rack and root must
        # physically settle before their measured pose anchors new layouts.
        from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _settle_initial_resets
        initial,_=env.reset(seed=42)
        _settle_initial_resets(env,initial)
        output.mkdir(parents=True,exist_ok=False)
        meta=dict(task_family='multi_box_v2',skill='grasp',robot_model='s63',gripper='leju-twofinger',
            rack_rollers=True,actor_obs_dim=464,critic_obs_dim=530,action_dim=24,
            action_terms=list(map(list,actions.items())),control_dt=env.step_dt,episode_seconds=30.,
            collection_source=('changed_contact_solver_frozen_probe_NOT_matching_Q_replay'
                               if solver_probe else StagedGoalSACPilot.artifact_type),training_contract=contract,
            contact_stability_probe=solver_probe,
            sim_device=str(env.device),multi_box=asdict(cfg.multi_box),old_demo_rewards_used=False,
            current_reward_verified_against_breakdown=True,
            initial_poses='independent_neutral_layouts_from_original_demo_then_physics_settled',
            episode_layouts=[dict(wave=i,environment=j,**row) for i,w in enumerate(waves) for j,row in enumerate(w['layouts'])])
        recorder=RlTransitionRecorder(output/'executed_transitions.hdf5',meta)
        (output/'manifest.json').write_text(json.dumps(contract|{'artifact_type':StagedGoalSACPilot.artifact_type,
            'training':args.training,'layout_waves':waves,'no_live_VR_or_IK':True},indent=2)+'\n')
        if solver_probe:
            manifest=json.loads((output/'manifest.json').read_text())
            (output/'manifest.json').write_text(json.dumps(manifest|{'contact_stability_probe':solver_probe},indent=2)+'\n')
        (output/'env.yaml').write_text(json.dumps(asdict(cfg.multi_box),indent=2)+'\n')
        outcomes=[];pilot=None;start=time.monotonic();total_rows=0
        guard=DevelopmentSuccessGuard(args.minimum_validation_region_success_rate) if args.stop_on_validation_regression else None
        development_checks=[]
        for wave_index,wave in enumerate(waves):
            if stopped['value']:break
            actors=torch.stack([layout_reset_observation(sources[r['episode_index']],
                GraspLayout(**r['layout']).validate(),cfg.multi_box,
                roller_clearance_m=resolve_rack_roller_settings().box_clearance_m) for r in wave['layouts']]).to(env.device)
            observation,settled=settle_batched_layouts(env,actors)
            stages=BatchedBaseStages(warm.coordinates,templates,observation['policy'])
            if pilot is None:
                pilot=StagedGoalSACPilot(warm,contract,output,stages.stages[0],checkpoint=args.checkpoint,
                    training=args.training,device=env.device)
                (output/'agent.yaml').write_text(json.dumps(pilot.contract,indent=2)+'\n')
            pilot.training=args.training and wave['split']=='train'
            updates_before=(pilot.actor_updates,pilot.critic_updates,pilot.replay.size)
            active=torch.ones(n,device=env.device,dtype=torch.bool)
            snapshots=[capture_rl_initial_state(env,observation,env_index=i) for i in range(n)]
            rows=[[] for _ in range(n)];last=[None]*n;buffer={}
            compute=env.termination_manager.compute
            def capture_before_reset():
                result=compute();g=env._multi_box_privileged_grasp_step;s=env._multi_box_grasp_safety_step
                force=filtered_force_by_body(env,V2_RACK_SENSOR_NAMES)
                buffer.update(success=env.termination_manager.get_term('success').clone(),
                    unsafe=env.termination_manager.get_term('unsafe').clone(),
                    invalid_reset=env.termination_manager.get_term('invalid_reset').clone(),
                    time_out=env.termination_manager.get_term('time_out').clone(),
                    distance=g.matched_flap_distance_m.clone(),pinch=g.pinch.hand_pinching.clone(),
                    stable=g.stable_hands.clone(),proof_lift=g.success.proof_lift.clone(),
                    opposing=g.success.opposing_flaps.clone(),hold=g.success.hold_time_s.clone(),
                    clearance=g.rack_clearance_m.clone(),force=force.max(-1).values.clone(),
                    body=force.argmax(-1).clone(),base_pose=env.scene['robot'].data.root_pose_w.clone(),
                    causes={k:getattr(s,k).clone() for k in ('invalid_box_pose','invalid_flap_pose',
                        'robot_rack_collision','self_collision','obstacle_collision','workspace_limit',
                        'box_drop','box_lift_limit','box_speed_limit')},
                    box_pose=g.box_pose_world.clone(),box_velocity=g.box_velocity_world.clone(),
                    logical=g.target_logical_id.clone(),pool=g.target_pool_id.clone())
                return result
            env.termination_manager.compute=capture_before_reset
            rollout_start=time.monotonic();wave_rows=0
            try:
                with torch.no_grad():
                    for step in range(args.steps):
                        if stopped['value'] or not active.any():break
                        pre={k:v.clone() for k,v in observation.items()}
                        ids=stages.update(pre['policy'],env.scene['robot'].data.root_lin_vel_w,
                            env.scene['robot'].data.root_ang_vel_w,active,step)
                        action=stages.approach_commands(pre['policy'],active);action[:,20:22]=-1.
                        previous=None
                        if len(ids):
                            pilot.stage=stages.held_context(ids);pilot.anchor=stages.anchors[ids].clone()
                            clocks=stages.clocks(ids,step)
                            command,previous=pilot.act(pre['policy'][ids],
                                torch.cat((pre['policy'],pre['critic']),-1)[ids],clocks)
                            action[ids]=command
                        if not torch.allclose(projection(pre['policy'],action),action,atol=1e-6,rtol=0):
                            raise ValueError('Generated and executed jaw projections differ')
                        observation,reward,terminated,truncated,info=env.step(action)
                        if (info['transition_numerical_failure']&active).any():raise ValueError('Numerical failure in active layout')
                        terminal=info['transition_next_observations']
                        if not torch.allclose(reward,env._multi_box_grasp_reward_breakdown.total,atol=1e-5,rtol=1e-5):
                            raise ValueError('Actual vector reward differs from current breakdown')
                        if previous is not None:
                            pilot.observe(previous,terminal['policy'][ids],
                                torch.cat((terminal['policy'],terminal['critic']),-1)[ids],reward[ids],terminated[ids],clocks)
                        # One transfer per field, rather than per environment.
                        # Scene collection remains exactly the executed tensor.
                        pc={k:v.cpu().numpy() for k,v in pre.items()}
                        tc={k:v.cpu().numpy() for k,v in terminal.items()}
                        ac=action.cpu().numpy();rc=reward.cpu().numpy()
                        te=terminated.cpu().numpy();tr=truncated.cpu().numpy()
                        bc={k:({a:b.cpu().numpy() for a,b in v.items()} if isinstance(v,dict)
                               else v.cpu().numpy()) for k,v in buffer.items()}
                        for i in torch.where(active)[0].tolist():
                            r=dict(actor_obs=pc['policy'][i].copy(),
                                critic_obs=np.concatenate((pc['policy'][i],pc['critic'][i])),
                                action=ac[i].copy(),reward=float(rc[i]),
                                next_actor_obs=tc['policy'][i].copy(),
                                next_critic_obs=np.concatenate((tc['policy'][i],tc['critic'][i])),
                                terminated=bool(te[i]),truncated=bool(tr[i]),
                                success=bool(bc['success'][i]),unsafe=bool(bc['unsafe'][i]),
                                sim_time_s=env.common_step_counter*env.step_dt)
                            rows[i].append(r);total_rows+=1;wave_rows+=1
                            last[i]=dict(steps=step+1,success=r['success'],unsafe=r['unsafe'],
                                invalid_reset=bool(bc['invalid_reset'][i]),time_out=bool(bc['time_out'][i]),
                                pinching=bc['pinch'][i].tolist(),stable_hands=bc['stable'][i].tolist(),
                                proof_lift=bool(bc['proof_lift'][i]),opposing_flaps=bool(bc['opposing'][i]),
                                hold_time_s=float(bc['hold'][i]),rack_clearance_m=float(bc['clearance'][i]),
                                flap_distances=bc['distance'][i].tolist(),rack_peak_force_n=float(bc['force'][i]),
                                rack_peak_body=V2_COLLISION_BODY_NAMES[int(bc['body'][i])],
                                box_pose_world=bc['box_pose'][i].tolist(),
                                box_velocity_world=bc['box_velocity'][i].tolist(),
                                target_logical_id=int(bc['logical'][i]),target_pool_id=int(bc['pool'][i]),
                                unsafe_causes={k:bool(v[i]) for k,v in bc['causes'].items()},
                                staged_base=stages.stages[i].report())
                        active&=~(terminated|truncated)
                        if step%30==0:
                            progress=dict(wave=wave_index,split=wave['split'],step=step+1,
                                active=int(active.sum()),held=len(ids),actual_rows=total_rows,
                                transition_per_s=total_rows/(time.monotonic()-start),learner=pilot.report(),last=last)
                            progress['rollout_transition_per_s']=wave_rows/(time.monotonic()-rollout_start)
                            (output/'progress.json').write_text(json.dumps(progress)+'\n')
                            print('[BATCHED SAC] '+json.dumps({k:v for k,v in progress.items() if k not in ('learner','last')})+
                                f' actor={pilot.actor_updates} critic={pilot.critic_updates}',flush=True)
                        if pilot.training and pilot.critic_updates and pilot.critic_updates%1024==0:pilot.save()
            finally:
                env.termination_manager.compute=compute
            for i,samples in enumerate(rows):
                recorder.start_episode(initial_state=snapshots[i])
                for row in samples:recorder.append(row)
                success=bool(last[i] and last[i]['success'])
                recorder.finish_episode(success=success,reason='success' if success else 'failure' if not active[i] else 'interrupted_or_step_limit')
                outcomes.append(dict(wave=wave_index,split=wave['split'],environment=i,
                    layout=wave['layouts'][i]['layout'],result=last[i],complete=bool(not active[i])))
            if wave['split']!='train' and updates_before!=(pilot.actor_updates,pilot.critic_updates,pilot.replay.size):
                raise ValueError('Evaluation modified optimizer counters or replay')
            pilot.training=args.training
            if args.training:pilot.save(final=True)
            regression=baseline_failed=False
            if guard is not None and wave['split']=='validation':
                check=guard.evaluate(wave['layouts'],last,wave_index)
                development_checks.append(dict(wave=wave_index,actor_updates=pilot.actor_updates,
                    critic_updates=pilot.critic_updates,**check))
                regression=check['regression']
                baseline_failed=check['baseline_failed']
            (output/'metrics.json').write_text(json.dumps(dict(policy=pilot.artifact_type,outcomes=outcomes,
                learner=pilot.report(),actual_rows=total_rows,seconds=time.monotonic()-start,
                development_checks=development_checks),indent=2)+'\n')
            status='policy_regression' if regression else 'baseline_performance_failed' if baseline_failed else 'training' if wave_index+1<len(waves) else 'complete'
            (output/'status.json').write_text(json.dumps(dict(status=status,
                completed_waves=wave_index+1,total_waves=len(waves),interrupted=stopped['value']))+'\n')
            if regression or baseline_failed:
                print('[DEVELOPMENT PERFORMANCE STOP] '+json.dumps(development_checks[-1]),flush=True)
                return 1
        if stopped['value']:
            (output/'status.json').write_text(json.dumps(dict(status='interrupted',completed_waves=len(outcomes)//n))+'\n')
        return 0
    except Exception:
        import traceback
        failure=traceback.format_exc()
        print(failure,flush=True)
        output.mkdir(parents=True,exist_ok=True)
        if not (output/'manifest.json').exists():
            (output/'manifest.json').write_text(json.dumps(dict(artifact_type='batched_staged_runtime_failure',
                waves=waves,exception_before_collection=True))+'\n')
        (output/'failure.json').write_text(json.dumps(dict(traceback=failure))+'\n')
        (output/'status.json').write_text(json.dumps(dict(status='failed'))+'\n')
        raise
    finally:
        if recorder:recorder.close()
        if env:env.close()
        app.close()


if __name__=='__main__':raise SystemExit(main())
