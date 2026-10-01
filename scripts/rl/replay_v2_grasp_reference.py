#!/usr/bin/env python3
"""Compare a VR controller or deterministic SAC actor from an inferred VR seed."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import signal


def main():
    from isaaclab.app import AppLauncher
    from kuavo_isaaclab_scene.robots.base_drive import add_base_drive_cli_args, export_base_drive_cli
    from kuavo_isaaclab_scene.robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from kuavo_isaaclab_scene.workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli

    parser = argparse.ArgumentParser(description=__doc__)
    # AppLauncher inspects the parser before adding its flags. Register it
    # before required task arguments so --help works without a dummy dataset.
    AppLauncher.add_app_launcher_args(parser)
    parser.add_argument('--demo-dataset', type=Path, required=True)
    parser.add_argument('--episode-index', type=int, default=0)
    parser.add_argument('--training-manifest', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--capture-every', type=int, default=30)
    parser.add_argument('--no-video', action='store_true')
    parser.add_argument('--actor-checkpoint', type=Path,
                        help='Evaluate this deterministic actor instead of the VR/live-IK controller.')
    parser.add_argument('--executed-actions', type=Path,
                        help='Reproduce actual current GPU success commands as an open-loop diagnostic, not a learned policy.')
    parser.add_argument('--actor-reference-mix', type=float,
                        help='DAgger diagnostic: mix this fraction of actor commands into the VR controller and export actor-only correction labels.')
    parser.add_argument('--body-envelope', action='store_true',
                        help='Diagnostic only: bound actor body commands; does not change SAC training.')
    parser.add_argument('--joint-offset-rad', type=float,
                        help='Diagnostic only: decode a fitted joint-goal actor through the unchanged delta servo.')
    parser.add_argument('--residual-sac', action='store_true',
                        help='Fixed-scene SAC pilot: learn bounded corrections to measured --executed-actions; reference remains required.')
    parser.add_argument('--residual-checkpoint', type=Path)
    parser.add_argument('--residual-training', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--residual-scale', type=float, default=.05)
    parser.add_argument('--residual-controller',choices=('delta','goal'),default='delta',
                        help='Goal mode anchors accumulated base/torso/joint goals to measured reference controller state.')
    parser.add_argument('--steps', type=int, default=900)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    add_base_drive_cli_args(parser)
    parser.set_defaults(headless=True, robot_model='s63', gripper='leju-twofinger', rack_rollers=True)
    args = parser.parse_args()
    if args.capture_every < 1 or args.episode_index < 0 or not 1 <= args.steps <= 900:
        parser.error('Capture interval must be positive, episode index nonnegative, and steps in1..900')
    if args.body_envelope and not args.actor_checkpoint:
        parser.error('--body-envelope requires --actor-checkpoint')
    if args.joint_offset_rad is not None and (not args.actor_checkpoint or args.body_envelope
                                             or args.actor_reference_mix is not None):
        parser.error('Joint-goal comparison requires a matching actor and excludes other mix/envelope variants')
    if args.actor_checkpoint and not args.actor_checkpoint.is_file():
        parser.error('Missing actor checkpoint')
    if args.executed_actions and (args.actor_checkpoint or not args.executed_actions.is_file()):
        parser.error('--executed-actions needs an existing native dataset and excludes --actor-checkpoint')
    if args.residual_sac and (not args.executed_actions or not 0 < args.residual_scale <= .2):
        parser.error('Residual pilot needs measured commands and a scale in (0,.2]')
    if args.residual_checkpoint and (not args.residual_sac or not args.residual_checkpoint.is_file()):
        parser.error('Residual checkpoint requires --residual-sac and an existing file')
    if args.residual_sac and not args.residual_training and not args.residual_checkpoint:
        parser.error('Residual evaluation requires a learned residual checkpoint')
    if args.actor_reference_mix is not None and (
            not 0 <= args.actor_reference_mix <= .2 or not args.actor_checkpoint or args.body_envelope):
        parser.error('DAgger mix requires an actor checkpoint, fraction in0..0.2 and no body envelope')
    if args.robot_model != 's63' or args.gripper != 'leju-twofinger' or not args.rack_rollers:
        parser.error('This reference replay requires S63/Leju and rack rollers')
    if args.output_dir.exists():
        parser.error('Use a new, unique output directory')
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)
    export_base_drive_cli(args)
    stopped = {'value': False}
    app = AppLauncher(args).app
    # Kit installs native handlers during startup. Register ours afterwards,
    # so SIGTERM requests a loop exit and HDF/video close rather than entering
    # Kit shutdown asynchronously inside a physics callback.
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(value=True))
    env = recorder = writer = None
    try:
        import cv2
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from cpu_scene_video import SceneVideo
        from kuavo_isaaclab_scene.recording.rl_initial_state import capture_rl_initial_state
        from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder
        from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
        from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
        from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import GraspActionProjector
        from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import (
            VRJointTracker, select_reference_episode, settle_reference_scene,
        )
        from kuavo_isaaclab_scene.rl.multi_box.rewards import MultiBoxRewardWeights
        from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import GRASP_ASSIGNMENT_SCALE_M
        from kuavo_isaaclab_scene.rl.multi_box.state.isaac_privileged_grasp import (
            GRASP_APPROACH_REWARD_SCALE_M, GRASP_CAPTURE_REWARD_SCALE_M,
        )
        from kuavo_isaaclab_scene.rl.multi_box.metrics.potentials import (
            FRONT_STAGE_CLEARANCE_M, FRONT_STAGE_LANE_TOLERANCE_M, FRONT_STAGE_REWARD_SCALE_M,
        )
        from kuavo_isaaclab_scene.rl.multi_box.training_env_cfg import MultiBoxGraspAssemblyEnvCfg
        from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _settle_initial_resets

        contract = json.loads(args.training_manifest.read_text())
        if (contract.get('task_family') != 'multi_box_v2' or contract.get('skill') != 'grasp'
                or contract.get('robot_model') != 's63' or contract.get('gripper') != 'leju-twofinger'
                or contract.get('flap_pose_source', 'nominal') != 'nominal'
                or contract.get('reward_profile', {}).get('weights') != asdict(MultiBoxRewardWeights())):
            raise ValueError('Reference replay needs the current nominal v2 grasp contract/rewards')
        cfg = MultiBoxGraspAssemblyEnvCfg(num_envs=1)
        cfg.episode_length_s = 30.
        cfg.multi_box = replace(cfg.multi_box, self_collision_enabled=contract['self_collision']['enabled'])
        cfg.sim.device = args.device or 'cuda:0'
        profile = dict(weights=asdict(MultiBoxRewardWeights()), approach_scale_m=GRASP_APPROACH_REWARD_SCALE_M,
            assignment_scale_m=GRASP_ASSIGNMENT_SCALE_M, capture_scale_m=GRASP_CAPTURE_REWARD_SCALE_M,
            front_stage_clearance_m=FRONT_STAGE_CLEARANCE_M, front_stage_lane_tolerance_m=FRONT_STAGE_LANE_TOLERANCE_M,
            front_stage_scale_m=FRONT_STAGE_REWARD_SCALE_M,
            geometry_profile='rack_front_lane_then_opposing_flap_reach_v3')
        thresholds = dict(rack_contact_force_n=float(cfg.multi_box.rack_contact_force),
            obstacle_contact_force_n=float(cfg.task.obstacle_contact_force), workspace_radius_m=float(cfg.multi_box.workspace_radius),
            max_box_lift_height_m=float(cfg.multi_box.max_box_lift_height),
            max_box_linear_speed_mps=float(cfg.multi_box.max_box_linear_speed),
            max_box_angular_speed_radps=float(cfg.multi_box.max_box_angular_speed))
        if profile != contract['reward_profile'] or thresholds != contract['terminal_contract']['safety_thresholds'] \
                or contract['action_contract'] != 's63_upright_torso_xz_fixed_pitch_v1':
            raise ValueError('Reference reward geometry, safety thresholds or torso control changed; supply current manifest')

        class ReplayEnv(TerminalObservationMixin, ManagerBasedRLEnv):
            pass

        env = ReplayEnv(cfg)
        env.enable_numerical_dynamics_recovery()
        dims = {key: list(value) for key, value in env.observation_manager.group_obs_dim.items()}
        actions = {name: env.action_manager.get_term(name).action_dim
                   for name in env.action_manager.active_terms}
        if dims != contract['observations'] or actions != contract['actions'] \
                or contract.get('action_projection') != GraspActionProjector.name:
            raise ValueError('Reference replay action/observation contract differs')
        batch, _ = load_v2_grasp_demonstrations(args.demo_dataset,
            self_collision_enabled=cfg.multi_box.self_collision_enabled)
        demo = select_reference_episode(batch, args.episode_index)
        observation, _ = env.reset(seed=42)
        observation, _ = _settle_initial_resets(env, observation)
        observation, rack, settling_steps = settle_reference_scene(env, demo)
        projection = GraspActionProjector(list(actions.items()))
        teacher = None
        agent = state = limits = executed = joint_goal = residual = None
        initial_actor_error = None
        controller_name = 'VR_reference_plus_contact_confirmed_IK_NOT_SAC'
        if args.executed_actions:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import read_executed_successes
            measured, _ = read_executed_successes(args.executed_actions, contract)
            executed = select_reference_episode(measured, args.episode_index)
            initial_actor_error = float((observation['policy'][0].cpu()-executed['actor_obs'][0]).abs().max())
            if initial_actor_error > 1e-5:
                raise ValueError(f'Recorded-command replay starts with a different actor observation: {initial_actor_error}')
            controller_name = 'recorded_current_GPU_actions_open_loop_NOT_SAC'
            if args.residual_sac:
                if args.residual_controller == 'goal':
                    base=env.action_manager.get_term('base')
                    upper=env.action_manager.get_term('upper_body')
                    head=env.action_manager.get_term('head')
                    height=env.action_manager.get_term('height')
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
                    validate_goal_feedback_rates(base._scale,upper._scale,head._scale,
                                                 height.cfg.speed_m_s,env.step_dt)
                from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import ResidualSACPilot
                residual = ResidualSACPilot(executed, args.executed_actions, args.output_dir,
                    scale=args.residual_scale, device=env.device, checkpoint=args.residual_checkpoint,
                    training=args.residual_training,controller_mode=args.residual_controller)
                controller_name = 'fixed_scene_measured_reference_plus_SAC_residual_NOT_standalone_SAC'
        elif args.actor_checkpoint:
            from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
            from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
            from kuavo_isaaclab_scene.rl.runners.storage import load_checkpoint
            from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import PHYSICAL_KEYS
            source = json.loads((args.actor_checkpoint.parent/'manifest.json').read_text())
            for key in (*PHYSICAL_KEYS, 'flap_pose_source'):
                if source.get(key) != contract.get(key):
                    raise ValueError(f'Actor checkpoint physical contract differs: {key}')
            state = load_checkpoint(args.actor_checkpoint, device=env.device)
            if state.get('diagnostic_joint_offset_rad') != args.joint_offset_rad:
                raise ValueError('Actor joint-output coordinates differ from the requested diagnostic')
            agent = AsymmetricSAC(dims['policy'][0], dims['policy'][0]+dims['critic'][0],
                sum(actions.values()), SACConfig(**state['config']), env.device,
                action_projector=projection)
            agent.restore(state, training=False)
            agent.eval()
            controller_name = 'deterministic_SAC_actor_on_inferred_VR_scene'
            if args.joint_offset_rad is not None:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.joint_offset import JointOffsetController
                joint_goal = JointOffsetController(args.joint_offset_rad)
                controller_name = 'diagnostic_joint_goal_BC_actor_NOT_trained_SAC'
            if args.body_envelope:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.action_envelope import diagnostic_body_limits
                limits = diagnostic_body_limits(actions.items(), device=env.device)
                controller_name += '_diagnostic_body_envelope'
            if args.actor_reference_mix is not None:
                teacher = VRJointTracker(env, demo, rack)
                controller_name = 'mixed_VR_teacher_actor_DAgger_NOT_pure_SAC'
        else:
            teacher = VRJointTracker(env, demo, rack)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        if residual:
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type': 'fixed_scene_reference_residual_sac',
                'residual_contract': residual.contract}, indent=2)+'\n')
            (output/'env.yaml').write_text(json.dumps({'physical_contract':contract},indent=2)+'\n')
            (output/'agent.yaml').write_text(json.dumps({'residual_contract':residual.contract,
                'sac_config':asdict(residual.agent.config)},indent=2)+'\n')
            (output/'status.json').write_text(json.dumps({'status':'training' if args.residual_training else 'evaluating'})+'\n')
        meta = dict(task_family='multi_box_v2', skill='grasp', robot_model='s63',
            gripper='leju-twofinger', rack_rollers=True, controller_mapping='scaled',
            action_dim=sum(actions.values()), actor_obs_dim=dims['policy'][0],
            critic_obs_dim=dims['policy'][0]+dims['critic'][0], action_terms=list(map(list, actions.items())),
            control_dt=env.step_dt, multi_box=asdict(cfg.multi_box), episode_seconds=30.,
            collection_source=('fixed_scene_reference_residual_sac' if residual else
                               'mixed_VR_actor_DAgger' if args.actor_reference_mix is not None else
                               'executed_action_reproduction' if executed is not None else
                               'SAC_actor_reference_scene_evaluation' if agent else
                               'current_v2_environment_executed_vr_reference'),
            sim_device=str(env.device), current_reward_verified_against_breakdown=True,
            training_contract=contract, initial_poses='inferred_from_original_demo_then_physics_settled',
            old_demo_rewards_used=False, old_policy_executed=agent is not None,
            source_dataset=str(args.demo_dataset.resolve()), source_episode_index=args.episode_index,
            reset_kinematics='coherent_gpu_articulation_fk_v2',
            actor_checkpoint=str(args.actor_checkpoint.resolve()) if agent else None,
            executed_action_source=str(args.executed_actions.resolve()) if executed is not None else None,
            body_envelope_diagnostic=args.body_envelope, actor_reference_mix=args.actor_reference_mix,
            diagnostic_joint_offset_rad=args.joint_offset_rad,
            residual_contract=residual.contract if residual else None)
        recorder = RlTransitionRecorder(output/'executed_transitions.hdf5', meta)
        recorder.start_episode(initial_state=capture_rl_initial_state(env, observation))
        renderer = None if args.no_video else SceneVideo(env,
            caption=(f'Reference + SAC {args.residual_controller} residual | fixed scene | train={args.residual_training}' if residual else
                     f'Joint-goal BC actor | NOT trained SAC | offset {args.joint_offset_rad}rad' if joint_goal else
                     f'VR teacher + {args.actor_reference_mix:.0%} actor | NOT pure SAC' if args.actor_reference_mix is not None else
                     f'Recorded actual commands | NOT SAC | actual {env.device} PhysX' if executed is not None else
                     f'Actor | SAC updates={state.get("actor_updates", "unknown")} | envelope={args.body_envelope} | {env.device}' if agent else
                     f'VR reference + live IK | NOT SAC | actual {env.device} PhysX'))
        video_name = 'policy.mp4' if agent else 'reference.mp4'
        if renderer:
            writer = cv2.VideoWriter(str(output/video_name), cv2.VideoWriter_fourcc(*'mp4v'),
                                    30/args.capture_every, (960, 720))
            if not writer.isOpened():
                raise ValueError('Could not open reference video writer')
        history, pixels, counts = [], {}, dict(success=0, unsafe=0, invalid_reset=0, time_out=0)
        compute = env.termination_manager.compute

        def before_reset():
            result = compute()
            grasp = env._multi_box_privileged_grasp_step
            from kuavo_isaaclab_scene.rl.multi_box.debug.contact_sensors import V2_RACK_SENSOR_NAMES, V2_COLLISION_BODY_NAMES
            from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import filtered_force_by_body
            force = filtered_force_by_body(env, V2_RACK_SENSOR_NAMES)[0]
            safety = env._multi_box_grasp_safety_step
            row = dict(step=len(history)+1, pinching=grasp.pinch.hand_pinching[0].tolist(),
                flap_distances=grasp.matched_flap_distance_m[0].tolist(),
                box_pose=grasp.box_pose_world[0].tolist(), phase=int(teacher.phase[0]) if teacher else None,
                ik_position_errors=[float(s.target_position_error()[0]) for s in teacher.solvers] if teacher else None,
                action=action[0].tolist(), rack_peak_force=float(force.max()),
                rack_peak_body=V2_COLLISION_BODY_NAMES[int(force.argmax())],
                box_velocity=grasp.box_velocity_world[0].tolist(),
                unsafe_causes={key: bool(getattr(safety, key)[0]) for key in (
                    'invalid_box_pose', 'invalid_flap_pose', 'robot_rack_collision',
                    'self_collision', 'obstacle_collision', 'workspace_limit',
                    'box_drop', 'box_lift_limit', 'box_speed_limit')},
                **{key: bool(env.termination_manager.get_term(key)[0]) for key in counts})
            history.append(row)
            for key in counts:
                counts[key] += int(row[key])
            if renderer and (row['step'] % args.capture_every == 1 or row['success'] or row['unsafe']):
                frame = renderer.frame(env, row['step'], float(grasp.raw.matched_flap_distance_m[0]),
                                       bool(grasp.pinch.hand_pinching.all()))
                cv2.putText(frame, f"pinch L/R={row['pinching']} | success={int(row['success'])}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                pixels['frame'] = frame
            return result

        env.termination_manager.compute = before_reset
        frames = 0
        label_observations, label_actions = [], []
        with torch.no_grad():
            for step in range(min(args.steps, len(executed['action']))
                              if executed is not None and not residual else args.steps):
                if stopped['value']:
                    break
                pre = {key: value.clone() for key, value in observation.items()}
                if not bool(torch.isfinite(pre['policy']).all()):
                    raise ValueError('Nonfinite actor observation in physical replay')
                actor_input = joint_goal.actor_input(pre['policy']) if joint_goal else pre['policy']
                if residual:
                    raw_critic = torch.cat((pre['policy'], pre['critic']), -1)
                    action, residual_previous = residual.act(pre['policy'], raw_critic, step)
                else:
                    action = (executed['action'][step:step+1].to(env.device) if executed is not None else
                          agent.act(actor_input, deterministic=True) if agent else
                          projection(pre['policy'], teacher.act(pre['policy'])))
                if joint_goal:
                    action = joint_goal.physical_commands(pre['policy'], action)
                if args.actor_reference_mix is not None:
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.action_envelope import mix_teacher_actor
                    correction = projection(pre['policy'], teacher.act(pre['policy']))
                    label_observations.append(pre['policy'].cpu().clone())
                    label_actions.append(correction.cpu().clone())
                    action = mix_teacher_actor(correction, action, args.actor_reference_mix,
                                               gripper_columns=projection.columns)
                if limits is not None:
                    action = action.clamp(-limits, limits)
                observation, reward, terminated, truncated, info = env.step(action)
                terminal = info['transition_next_observations']
                if bool(info['transition_numerical_failure'].any()):
                    raise ValueError('Numerical recovery during replay; attempt cannot enter Q')
                if not torch.allclose(reward, env._multi_box_grasp_reward_breakdown.total, atol=1e-5, rtol=1e-5):
                    raise ValueError('Current executed reward differs from reward breakdown')
                if residual:
                    residual.observe(residual_previous, terminal['policy'],
                        torch.cat((terminal['policy'], terminal['critic']), -1),
                        reward, terminated, step)
                    if args.residual_training and residual.actor_updates and residual.actor_updates % 512 == 0:
                        residual.save()
                row = history[-1]
                recorder.append(dict(actor_obs=pre['policy'][0].cpu().numpy(),
                    critic_obs=torch.cat((pre['policy'], pre['critic']), -1)[0].cpu().numpy(),
                    action=action[0].cpu().numpy(), reward=float(reward[0]),
                    next_actor_obs=terminal['policy'][0].cpu().numpy(),
                    next_critic_obs=torch.cat((terminal['policy'], terminal['critic']), -1)[0].cpu().numpy(),
                    terminated=bool(terminated[0]), truncated=bool(truncated[0]),
                    success=row['success'], unsafe=row['unsafe'], sim_time_s=env.common_step_counter*env.step_dt))
                if 'frame' in pixels:
                    frame = pixels.pop('frame')
                    writer.write(frame)
                    if frames == 0 or row['success'] or row['unsafe']:
                        cv2.imwrite(str(output/'preview.png'), frame)
                    frames += 1
                if step % 30 == 0:
                    (output/'progress.json').write_text(json.dumps(row))
                    print(f"[REFERENCE] step={step+1} pinch={row['pinching']} success={row['success']}", flush=True)
                    if residual:
                        print('[RESIDUAL SAC] '+json.dumps(residual.report()), flush=True)
                if bool((terminated | truncated)[0]):
                    recorder.finish_episode(success=row['success'],
                        reason='success' if row['success'] else 'failure')
                    break
        if label_actions:
            # A correction action was proposed, not necessarily executed.
            # This archive deliberately has no model, Q, reward or next state.
            from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint
            save_checkpoint(output, dict(algorithm='asymmetric_sac',
                actor_obs_dim=dims['policy'][0], action_dim=sum(actions.values()),
                actor_labels_only=True, teacher_imitation={
                    'actor_obs': torch.cat(label_observations), 'action': torch.cat(label_actions)}),
                iteration=0, keep=None)
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type': 'actor_labels_only', 'collection_source': meta['collection_source'],
                'actor_reference_mix': args.actor_reference_mix,
                'correction_labels_are_Q_transitions': False}, indent=2)+'\n')
        if residual and args.residual_training:
            residual.save(final=True)
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type': 'fixed_scene_reference_residual_sac',
                'residual_contract': residual.contract}, indent=2)+'\n')
        report = dict(policy=controller_name,
                      steps=len(history), outcomes=counts, frames=frames, history=history,
                      initial_settling_steps=settling_steps, sim_device=str(env.device),
                      interrupted=stopped['value'], completed_attempt=bool(sum(counts.values())),
                      video=str(output/video_name) if renderer else None,
                      body_envelope_diagnostic=args.body_envelope,
                      actor_reference_mix=args.actor_reference_mix, correction_label_rows=len(label_actions),
                      diagnostic_joint_offset_rad=args.joint_offset_rad,
                      initial_actor_error=initial_actor_error,
                      checkpoint_actor_updates=state.get('actor_updates') if state else None,
                      checkpoint_actor_refit=state.get('diagnostic_actor_refit') if state else None)
        if residual:
            report['residual_sac'] = residual.report()
        (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        if residual:
            (output/'status.json').write_text(json.dumps({'status':'stopped' if stopped['value'] else 'complete',
                'outcomes':counts,'actor_updates':residual.actor_updates})+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'history'}), flush=True)
    except BaseException as error:
        # Kit shutdown can replace Python's nonzero exit and suppress the
        # uncaught traceback. Preserve the failure before closing the app.
        import traceback
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'failure.json').write_text(json.dumps(
            {'phase': 'failed', 'error': type(error).__name__, 'reason': str(error)})+'\n')
        if args.residual_sac:
            (args.output_dir/'status.json').write_text(json.dumps({'status':'failed'})+'\n')
        raise
    finally:
        if writer is not None:
            writer.release()
        if recorder is not None:
            recorder.close()
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
