#!/usr/bin/env python3
"""Measure a VR reference in current v2 physics; this is not a SAC evaluation."""

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
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    add_base_drive_cli_args(parser)
    parser.set_defaults(headless=True, robot_model='s63', gripper='leju-twofinger', rack_rollers=True)
    args = parser.parse_args()
    if args.capture_every < 1 or args.episode_index < 0:
        parser.error('Capture interval must be positive and episode index nonnegative')
    if args.robot_model != 's63' or args.gripper != 'leju-twofinger' or not args.rack_rollers:
        parser.error('This reference replay requires S63/Leju and rack rollers')
    if args.output_dir.exists():
        parser.error('Use a new, unique output directory')
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)
    export_base_drive_cli(args)
    stopped = {'value': False}
    signal.signal(signal.SIGTERM, lambda *_: stopped.update(value=True))
    app = AppLauncher(args).app
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
        teacher = VRJointTracker(env, demo, rack)
        projection = GraspActionProjector(list(actions.items()))
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        meta = dict(task_family='multi_box_v2', skill='grasp', robot_model='s63',
            gripper='leju-twofinger', rack_rollers=True, controller_mapping='scaled',
            action_dim=sum(actions.values()), actor_obs_dim=dims['policy'][0],
            critic_obs_dim=dims['policy'][0]+dims['critic'][0], action_terms=list(map(list, actions.items())),
            control_dt=env.step_dt, multi_box=asdict(cfg.multi_box), episode_seconds=30.,
            collection_source='current_v2_environment_executed_vr_reference',
            sim_device=str(env.device), current_reward_verified_against_breakdown=True,
            training_contract=contract, initial_poses='inferred_from_original_demo_then_physics_settled',
            old_demo_rewards_used=False, old_policy_executed=False,
            source_dataset=str(args.demo_dataset.resolve()), source_episode_index=args.episode_index,
            reset_kinematics='coherent_gpu_articulation_fk_v2')
        recorder = RlTransitionRecorder(output/'executed_transitions.hdf5', meta)
        recorder.start_episode(initial_state=capture_rl_initial_state(env, observation))
        renderer = None if args.no_video else SceneVideo(env,
            caption=f'VR reference + live IK | NOT SAC | actual {env.device} PhysX')
        if renderer:
            writer = cv2.VideoWriter(str(output/'reference.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                    30/args.capture_every, (960, 720))
            if not writer.isOpened():
                raise ValueError('Could not open reference video writer')
        history, pixels, counts = [], {}, dict(success=0, unsafe=0, invalid_reset=0, time_out=0)
        compute = env.termination_manager.compute

        def before_reset():
            result = compute()
            grasp = env._multi_box_privileged_grasp_step
            row = dict(step=len(history)+1, pinching=grasp.pinch.hand_pinching[0].tolist(),
                flap_distances=grasp.matched_flap_distance_m[0].tolist(),
                box_pose=grasp.box_pose_world[0].tolist(), phase=int(teacher.phase[0]),
                ik_position_errors=[float(s.target_position_error()[0]) for s in teacher.solvers],
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
        with torch.no_grad():
            for step in range(900):
                if stopped['value']:
                    break
                pre = {key: value.clone() for key, value in observation.items()}
                action = projection(pre['policy'], teacher.act(pre['policy']))
                observation, reward, terminated, truncated, info = env.step(action)
                terminal = info['transition_next_observations']
                if bool(info['transition_numerical_failure'].any()):
                    raise ValueError('Numerical recovery during replay; attempt cannot enter Q')
                if not torch.allclose(reward, env._multi_box_grasp_reward_breakdown.total, atol=1e-5, rtol=1e-5):
                    raise ValueError('Current executed reward differs from reward breakdown')
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
                    if frames == 0 or row['success']:
                        cv2.imwrite(str(output/'preview.png'), frame)
                    frames += 1
                if step % 30 == 0:
                    (output/'progress.json').write_text(json.dumps(row))
                    print(f"[REFERENCE] step={step+1} pinch={row['pinching']} success={row['success']}", flush=True)
                if bool((terminated | truncated)[0]):
                    recorder.finish_episode(success=row['success'],
                        reason='success' if row['success'] else 'failure')
                    break
        report = dict(policy='VR_reference_plus_contact_confirmed_IK_NOT_SAC',
                      steps=len(history), outcomes=counts, frames=frames, history=history,
                      initial_settling_steps=settling_steps, sim_device=str(env.device))
        (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'history'}), flush=True)
    except BaseException as error:
        # Kit shutdown can replace Python's nonzero exit and suppress the
        # uncaught traceback. Preserve the failure before closing the app.
        import traceback
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'failure.json').write_text(json.dumps(
            {'phase': 'failed', 'error': type(error).__name__, 'reason': str(error)})+'\n')
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
