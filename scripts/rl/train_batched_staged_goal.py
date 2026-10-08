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
import sys
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
    parser.add_argument('--measured-train-credit',
                        choices=('one-step', 'measured-nstep16', 'measured-nstep16-terminal25', 'measured-episode-return'), default=None,
        help='Opt-in actual-flap learner objective: real completed successful and failed TRAIN n-step credit')
    parser.add_argument('--critic-episode-clock', choices=('task-remaining',), default=None,
        help='Measured task time remaining for the critic; requires fresh matching inputs')
    parser.add_argument('--policy-servo-diagnostics', action='store_true',
        help='Read-only measured held states: report production step clipping by region/box type every30steps')
    parser.add_argument('--jaw-behavior', choices=('policy', 'joint-epsilon10', 'joint-epsilon30'), default=None,
        help='Actual-flap TRAIN collection only: 10 or 30 percent correlated uniform joint jaws with unchanged production gate')
    parser.add_argument('--body-behavior', choices=('off', 'ramped-arm-bias20', 'arm20-explore-rest-greedy'), default=None,
        help='Actual-flap TRAIN only:20% coherent arm exploration; opt-in greedy current policy for remaining episodes')
    parser.add_argument('--body-saturation-penalty', choices=('off', 'mean3-soft'), default=None,
        help='Actual-flap TRAIN actor loss only: soft recovery of body means beyond abs3; no clipping')
    parser.add_argument('--jaw-saturation-penalty', choices=('off', 'logit4-soft', 'logit4-soft-strong'), default=None,
        help='Opt-in actual-flap TRAIN actor loss for saturated near-jaw logits; no clipping or prescribed jaws')
    parser.add_argument('--success-jaw-balance', choices=('off', 'region-hand-class'), default=None,
        help='Balance present regions/hands/open-close classes of actual successful TRAIN jaw NLL only')
    parser.add_argument('--stop-on-validation-regression',action='store_true')
    parser.add_argument('--minimum-validation-region-success-rate',type=float,default=0.)
    parser.add_argument('--validation-regression-significance',type=float,default=0.,
        help='0=strict regional count; otherwise exact paired tests with alpha spending over looks/regions')
    probes=parser.add_mutually_exclusive_group()
    probes.add_argument('--contact-stability-probe',action='store_true',
        help='Frozen-only finer physics integration/box solver diagnostic; cannot seed or train Q')
    probes.add_argument('--tgs-zero-velocity-probe',action='store_true',
        help='Frozen-only TGS velocity iteration diagnostic; does not alter control dt or position iterations')
    probes.add_argument('--contact-last-probe',action='store_true',
        help='Frozen-only articulation contact solver ordering diagnostic; never contributes matching Q replay')
    probes.add_argument('--pgs-probe',action='store_true',
        help='Frozen-only PGS/TGS solver comparison; preserves timestep, iterations and safety')
    probes.add_argument('--gripper-drive-probe',action='store_true',
        help='Frozen-only original/soft_2nm motor-drive comparison; never supplies matching Q replay')
    probes.add_argument('--reset-solver-probe',choices=('PGS','TGS'),default=None,
        help='Frozen DEV reset --steps 1 only: change just the physics solver; no TRAIN/FINAL or matching Q replay')
    probes.add_argument('--passive-bearing-probe-layer',type=Path,default=None,
        help='Frozen DEV reset --steps 1 only: select a separately verified canonical bearing drive overlay; no Q import')
    probes.add_argument('--reset-independent-scene-probe',action='store_true',
        help='Frozen DEV/steps1 with matched world/passive state only: disable scene physics replication; retain collision filtering')
    parser.add_argument('--centered-world-probe',action='store_true',
        help='Frozen-only shared origins with GPU environment collision IDs; no Q/replay training')
    parser.add_argument('--packed-background-probe',action='store_true',
        help='Frozen-only original/packed/original reset comparison; target/base randomization unchanged')
    parser.add_argument('--base-waypoint-probe',action='store_true',
        help='Frozen-only per-case workplace candidates; never contributes matching Q replay')
    parser.add_argument('--unmeasured-size-workplace-probe',action='store_true',
        help='Frozen TRAIN workplace measurement of supported new sizes; never relabels a measured waypoint or imports Q rows')
    parser.add_argument('--workplace-reset-diagnostics',action='store_true',
        help='Read-only reset trace for the explicit frozen TRAIN workplace search; existing contact reporters only')
    parser.add_argument('--base-substep-trace-env-indices',type=int,nargs='+',default=None,
        help='Read-only state/wrench trace for up to8 environments in a frozen TRAIN workplace search')
    parser.add_argument('--base-attitude-gain-probe',choices=('soft15_2',),default=None,
        help='Explicit frozen TRAIN128 comparison: lower only dynamic-base tilt gains; no Q import or learning')
    parser.add_argument('--steps',type=int,default=900)
    parser.add_argument('--reset-failure-diagnostics',action='store_true',
        help='Frozen DEV --steps 1: trace original box/link velocities and existing normal contacts before partial respawn')
    parser.add_argument('--frozen-physics-backend-eval',action='store_true',
        help='Explicit frozen original DEV128/900steps CPU or GPU policy comparison; never supplies matching Q replay')
    parser.add_argument('--cpu-physics-training', action='store_true',
        help='Separate CPU PhysX PGS MDP: fresh matching Q/replay, actual TRAIN and cuda:0 learner')
    parser.add_argument('--cpu-workplace-probe', action='store_true',
        help='Frozen CPU-only search of eight base waypoints on16 fresh TRAIN cases; never supplies matching Q replay')
    parser.add_argument('--learner-device', choices=('cpu','cuda:0'), default=None)
    parser.add_argument('--eval-video-env-indices', type=int, nargs='+', default=None,
        help='Record up to6 actual DEV environments before reset; complete original distribution remains evaluated')
    parser.add_argument('--grasp-observation-audit',action='store_true',
        help='Frozen DEV1..16 cases: compare actual flap/jaw/contact geometry throughout grasp; inputs and physics unchanged')
    parser.add_argument('--full-distribution-grasp-observation-audit',action='store_true',
        help='With frozen grasp audit: all128 DEV cases parallel or serial,32 per region; never imports Q rows')
    parser.add_argument('--reset-world-frame-probe',type=Path,default=None,
        help='Frozen DEV reset only: original measured world origins/rack/support poses; passive DOF history retained')
    parser.add_argument('--zero-passive-roller-velocities-probe',action='store_true',
        help='Frozen reset diagnostic only: preserve roller angles/poses but remove inherited angular velocities')
    parser.add_argument('--rear5-support-gap-probe-m',type=float,default=None,
        help='Frozen reset diagnostic only: keep dynamic background5, change its initial support gap (default0.008m)')
    parser.add_argument('--reset-contact-pair-diagnostics',action='store_true',
        help='Frozen reset diagnostic only: extend existing box reporters to rack/rollers/other box bodies and measure support motion')
    parser.add_argument('--reset-flap-contact-pair-diagnostics',action='store_true',
        help='Frozen reset pair audit only: add one-source reporters for all four flaps; collision rules unchanged')
    parser.add_argument('--reset-flap-contact-physical-pools',type=int,nargs='+',default=None,
        help='Optional focused source pools for the frozen flap audit; other-box target filters remain exhaustive')
    add_robot_model_cli_args(parser);add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser);add_base_drive_cli_args(parser)
    parser.set_defaults(headless=True,robot_model='s63',gripper='leju-twofinger',rack_rollers=True)
    args=parser.parse_args()
    if not 1<=args.steps<=900 or args.output_dir.exists():parser.error('New output and 1..900 steps required')
    if (args.contact_stability_probe or args.tgs_zero_velocity_probe or args.contact_last_probe or args.pgs_probe or args.gripper_drive_probe
            or args.centered_world_probe or args.packed_background_probe) and args.training:
        parser.error('Contact stability probe changes solver dynamics and is frozen-only')
    waves=json.loads(args.waves_json.read_text())
    learner_device=args.learner_device or args.device or 'cuda:0'
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import (
        INCOMPATIBLE_FLAGS as WORKPLACE_FLAGS, validate_cpu_workplace_probe)
    try:
        if args.unmeasured_size_workplace_probe and not args.cpu_workplace_probe:
            raise ValueError('Unmeasured size candidates require the explicit frozen workplace route')
        workplace_eval=validate_cpu_workplace_probe(waves,json.loads(args.training_manifest.read_text()),
            enabled=args.cpu_workplace_probe,device=args.device or 'cuda:0',training=args.training,
            steps=args.steps,waypoint_enabled=args.base_waypoint_probe,
            unmeasured_size_probe=args.unmeasured_size_workplace_probe,
            workplace_reset_diagnostics=args.workplace_reset_diagnostics,
            explicit_frozen='--no-training' in sys.argv and '--training' not in sys.argv,
            other_probe=any(s.split('=')[0] in WORKPLACE_FLAGS for s in sys.argv[1:]))
    except (ValueError,OSError) as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.base_attitude_probe import validate_base_attitude_probe
    try:
        base_attitude_probe=validate_base_attitude_probe(args.base_attitude_gain_probe,
            workplace=workplace_eval,training=args.training,num_envs=len(waves[0]['layouts']))
    except ValueError as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.debug.base_substep_trace import validate_base_substep_trace
    try:
        base_trace_contract=validate_base_substep_trace(args.base_substep_trace_env_indices,
            workplace=workplace_eval,training=args.training,num_envs=len(waves[0]['layouts']))
    except ValueError as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_physics_training import (
        INCOMPATIBLE_FLAGS as CPU_TRAIN_PROBE_FLAGS, SOURCE as CPU_TRAIN_SOURCE,
        validate_cpu_physics_training, act_measured_held_rows)
    try:
        cpu_training=validate_cpu_physics_training(waves,json.loads(args.training_manifest.read_text()),
            enabled=args.cpu_physics_training,physics_device=args.device or 'cuda:0',
            learner_device=learner_device,training=args.training,steps=args.steps,
            other_probe=any(s.split('=')[0] in CPU_TRAIN_PROBE_FLAGS for s in sys.argv[1:]))
        if cpu_training is None and learner_device != (args.device or 'cuda:0'):
            raise ValueError('Separate learner device requires explicit CPU physics learning')
    except (ValueError,OSError) as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import (
        SOURCE as BACKEND_EVAL_SOURCE,validate_frozen_backend_policy_eval)
    try:
        backend_eval=validate_frozen_backend_policy_eval(waves,enabled=args.frozen_physics_backend_eval,
            device=args.device or 'cuda:0',training=args.training,steps=args.steps,
            explicit_frozen='--no-training' in sys.argv and '--training' not in sys.argv,
            other_probe=any((args.contact_stability_probe,args.tgs_zero_velocity_probe,args.contact_last_probe,
                args.pgs_probe,args.gripper_drive_probe,args.centered_world_probe,args.packed_background_probe,
                args.base_waypoint_probe,args.reset_solver_probe,args.passive_bearing_probe_layer,
                args.reset_independent_scene_probe,args.zero_passive_roller_velocities_probe,
                args.rear5_support_gap_probe_m is not None,args.grasp_observation_audit,
                args.measured_train_credit,args.jaw_behavior,args.body_behavior,args.body_saturation_penalty,
                args.jaw_saturation_penalty,args.success_jaw_balance)))
        if args.device=='cpu' and backend_eval is None and cpu_training is None and workplace_eval is None and (args.training or '--no-training' not in sys.argv
                or '--training' in sys.argv or not args.reset_failure_diagnostics or args.steps!=1):
            raise ValueError('CPU requires explicit frozen reset or full backend policy evaluation')
    except ValueError as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.debug.grasp_observation_audit import validate_grasp_observation_audit
    try:validate_grasp_observation_audit(waves,enabled=args.grasp_observation_audit,
        full_distribution=args.full_distribution_grasp_observation_audit,
        training=args.training,steps=args.steps,other_probe=any((args.reset_failure_diagnostics,
            args.contact_stability_probe,args.tgs_zero_velocity_probe,args.contact_last_probe,args.pgs_probe,
            args.gripper_drive_probe,args.centered_world_probe,args.packed_background_probe,args.base_waypoint_probe,
            args.reset_solver_probe,args.passive_bearing_probe_layer,args.reset_independent_scene_probe,
            args.reset_world_frame_probe,args.zero_passive_roller_velocities_probe,args.rear5_support_gap_probe_m is not None)))
    except ValueError as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import (
        validate_reset_diagnostic_request,validate_rear5_support_gap_diagnostic_request)
    try:validate_reset_diagnostic_request(waves,enabled=args.reset_failure_diagnostics,
        training=args.training,steps=args.steps,
        frozen_backend_evaluation=backend_eval is not None,physics_device=args.device)
    except ValueError as error:parser.error(str(error))
    world_frame_probe=None
    if args.reset_world_frame_probe:
        from kuavo_isaaclab_scene.rl.multi_box.scene.reset_world_frame import validate_reset_world_frame_request
        try:
            world_frame_probe=json.loads(args.reset_world_frame_probe.read_text())
            validate_reset_world_frame_request(waves,world_frame_probe,reset_enabled=args.reset_failure_diagnostics,
                training=args.training,steps=args.steps,
                frozen_backend_evaluation=backend_eval is not None,physics_device=args.device,
                other_probe=any((args.contact_stability_probe,
                    args.tgs_zero_velocity_probe,args.contact_last_probe,args.pgs_probe,args.gripper_drive_probe,
                    args.centered_world_probe,args.packed_background_probe,args.base_waypoint_probe,
                    args.reset_solver_probe,args.passive_bearing_probe_layer,args.zero_passive_roller_velocities_probe,
                    args.rear5_support_gap_probe_m is not None)))
        except (OSError,ValueError) as error:parser.error(str(error))
    from kuavo_isaaclab_scene.rl.multi_box.scene.reset_replication import validate_independent_scene_request
    try:validate_independent_scene_request(waves,world_frame_probe,
        enabled=args.reset_independent_scene_probe,reset_enabled=args.reset_failure_diagnostics,
        training=args.training,steps=args.steps,other_probe=any((args.centered_world_probe,
            args.packed_background_probe,args.base_waypoint_probe,args.zero_passive_roller_velocities_probe,
            args.rear5_support_gap_probe_m is not None)))
    except ValueError as error:parser.error(str(error))
    if args.reset_contact_pair_diagnostics and not args.reset_failure_diagnostics:
        parser.error('Contact pair diagnostics require frozen reset diagnostics')
    if args.reset_flap_contact_pair_diagnostics and not args.reset_contact_pair_diagnostics:
        parser.error('Flap pair diagnostics require frozen reset and contact pair diagnostics')
    if args.reset_flap_contact_physical_pools is not None and not args.reset_flap_contact_pair_diagnostics:
        parser.error('Flap source-pool selection requires frozen flap contact diagnostics')
    if args.reset_solver_probe and not args.reset_failure_diagnostics:
        parser.error('Reset solver probe requires frozen reset diagnostics')
    if args.passive_bearing_probe_layer and (not args.reset_failure_diagnostics
            or not args.passive_bearing_probe_layer.is_file() or args.zero_passive_roller_velocities_probe
            or args.rear5_support_gap_probe_m is not None or args.centered_world_probe
            or args.packed_background_probe or args.base_waypoint_probe):
        parser.error('Bearing layer requires an existing overlay and otherwise unchanged frozen reset diagnostics')
    try:validate_rear5_support_gap_diagnostic_request(waves,gap_m=args.rear5_support_gap_probe_m,
        reset_enabled=args.reset_failure_diagnostics,training=args.training,steps=args.steps,
        other_probe=any((args.zero_passive_roller_velocities_probe,args.contact_stability_probe,
            args.tgs_zero_velocity_probe,args.contact_last_probe,args.pgs_probe,
            args.gripper_drive_probe,args.reset_solver_probe,args.passive_bearing_probe_layer,args.centered_world_probe,args.packed_background_probe,args.base_waypoint_probe)))
    except ValueError as error:parser.error(str(error))
    if args.zero_passive_roller_velocities_probe and (not args.reset_failure_diagnostics or
            args.contact_stability_probe or args.tgs_zero_velocity_probe or args.contact_last_probe or
            args.pgs_probe or args.gripper_drive_probe or args.reset_solver_probe or args.passive_bearing_probe_layer or args.centered_world_probe or
            args.packed_background_probe or args.base_waypoint_probe):
        parser.error('Passive roller velocity probe requires an otherwise unchanged frozen reset diagnostic')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.gripper_drive_probe import validate_gripper_drive_probe
    try:validate_gripper_drive_probe(waves,enabled=args.gripper_drive_probe,training=args.training)
    except ValueError as error:parser.error(str(error))
    if args.gripper_drive_probe and (args.centered_world_probe or args.packed_background_probe or args.base_waypoint_probe):
        parser.error('Compare gripper drives with the original physics origins, background and waypoints')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.waypoint_probe import validate_waypoint_probe
    try:validate_waypoint_probe(waves,enabled=args.base_waypoint_probe,training=args.training)
    except ValueError as error:parser.error(str(error))
    n=len(waves[0]['layouts']) if waves else 0
    if not 1<=n<=128 or any(len(w['layouts'])!=n for w in waves):
        parser.error('Each wave must contain the same 1..128 independent layouts')
    for w in waves:
        if w.get('background_placement','original') not in ('original','packed') or (
                w.get('background_placement','original')!='original' and not args.packed_background_probe):
            parser.error('Packed background reset is an explicitly frozen diagnostic')
        if w['split'] not in ('train','validation','holdout'):parser.error('Unknown wave split')
        if w['split']=='train' and not args.training and workplace_eval is None:
            parser.error('Frozen runs cannot contain TRAIN waves outside the explicit CPU workplace search')
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
    if not 0<=args.validation_regression_significance<1 or (
            args.validation_regression_significance and not args.stop_on_validation_regression):
        parser.error('Regression significance within0..1, excluding1, requires the regression guard')
    if not args.base_waypoint_probe and any(len({r['layout']['seed'] for r in w['layouts']})!=n for w in waves):
        parser.error('Each parallel wave needs distinct initial cases')
    if args.robot_model!='s63' or args.gripper!='leju-twofinger' or not args.rack_rollers:
        parser.error('Current held-base checkpoint requires S63/Leju/rack rollers')
    from selected_scene_videos import validate_video_selection
    try:validate_video_selection(args.eval_video_env_indices,n,args.steps)
    except ValueError as error:parser.error(str(error))
    export_robot_model_cli(args);export_gripper_cli(args);export_rack_roller_cli(args);export_base_drive_cli(args)
    app=AppLauncher(args).app
    stopped={'value':False};signal.signal(signal.SIGTERM,lambda *_:stopped.update(value=True))
    env=recorder=pilot=grasp_audit=base_substep_trace=None;output=args.output_dir.resolve()
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
        from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import GraspLayout,layout_reset_observation,layout_generation_contract
        from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
        from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages,settle_batched_layouts,DevelopmentSuccessGuard,WAVE_RESET_CONTROLLER_CONTRACT,evaluate_development_wave,measured_wave_mask,observe_measured_held_rows
        from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import GraspActionProjector
        from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
        from kuavo_isaaclab_scene.rl.multi_box.debug.contact_sensors import V2_RACK_SENSOR_NAMES,V2_COLLISION_BODY_NAMES
        from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import filtered_force_by_body
        from kuavo_isaaclab_scene.workcell.rack_rollers import resolve_rack_roller_settings
        from kuavo_isaaclab_scene.recording.rl_initial_state import capture_rl_initial_state
        from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder
        contract=json.loads(args.training_manifest.read_text())
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import require_current_lift_contract,frozen_prior_lift_contract
        require_current_lift_contract(contract)
        from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
            configured_reward_weights,frozen_actor_reward_contract,learning_termination_mask)
        reward_weights=configured_reward_weights(contract['reward_profile'])
        state=torch.load(args.checkpoint,map_location=learner_device,weights_only=True)
        from kuavo_isaaclab_scene.rl.multi_box.observations.task_timing import (
            resolve_critic_episode_clock, TASK_TIMING_GROUP)
        critic_episode_clock=resolve_critic_episode_clock(state,args.critic_episode_clock)
        cfg=MultiBoxGraspAssemblyEnvCfg(num_envs=n);cfg.episode_length_s=30.
        from kuavo_isaaclab_scene.rl.multi_box.experiments.base_attitude_probe import (
            configure_base_attitude_probe, verify_base_attitude_probe)
        configure_base_attitude_probe(cfg,base_attitude_probe)
        if critic_episode_clock is not None:
            from kuavo_isaaclab_scene.rl.multi_box.managers.v2_observations import TaskTimeRemainingCfg
            setattr(cfg.observations,TASK_TIMING_GROUP,TaskTimeRemainingCfg())
        from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import (
            SUPPLEMENTAL_GROUP,SUPPLEMENTAL_DIM,supplemental_perception_contract)
        supplemental=contract.get('supplemental_perception')
        if supplemental is not None:
            if supplemental!=supplemental_perception_contract():raise ValueError('Unknown supplemental perception contract')
            from kuavo_isaaclab_scene.rl.multi_box.managers.v2_observations import ActualFlapRelationsCfg
            setattr(cfg.observations,SUPPLEMENTAL_GROUP,ActualFlapRelationsCfg())
        from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import configure_flap_dynamics
        configure_flap_dynamics(cfg,contract)
        if 'contact_shaping' in contract['reward_profile']:
            cfg.rewards.grasp.params=dict(reward_profile=contract['reward_profile'])
        from kuavo_isaaclab_scene.rl.multi_box.scene.reset_replication import (
            configure_independent_scene_probe,verify_independent_scene_probe)
        replication_probe=configure_independent_scene_probe(cfg,enabled=args.reset_independent_scene_probe)
        pair_filter_manifest=None;flap_contact_reporters=[]
        if args.reset_contact_pair_diagnostics:
            from kuavo_isaaclab_scene.rl.multi_box.debug.contact_sensors import BELT_CONTACT_SENSOR_NAMES,_rack_contact_targets
            from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
            from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry
            from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import extend_startup_contact_pair_filters
            targets=_rack_contact_targets(cfg.scene)
            flaps=('flap_front','flap_back','flap_right','flap_left')
            box_paths=[]
            for name in physical_asset_names():
                asset=getattr(cfg.scene,name);geometry=box_geometry(asset,flaps)
                paths=[geometry.body_path,*[geometry.flaps[f].body_path for f in flaps]]
                targets.extend(asset.prim_path+('/'+p if p!='.' else '') for p in paths)
                box_paths.append(dict(asset_name=name,
                    body=asset.prim_path+'/'+geometry.body_path,
                    flaps={f:asset.prim_path+'/'+geometry.flaps[f].body_path for f in flaps}))
            pair_filter_manifest=extend_startup_contact_pair_filters(cfg.scene,BELT_CONTACT_SENSOR_NAMES,targets)
            if args.reset_flap_contact_pair_diagnostics:
                from kuavo_isaaclab_scene.rl.multi_box.scene.reset_flap_contacts import add_startup_flap_contact_reporters
                flap_contact_reporters=add_startup_flap_contact_reporters(cfg.scene,BELT_CONTACT_SENSOR_NAMES,box_paths,
                    physical_pools=args.reset_flap_contact_physical_pools)
        if args.centered_world_probe:
            if not str(args.device).startswith('cuda') or not cfg.scene.replicate_physics or not cfg.scene.filter_collisions:
                raise ValueError('Shared-origin probe requires replicated GPU physics with environment collision IDs')
            cfg.scene.env_spacing=0.
        cfg.multi_box=replace(cfg.multi_box,self_collision_enabled=contract['self_collision']['enabled'])
        cfg.sim.device=args.device or 'cuda:0'
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import configure_staged_physics
        configure_staged_physics(cfg,contract)
        from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import configure_reset_solver_probe
        reset_solver_probe=configure_reset_solver_probe(cfg,waves,solver=args.reset_solver_probe,
            reset_enabled=args.reset_failure_diagnostics,training=args.training,steps=args.steps,
            other_probe=any((args.zero_passive_roller_velocities_probe,args.rear5_support_gap_probe_m is not None,
                args.centered_world_probe,args.packed_background_probe,args.base_waypoint_probe)))
        if contract.get('physics_dynamics') and (args.contact_stability_probe or args.tgs_zero_velocity_probe
                                               or args.contact_last_probe or args.pgs_probe):
            raise ValueError('Frozen dynamics probes require the original TGS source contract')
        solver_probe=reset_solver_probe
        if args.passive_bearing_probe_layer:
            import hashlib
            cfg.scene.rack_assembly.spawn.usd_path=str(args.passive_bearing_probe_layer.resolve())
            solver_probe=dict(name='canonical_passive_bearing_drive_overlay',frozen_only=True,
                Q_import_eligible=False,source_layer_sha256=hashlib.sha256(args.passive_bearing_probe_layer.read_bytes()).hexdigest(),
                stiffness_nm_per_rad=0.,damping_nm_s_per_rad=resolve_rack_roller_settings().angular_damping,
                max_force_nm=.05,all_geometry_masses_friction_anchors_randomization_success_safety_unchanged=True)
        if args.gripper_drive_probe:
            solver_probe=dict(frozen_only=True,name='four_claw_motor_drive_comparison',
                Q_import_eligible=False,success_and_safety_unchanged=True,
                physics_dt_s=cfg.sim.dt,control_dt_s=cfg.sim.dt*cfg.decimation,
                solver_and_iterations_unchanged=True)
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
        if args.tgs_zero_velocity_probe:
            from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
            cfg.sim.physx.solver_type=1
            cfg.sim.physx.min_velocity_iteration_count=0
            cfg.sim.physx.max_velocity_iteration_count=0
            cfg.scene.robot.spawn.articulation_props.solver_velocity_iteration_count=0
            for name in physical_asset_names():
                getattr(cfg.scene,name).spawn.articulation_props.solver_velocity_iteration_count=0
            solver_probe=dict(frozen_only=True,name='TGS_zero_velocity_iterations',
                physics_dt_s=cfg.sim.dt,control_dt_s=cfg.sim.dt*cfg.decimation,
                scene_velocity_iteration_min=0,scene_velocity_iteration_max=0,
                robot_and_box_velocity_iterations=0,position_iterations_unchanged=True,
                success_and_safety_unchanged=True,Q_import_eligible=False)
        if args.contact_last_probe:
            cfg.sim.physx.solve_articulation_contact_last=True
            solver_probe=dict(frozen_only=True,name='articulation_contact_last',
                solve_articulation_contact_last=True,
                physics_dt_s=cfg.sim.dt,control_dt_s=cfg.sim.dt*cfg.decimation,
                all_iteration_counts_and_limits_unchanged=True,
                success_and_safety_unchanged=True,Q_import_eligible=False)
        if args.pgs_probe:
            cfg.sim.physx.solver_type=0
            solver_probe=dict(frozen_only=True,name='PGS_instead_of_TGS',
                physics_dt_s=cfg.sim.dt,control_dt_s=cfg.sim.dt*cfg.decimation,
                all_iteration_counts_and_limits_unchanged=True,
                success_and_safety_unchanged=True,Q_import_eligible=False)
        profile=dict(weights=asdict(reward_weights),approach_scale_m=GRASP_APPROACH_REWARD_SCALE_M,
            assignment_scale_m=GRASP_ASSIGNMENT_SCALE_M,capture_scale_m=GRASP_CAPTURE_REWARD_SCALE_M,
            front_stage_clearance_m=FRONT_STAGE_CLEARANCE_M,front_stage_lane_tolerance_m=FRONT_STAGE_LANE_TOLERANCE_M,
            front_stage_scale_m=FRONT_STAGE_REWARD_SCALE_M,geometry_profile='rack_front_lane_then_opposing_flap_reach_v3')
        if 'contact_shaping' in contract['reward_profile']:
            profile['contact_shaping']=contract['reward_profile']['contact_shaping']
        from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import configured_capture_geometry
        capture_geometry=configured_capture_geometry(contract['reward_profile'])
        profile['capture_scale_m']=capture_geometry['capture_scale_m']
        if 'precision_capture' in contract['reward_profile']:
            profile['precision_capture']=contract['reward_profile']['precision_capture']
        from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import configured_geometry_shaping
        geometry_shaping=configured_geometry_shaping(contract['reward_profile'])
        if geometry_shaping is not None:
            profile['absorbing_geometry']=geometry_shaping
        from kuavo_isaaclab_scene.rl.multi_box.rewards.success_value import configured_success_value
        success_value=configured_success_value(contract['reward_profile'])
        if success_value is not None:
            profile['success_value']=success_value
        from kuavo_isaaclab_scene.rl.multi_box.geometry.box_drop import (
            grasp_drop_safety_thresholds, configured_drop_limit, DROP_REFERENCE)
        drop_limit=configured_drop_limit(contract)
        if cfg.multi_box.max_box_drop_height != drop_limit:
            raise ValueError('Recorded grasp drop guard was not restored')
        thresholds=dict(**grasp_drop_safety_thresholds(cfg.multi_box),
            rack_contact_force_n=float(cfg.multi_box.rack_contact_force),
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
        if env.cfg.multi_box.max_box_drop_height != drop_limit:
            raise ValueError('Initialized environment changed the recorded grasp drop guard')
        if drop_limit is not None:
            print('[GRASP BOX DROP GUARD] '+json.dumps(dict(
                reference=DROP_REFERENCE,max_drop_height_m=drop_limit,
                actual_initialized_environment_verified=True)),flush=True)
        if geometry_shaping is not None:
            initialized_reward=env.reward_manager.get_term_cfg('grasp')
            if (initialized_reward.params.get('reward_profile') != contract['reward_profile']
                    or initialized_reward.func.model.geometry_shaping != geometry_shaping):
                raise ValueError('Initialized reward manager did not apply the absorbing geometry contract')
            print('[ABSORBING GEOMETRY REWARD] '+json.dumps(geometry_shaping),flush=True)
        if success_value is not None:
            initialized_reward=env.reward_manager.get_term_cfg('grasp')
            if (initialized_reward.params.get('reward_profile') != contract['reward_profile']
                    or initialized_reward.func.model.weights != reward_weights):
                raise ValueError('Initialized reward manager did not apply safe success64')
            print('[SAFE SUCCESS VALUE REWARD] '+json.dumps(success_value),flush=True)
        replication_probe=verify_independent_scene_probe(env,replication_probe)
        if replication_probe is not None:
            print('[FROZEN INDEPENDENT SCENE] '+json.dumps(replication_probe),flush=True)
        env._reset_contact_pair_diagnostics=args.reset_contact_pair_diagnostics
        if flap_contact_reporters:
            from kuavo_isaaclab_scene.rl.multi_box.scene.reset_flap_contacts import resolve_startup_flap_contact_paths
            flap_contact_reporters=resolve_startup_flap_contact_paths(flap_contact_reporters,env.scene.env_regex_ns)
        env._reset_flap_contact_reporters=flap_contact_reporters
        if args.passive_bearing_probe_layer:
            from kuavo_isaaclab_scene.rl.multi_box.scene.reset_diagnostics import verify_passive_bearing_drives
            solver_probe['initialized_passive_drives']=verify_passive_bearing_drives(
                env,resolve_rack_roller_settings().angular_damping)
            print('[INITIALIZED PASSIVE BEARING PROBE] '+json.dumps(solver_probe),flush=True)
        if args.centered_world_probe:
            if not torch.allclose(env.scene.env_origins,torch.zeros_like(env.scene.env_origins),atol=0,rtol=0):
                raise ValueError('Shared-origin probe did not apply zero world origins')
            print('[CENTERED WORLD PROBE] '+json.dumps(dict(frozen_only=True,environment_origins_zero=True,
                replicate_physics=cfg.scene.replicate_physics,filter_collisions=cfg.scene.filter_collisions,
                Q_import_eligible=False,success_and_safety_unchanged=True)),flush=True)
        if contract.get('physics_dynamics'):
            actual=env.sim.stage.GetPrimAtPath(cfg.sim.physics_prim_path).GetAttribute('physxScene:solverType').Get()
            expected=args.reset_solver_probe or contract['physics_dynamics']['solver']
            if actual!=expected:raise ValueError('Requested physics solver was not applied')
            if reset_solver_probe:
                reset_solver_probe['actual_USD_solver']=actual
                print('[FROZEN RESET SOLVER PROBE] '+json.dumps(reset_solver_probe),flush=True)
            else:
                print('[TRAINING PHYSICS CONTRACT] '+json.dumps(contract['physics_dynamics']|dict(actual_USD_solver=actual)),flush=True)
        if args.contact_last_probe:
            value=env.sim.stage.GetPrimAtPath(cfg.sim.physics_prim_path).GetAttribute(
                'physxScene:solveArticulationContactLast').Get()
            if value is not True:raise ValueError('Requested contact-last solver ordering was not applied')
            print('[FROZEN SOLVER PROBE] '+json.dumps(solver_probe|{'actual_USD_flag':value}),flush=True)
        if args.pgs_probe:
            value=env.sim.stage.GetPrimAtPath(cfg.sim.physics_prim_path).GetAttribute(
                'physxScene:solverType').Get()
            if value!='PGS':raise ValueError('Requested PGS solver was not applied')
            print('[FROZEN SOLVER PROBE] '+json.dumps(solver_probe|{'actual_USD_solver':value}),flush=True)
        dims={k:list(v) for k,v in env.observation_manager.group_obs_dim.items()}
        actions={k:env.action_manager.get_term(k).action_dim for k in env.action_manager.active_terms}
        expected_dims=contract['observations']|({SUPPLEMENTAL_GROUP:[SUPPLEMENTAL_DIM]} if supplemental else {})
        if critic_episode_clock is not None:
            expected_dims[TASK_TIMING_GROUP] = [1]
        if dims!=expected_dims or actions!=contract['actions']:
            raise ValueError('Batched physical observation/action widths differ')
        base=env.action_manager.get_term('base');upper=env.action_manager.get_term('upper_body')
        head=env.action_manager.get_term('head');height=env.action_manager.get_term('height')
        validate_goal_feedback_rates(base._scale,upper._scale,head._scale,height.cfg.speed_m_s,env.step_dt)
        batch,_=load_v2_grasp_demonstrations(args.demo_dataset,self_collision_enabled=cfg.multi_box.self_collision_enabled)
        sources={i:select_reference_episode(batch,i)['actor_obs'][0] for i in
                 {row['episode_index'] for w in waves for row in w['layouts']}}
        pilot_class=staged_policy_class(state.get('artifact_type'))
        if pilot_class is None:
            raise ValueError('Batched learner requires the separately initialized staged checkpoint')
        if bool(pilot_class.supplemental_observation_dim)!=bool(supplemental):
            raise ValueError('Staged policy and current measured supplemental perception differ')
        if args.measured_train_credit is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Measured TRAIN credit is explicitly for actual-flap correction training')
        if args.jaw_behavior is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Joint jaw behavior is explicitly for actual-flap TRAIN collection')
        if args.body_behavior is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Body behavior is explicitly for actual-flap TRAIN collection')
        if args.body_saturation_penalty is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Body saturation penalty is explicitly for actual-flap TRAIN actor updates')
        if args.jaw_saturation_penalty is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Jaw saturation penalty is explicitly for actual-flap TRAIN actor updates')
        if args.success_jaw_balance is not None:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
            if not issubclass(pilot_class, ActualFlapResidualSACPilot) or not args.training:
                raise ValueError('Successful jaw balance is explicitly for actual-flap TRAIN actor updates')
        warm=PoseGoalSACPilot(state['frozen_warm_start'],args.native_seed,
                            frozen_prior_lift_contract(frozen_actor_reward_contract(contract)),output,training=False,device=learner_device)
        if warm.coordinates.exact_projected_base:
            env.enable_projected_base_safety()
        templates=json.loads(args.waypoints.read_text())
        if templates['physical_action_contract']!=contract['action_contract']:raise ValueError('Waypoint travel differs')
        projection=GraspActionProjector(list(actions.items()))
        # Match the single-env reference lifecycle: the rack and root must
        # physically settle before their measured pose anchors new layouts.
        from kuavo_isaaclab_scene.rl.runners.train_asymmetric_sac import _settle_initial_resets
        initial,_=env.reset(seed=42)
        _settle_initial_resets(env,initial)
        world_frame_audit=None
        if world_frame_probe is not None:
            from kuavo_isaaclab_scene.rl.multi_box.scene.reset_world_frame import apply_startup_world_frame
            world_frame_audit=apply_startup_world_frame(env,world_frame_probe)
            print('[FROZEN ORIGINAL WORLD FRAME] '+json.dumps(world_frame_audit),flush=True)
        output.mkdir(parents=True,exist_ok=False)
        # Read initialized PhysX properties, rather than treating USD's unset
        # mass/density attributes as the runtime masses used by the solver.
        initialized_physics={}
        from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import physical_asset_names
        for name in ('robot',*physical_asset_names()):
            asset=env.scene[name];mass=asset.root_physx_view.get_masses()
            inertia=asset.root_physx_view.get_inertias()
            initialized_physics[name]=dict(body_names=asset.body_names,
                environment0_mass_kg=mass[0].tolist(),
                mass_kg_min=float(mass.min()),mass_kg_max=float(mass.max()),
                environment0_inertia_matrix=inertia[0].tolist(),
                joint_names=asset.joint_names,
                environment0_effort_limits=asset.data.joint_effort_limits[0].tolist(),
                environment0_stiffness=asset.data.joint_stiffness[0].tolist(),
                environment0_damping=asset.data.joint_damping[0].tolist(),
                environment0_armature=asset.data.joint_armature[0].tolist())
        print('[INITIALIZED PHYSICS] '+json.dumps({name:{k:value for k,value in data.items()
            if k in ('mass_kg_min','mass_kg_max')} for name,data in initialized_physics.items()}),flush=True)
        meta=dict(task_family='multi_box_v2',skill='grasp',robot_model='s63',gripper='leju-twofinger',
            rack_rollers=True,actor_obs_dim=464,critic_obs_dim=530,action_dim=24,
            action_terms=list(map(list,actions.items())),control_dt=env.step_dt,episode_seconds=30.,
            collection_source=(CPU_TRAIN_SOURCE if cpu_training else BACKEND_EVAL_SOURCE if backend_eval else 'grasp_observation_audit_NOT_matching_Q_replay'
                               if args.grasp_observation_audit else 'reset_failure_diagnostic_NOT_matching_Q_replay'
                               if args.reset_failure_diagnostics else 'changed_gripper_drive_frozen_probe_NOT_matching_Q_replay'
                               if args.gripper_drive_probe else 'changed_contact_solver_frozen_probe_NOT_matching_Q_replay'
                               if solver_probe else 'background_placement_frozen_probe_NOT_matching_Q_replay'
                               if args.packed_background_probe else 'base_waypoint_frozen_probe_NOT_matching_Q_replay'
                               if args.base_waypoint_probe else pilot_class.artifact_type),training_contract=contract,
            contact_stability_probe=solver_probe,
            startup_flap_contact_reporters=flap_contact_reporters,
            startup_world_frame_probe=world_frame_audit,
            startup_scene_replication_probe=replication_probe,
            sim_device=str(env.device),learner_device=learner_device,multi_box=asdict(cfg.multi_box),old_demo_rewards_used=False,
            frozen_physics_backend_evaluation=backend_eval,
            CPU_physics_training=cpu_training,
            CPU_workplace_probe=workplace_eval,
            current_reward_verified_against_breakdown=True,
            initial_poses='independent_neutral_layouts_from_original_demo_then_physics_settled',
            wave_reset_controller_contract=WAVE_RESET_CONTROLLER_CONTRACT,
            initialized_physics=initialized_physics,
            centered_world_probe=args.centered_world_probe,
            background_placement_probe=dict(enabled=args.packed_background_probe,
                frozen_only=args.packed_background_probe,Q_import_eligible=not args.packed_background_probe),
            base_waypoint_probe=dict(enabled=args.base_waypoint_probe,frozen_only=args.base_waypoint_probe,
                Q_import_eligible=not args.base_waypoint_probe),
            episode_layouts=[dict(wave=i,environment=j,**row) for i,w in enumerate(waves) for j,row in enumerate(w['layouts'])])
        if supplemental:
            meta.update(supplemental_actor_obs_dim=SUPPLEMENTAL_DIM,supplemental_perception=supplemental)
        if critic_episode_clock is not None:
            meta['critic_episode_clock'] = critic_episode_clock
        from kuavo_isaaclab_scene.rl.multi_box.geometry.projected_base import projected_base_safety_contract
        coordinate_safety = projected_base_safety_contract() if warm.coordinates.exact_projected_base else None
        meta['controller_coordinate_safety'] = coordinate_safety
        from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_behavior_exploration import jaw_behavior_config
        collection_jaw_behavior = (jaw_behavior_config(args.jaw_behavior)
            if args.jaw_behavior is not None else state.get('jaw_behavior'))
        meta['TRAIN_jaw_behavior'] = collection_jaw_behavior
        from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import body_behavior_config
        collection_body_behavior = (body_behavior_config(args.body_behavior)
            if args.body_behavior is not None else state.get('body_behavior'))
        meta['TRAIN_body_behavior'] = collection_body_behavior
        from kuavo_isaaclab_scene.rl.multi_box.experiments.body_saturation import body_saturation_config
        actor_body_regularization = (body_saturation_config(args.body_saturation_penalty)
            if args.body_saturation_penalty is not None else state.get('body_saturation'))
        meta['TRAIN_actor_body_regularization'] = actor_body_regularization
        from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_saturation import jaw_saturation_config
        actor_jaw_regularization = (jaw_saturation_config(args.jaw_saturation_penalty)
            if args.jaw_saturation_penalty is not None else state.get('jaw_saturation'))
        meta['TRAIN_actor_jaw_regularization'] = actor_jaw_regularization
        from kuavo_isaaclab_scene.rl.multi_box.experiments.success_jaw_balance import success_jaw_balance_config
        successful_jaw_balance = (success_jaw_balance_config(args.success_jaw_balance)
            if args.success_jaw_balance is not None else state.get('success_jaw_balance'))
        meta['TRAIN_successful_jaw_balance'] = successful_jaw_balance
        meta['frozen_evaluation_uses_learned_policy_without_behavior_mixture'] = True
        if args.reset_failure_diagnostics or args.workplace_reset_diagnostics:
            meta['reset_diagnostic_physics_device']=str(env.device)
        recorder=RlTransitionRecorder(output/'executed_transitions.hdf5',meta)
        base_attitude_probe=verify_base_attitude_probe(env,base_attitude_probe)
        if base_trace_contract is not None:
            from kuavo_isaaclab_scene.rl.multi_box.debug.base_substep_trace import BaseSubstepTrace
            base_substep_trace=BaseSubstepTrace(env,output,base_trace_contract)
        if args.grasp_observation_audit:
            from kuavo_isaaclab_scene.rl.multi_box.debug.grasp_observation_audit import GraspObservationAudit
            grasp_audit=GraspObservationAudit(env,output)
        from kuavo_isaaclab_scene.rl.multi_box.experiments.policy_manifest import checkpoint_manifest_fields
        policy_metadata=checkpoint_manifest_fields(contract,state,artifact_type=pilot_class.artifact_type)
        (output/'manifest.json').write_text(json.dumps(contract|policy_metadata|{'artifact_type':pilot_class.artifact_type,
            'training':args.training,'layout_waves':waves,'no_live_VR_or_IK':True,
            'layout_generation_contract':layout_generation_contract(),
            'frozen_physics_backend_evaluation':backend_eval,
            'CPU_physics_training':cpu_training,
            'CPU_workplace_probe':workplace_eval,
            'workplace_reset_diagnostics':(workplace_eval or {}).get('workplace_reset_diagnostics'),
            'base_substep_trace':base_trace_contract,
            'base_attitude_gain_probe':base_attitude_probe,
            'learner_device':learner_device,'sim_device':str(env.device),
            'TRAIN_jaw_behavior':collection_jaw_behavior,
            'TRAIN_body_behavior':collection_body_behavior,
            'TRAIN_actor_body_regularization':actor_body_regularization,
            'TRAIN_actor_jaw_regularization':actor_jaw_regularization,
            'TRAIN_successful_jaw_balance':successful_jaw_balance,
            'controller_coordinate_safety':coordinate_safety,
            'initialized_physics':initialized_physics,
            'startup_world_frame_probe':world_frame_audit,
            'startup_scene_replication_probe':replication_probe,
            'initial_layout_guard_storage':'whole_wave_initial_layout_guard_reference_v1',
            'grasp_observation_audit':dict(enabled=args.grasp_observation_audit,
                full_original_DEV_distribution=args.full_distribution_grasp_observation_audit,
                parallel_environments=n,requested_cases=sum(len(w['layouts']) for w in waves),
                frozen_only=True,Q_import_eligible=False,actor_input_and_physics_unchanged=True),
            'centered_world_probe':dict(enabled=args.centered_world_probe,frozen_only=args.centered_world_probe,
                Q_import_eligible=not args.centered_world_probe,environment_origins=env.scene.env_origins.tolist()),
            'base_waypoint_probe':dict(enabled=args.base_waypoint_probe,frozen_only=args.base_waypoint_probe,
                Q_import_eligible=not args.base_waypoint_probe,initial_robot_box_poses_unchanged=True),
            'wave_reset_controller_contract':WAVE_RESET_CONTROLLER_CONTRACT},indent=2)+'\n')
        if args.reset_failure_diagnostics:
            manifest=json.loads((output/'manifest.json').read_text())
            manifest['reset_failure_diagnostics']=dict(enabled=True,frozen_only=True,
                Q_import_eligible=False,first_failure_before_respawn=True,
                physics_device=str(env.device),
                physics_randomization_success_safety_unchanged=solver_probe is None and str(env.device).startswith('cuda'),
                reset_solver_probe=reset_solver_probe,
                passive_bearing_drive_probe=solver_probe if args.passive_bearing_probe_layer else None,
                initial_passive_roller_velocity_changed=args.zero_passive_roller_velocities_probe,
                zero_passive_roller_velocity_probe=args.zero_passive_roller_velocities_probe,
                rear5_initial_support_gap_probe_m=args.rear5_support_gap_probe_m,
                initial_box_pose_changed=args.rear5_support_gap_probe_m is not None,
                normal_contact_pair_filters=pair_filter_manifest,
                contact_pair_reporting_extended=args.reset_contact_pair_diagnostics,
                flap_contact_reporters=flap_contact_reporters,
                new_sensors_added=bool(flap_contact_reporters),
                additional_one_source_reporter_count=len(flap_contact_reporters))
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if args.packed_background_probe:
            manifest=json.loads((output/'manifest.json').read_text())
            manifest['background_placement_probe']=dict(enabled=True,frozen_only=True,
                Q_import_eligible=False,target_base_randomization_and_safety_unchanged=True)
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        if solver_probe:
            manifest=json.loads((output/'manifest.json').read_text())
            (output/'manifest.json').write_text(json.dumps(manifest|{'contact_stability_probe':solver_probe},indent=2)+'\n')
        (output/'env.yaml').write_text(json.dumps(asdict(cfg.multi_box),indent=2)+'\n')
        drive_probe=None
        if args.gripper_drive_probe:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.gripper_drive_probe import GripperDriveProbe
            drive_probe=GripperDriveProbe(env.scene['robot'])
        drive_audits=[]
        outcomes=[];pilot=None;start=time.monotonic();total_rows=0;completed_wave_count=0
        guard=DevelopmentSuccessGuard(args.minimum_validation_region_success_rate,
            regression_significance=args.validation_regression_significance) if args.stop_on_validation_regression else None
        development_checks=[]
        for wave_index,wave in enumerate(waves):
            if stopped['value']:break
            if grasp_audit is not None:grasp_audit.begin_wave(wave_index,wave['layouts'])
            if drive_probe is not None:
                drive_audits.append(dict(wave=wave_index,**drive_probe.apply(wave['gripper_drive_probe'])))
                (output/'gripper_drive_audit.json').write_text(json.dumps(drive_audits,indent=2)+'\n')
            actors=torch.stack([layout_reset_observation(sources[r['episode_index']],
                GraspLayout(**r['layout']).validate(),cfg.multi_box,
                roller_clearance_m=resolve_rack_roller_settings().box_clearance_m) for r in wave['layouts']]).to(env.device)
            if args.rear5_support_gap_probe_m is not None:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import rear5_support_gap_reset_observation
                actors=torch.stack([rear5_support_gap_reset_observation(actor,args.rear5_support_gap_probe_m) for actor in actors])
            if wave.get('background_placement','original')=='packed':
                from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import packed_background_reset_observation
                actors=torch.stack([packed_background_reset_observation(actor,cfg.multi_box,
                    roller_clearance_m=resolve_rack_roller_settings().box_clearance_m) for actor in actors])
            observation,settled,valid_layout,layout_guard=settle_batched_layouts(env,actors,allow_partial=True,
                capture_reset_diagnostics=args.reset_failure_diagnostics or args.workplace_reset_diagnostics,
                zero_passive_roller_velocity_probe=args.zero_passive_roller_velocities_probe)
            if contract.get('flap_dynamics'):
                from kuavo_isaaclab_scene.rl.multi_box.scene.flap_dynamics import current_flap_dynamics_audit
                (output/f'flap_dynamics_wave_{wave_index:04d}.json').write_text(json.dumps(
                    dict(wave=wave_index,split=wave['split'],
                         **current_flap_dynamics_audit(env,original_layout_valid=valid_layout)),indent=2)+'\n')
            if args.reset_failure_diagnostics or args.workplace_reset_diagnostics:
                if args.passive_bearing_probe_layer:
                    captured=layout_guard['reset_failure_diagnostics']
                    captured['passive_bearing_drive_probe']=solver_probe
                    captured['physics_parameters_unchanged']=False
                    captured['physical_state_unchanged']=False
                    captured['box_base_poses_randomization_physics_parameters_success_and_safety_unchanged']=False
                    captured['box_base_poses_randomization_success_and_safety_unchanged']=True
                if reset_solver_probe:
                    captured=layout_guard['reset_failure_diagnostics']
                    captured['reset_solver_probe']=reset_solver_probe
                    captured['physics_parameters_unchanged']=False
                    captured['physical_state_unchanged']=False
                    captured['box_base_poses_randomization_physics_parameters_success_and_safety_unchanged']=False
                    captured['box_base_poses_randomization_success_and_safety_unchanged']=True
                layout_guard['reset_failure_diagnostics']['rear5_initial_support_gap_probe_m']=args.rear5_support_gap_probe_m
                if args.rear5_support_gap_probe_m is not None:
                    layout_guard['reset_failure_diagnostics']['initial_box_pose_changed']=True
                    layout_guard['reset_failure_diagnostics']['physical_state_unchanged']=False
                    layout_guard['reset_failure_diagnostics']['box_base_poses_randomization_physics_parameters_success_and_safety_unchanged']=False
                captured=layout_guard['reset_failure_diagnostics']
                if args.workplace_reset_diagnostics:
                    captured['frozen_TRAIN_workplace_reset_capture']=workplace_eval['workplace_reset_diagnostics']
                    captured['not_an_independent_confirmation']=True
                elif str(env.device)=='cpu':
                    captured['reset_physics_device_diagnostic']=dict(name='frozen_DEV_reset_CPU_PhysX',
                        physics_device='cpu',Q_import_eligible=False,
                        constructor_and_contact_solver_history_not_matched=True,
                        not_a_grasp_performance_evaluation=backend_eval is None)
                    captured['physical_state_unchanged']=False
                    captured['box_base_poses_randomization_physics_parameters_success_and_safety_unchanged']=False
                    captured['requested_box_base_layouts_and_dynamics_parameters_unchanged']=True
                if replication_probe is not None:
                    captured['startup_scene_replication_probe']=replication_probe
                    captured['physical_state_unchanged']=False
                if world_frame_audit is not None:
                    captured['startup_world_frame_probe']=world_frame_audit
                    captured['initial_world_placement_changed']=world_frame_audit['world_root_placements_changed']
                    captured['initial_passive_joint_state_changed']=world_frame_audit['initial_passive_joint_state_changed']
                    captured['initial_rack_relative_requested_layout_unchanged']=True
                    captured['physical_state_unchanged']=False
                    captured['box_base_poses_randomization_physics_parameters_success_and_safety_unchanged']=False
                diagnostic_name=f'reset_failure_diagnostics_wave_{wave_index:04d}.json'
                (output/diagnostic_name).write_text(json.dumps(
                    dict(wave=wave_index,split=wave['split'],layouts=wave['layouts'],guard=layout_guard),indent=2)+'\n')
                # Complete link/physics traces live once in the closed audit
                # file, rather than being copied into all 128 outcome rows.
                layout_guard['reset_failure_diagnostics']=dict(file=diagnostic_name,
                    first_selected_failure_count=len(captured['first_invalid_before_respawn']),
                    neutral_hold_trace_steps=[x['physics_step'] for x in captured['neutral_hold_trace']],
                    normal_contact_trace_steps=[x['physics_step'] for x in captured['neutral_normal_contact_trace']],
                    complete_trace_in_diagnostic_file=True)
            from kuavo_isaaclab_scene.rl.multi_box.debug.layout_guard_storage import store_layout_guard
            layout_guard_references=store_layout_guard(output,wave_index,n,layout_guard)
            # An invalid requested case remains a failed attempt in the
            # denominator. Its replacement never supplies a snapshot/action
            # or transition to this layout's replay.
            stage_seed=observation['policy'].clone()
            stage_seed[~valid_layout]=actors[~valid_layout]
            stages=BatchedBaseStages(warm.coordinates,templates,stage_seed,
                unmeasured_size_probe=bool(workplace_eval and args.unmeasured_size_workplace_probe))
            from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import validate_requested_region_stages
            validate_requested_region_stages(stages.stages,wave['layouts'])
            if pilot is None:
                pilot_options = ({'measured_train_credit': args.measured_train_credit}
                    if args.measured_train_credit is not None else {})
                if args.jaw_behavior is not None:
                    pilot_options['jaw_behavior'] = args.jaw_behavior
                if args.body_behavior is not None:
                    pilot_options['body_behavior'] = args.body_behavior
                if args.critic_episode_clock is not None:
                    pilot_options['critic_episode_clock'] = args.critic_episode_clock
                if args.body_saturation_penalty is not None:
                    pilot_options['body_saturation'] = args.body_saturation_penalty
                if args.jaw_saturation_penalty is not None:
                    pilot_options['jaw_saturation'] = args.jaw_saturation_penalty
                if args.success_jaw_balance is not None:
                    pilot_options['success_jaw_balance'] = args.success_jaw_balance
                pilot=pilot_class(warm,contract,output,stages.stages[0],checkpoint=args.checkpoint,
                    training=args.training,device=learner_device, **pilot_options)
                (output/'agent.yaml').write_text(json.dumps(pilot.contract,indent=2)+'\n')
            if args.base_waypoint_probe:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.waypoint_probe import apply_waypoint_probe
                apply_waypoint_probe(stages,wave['layouts'])
            pilot.training=args.training and wave['split']=='train'
            pilot.reset_exploration(n)
            updates_before=(pilot.actor_updates,pilot.critic_updates,pilot.replay.size)
            if backend_eval or workplace_eval:
                frozen_eval_contract=backend_eval or workplace_eval
                frozen_eval_contract['initial_learner_counters']=dict(
                    actor_updates=pilot.actor_updates,critic_updates=pilot.critic_updates,
                    replay_size=pilot.replay.size,online_rows=pilot.online_rows)
            frozen_integrity=None
            if backend_eval or workplace_eval:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import (
                    frozen_network_snapshot,verify_frozen_network_snapshot)
                frozen_integrity=frozen_network_snapshot(pilot)
            active=valid_layout.clone()
            snapshots=[capture_rl_initial_state(env,observation,env_index=i) if valid_layout[i] else None for i in range(n)]
            rows=[[] for _ in range(n)]
            measured_goal_batches=[]
            last=[None if valid_layout[i] else dict(steps=0,success=False,unsafe=False,
                invalid_reset=True,time_out=False,pinching=[False,False],flap_distances=None,
                original_layout_replaced_during_settling=True,replay_rows=0) for i in range(n)]
            buffer={};scene_videos=None
            if args.eval_video_env_indices and (wave['split']=='validation' or workplace_eval):
                from selected_scene_videos import SelectedSceneVideos
                scene_videos=SelectedSceneVideos(env,output,wave_index,wave['layouts'],args.eval_video_env_indices,
                    actor_updates=pilot.actor_updates,critic_updates=pilot.critic_updates,
                    split=wave['split'],workplace_search=bool(workplace_eval))
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
                        'robot_rack_collision','self_collision','obstacle_collision','workspace_limit','base_projection_invalid',
                        'box_drop','box_lift_limit','box_speed_limit')},
                    box_pose=g.box_pose_world.clone(),box_velocity=g.box_velocity_world.clone(),
                    logical=g.target_logical_id.clone(),pool=g.target_pool_id.clone())
                if 'contact_shaping' in contract['reward_profile']:
                    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import opposing_pad_contact_quality
                    buffer['contact_quality']=opposing_pad_contact_quality(g.contacts)
                if grasp_audit is not None:grasp_audit.finish(g,s)
                if scene_videos is not None:
                    scene_videos.capture(step,active,buffer['success'],buffer['unsafe'],buffer['time_out'],buffer['distance'])
                return result
            env.termination_manager.compute=capture_before_reset
            rollout_start=time.monotonic();wave_rows=0;policy_servo_diagnostics=[]
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
                            clocks=stages.clocks(ids,step)
                            command,previous=act_measured_held_rows(pilot,stages,ids,pre,clocks,
                                supplemental_group=SUPPLEMENTAL_GROUP if supplemental else None)
                            action[ids]=command
                            if args.policy_servo_diagnostics and step%30==0:
                                from kuavo_isaaclab_scene.rl.multi_box.debug.servo_policy_diagnostics import measured_policy_servo_diagnostics
                                policy_servo_diagnostics.append(dict(wave=wave_index,split=wave['split'],step=step+1,
                                    statistics=measured_policy_servo_diagnostics(pilot.agent,previous,ids,wave['layouts'])))
                        if not torch.allclose(projection(pre['policy'],action),action,atol=1e-6,rtol=0):
                            raise ValueError('Generated and executed jaw projections differ')
                        if grasp_audit is not None:grasp_audit.prepare(step,pre['policy'],action,active,ids,previous)
                        if base_substep_trace is not None:
                            base_substep_trace.prepare(wave_index,step,active,stages.stages,action)
                        try:
                            observation,reward,terminated,truncated,info=env.step(action)
                        finally:
                            if base_substep_trace is not None:base_substep_trace.finish_step()
                        active=measured_wave_mask(active,info['transition_numerical_failure'],
                            info.get('transition_numerical_diagnostics',{}),last,step)
                        terminal=info['transition_next_observations']
                        if not torch.allclose(reward[active],env._multi_box_grasp_reward_breakdown.total[active],atol=1e-5,rtol=1e-5):
                            raise ValueError('Actual vector reward differs from current breakdown')
                        if previous is not None:
                            added=observe_measured_held_rows(pilot,stages,ids,previous,terminal,
                                reward,learning_termination_mask(contract['reward_profile'],terminated,truncated),clocks,active)
                            if added and pilot.training and (pilot.success_bank is not None
                                    or getattr(pilot, 'measured_credit_bank', None) is not None):
                                measured_goal_batches.append((ids[active[ids]].detach().cpu(),pilot.history[-1]))
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
                            if supplemental:
                                r.update(actor_supplemental=pc[SUPPLEMENTAL_GROUP][i].copy(),
                                    next_actor_supplemental=tc[SUPPLEMENTAL_GROUP][i].copy())
                            if critic_episode_clock is not None:
                                r.update(critic_episode_remaining=pc[TASK_TIMING_GROUP][i].copy(),
                                    next_critic_episode_remaining=tc[TASK_TIMING_GROUP][i].copy())
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
                            if 'contact_quality' in bc:last[i]['contact_quality']=float(bc['contact_quality'][i])
                        active&=~(terminated|truncated)
                        if step%30==0:
                            progress=dict(wave=wave_index,split=wave['split'],step=step+1,
                                active=int(active.sum()),held=len(ids),actual_rows=total_rows,
                                transition_per_s=total_rows/(time.monotonic()-start),learner=pilot.report(),last=last)
                            progress['rollout_transition_per_s']=wave_rows/(time.monotonic()-rollout_start)
                            if policy_servo_diagnostics and policy_servo_diagnostics[-1]['step']==step+1:
                                progress['policy_servo_diagnostics']=policy_servo_diagnostics[-1]
                            (output/'progress.json').write_text(json.dumps(progress)+'\n')
                            print('[BATCHED SAC] '+json.dumps({k:v for k,v in progress.items() if k not in ('learner','last')})+
                                f' actor={pilot.actor_updates} critic={pilot.critic_updates}',flush=True)
                        if pilot.training and pilot.critic_updates and pilot.critic_updates%1024==0:pilot.save()
            finally:
                env.termination_manager.compute=compute
                if scene_videos is not None:scene_videos.close(last)
            if args.policy_servo_diagnostics:
                (output/f'policy_servo_diagnostics_wave_{wave_index:04d}.json').write_text(json.dumps(
                    dict(role='read_only_measured_policy_servo_step_clipping',wave=wave_index,
                         split=wave['split'],samples=policy_servo_diagnostics,
                         sample_period_control_steps=30,whole_wave_result_requires_metrics=True),indent=2)+'\n')
            for i,samples in enumerate(rows):
                success=bool(last[i] and last[i]['success'])
                collection_mode=None
                body_sampler=getattr(pilot,'body_behavior_sampler',None)
                if wave['split']=='train' and getattr(body_sampler,'greedy_unselected_policy',False):
                    collection_mode=('coherent_arm_exploration' if body_sampler.selected[i] else
                        'greedy_current_policy') if body_sampler.initialized[i] else 'no_held_mode_draw'
                if samples:
                    recorder.start_episode(initial_state=snapshots[i])
                    recorder.episode.attrs['wave']=wave_index
                    recorder.episode.attrs['environment']=i
                    recorder.episode.attrs['layout_json']=json.dumps(wave['layouts'][i]['layout'],sort_keys=True)
                    recorder.episode.attrs['initial_layout_guard_valid']=bool(valid_layout[i])
                    if collection_mode is not None:
                        recorder.episode.attrs['collection_policy_mode']=collection_mode
                    recorder.append_many(samples)
                    recorder.finish_episode(success=success,reason='numerical_failure_excluded_corrupt_row'
                        if last[i].get('numerical_failure') else 'success' if success else 'failure'
                        if not active[i] else 'interrupted_or_step_limit')
                outcomes.append(dict(wave=wave_index,split=wave['split'],environment=i,
                    layout=wave['layouts'][i]['layout'],result=last[i],complete=bool(not active[i]),
                    initial_layout_valid=bool(valid_layout[i]),initial_settling_steps=settled,
                    initial_layout_guard=layout_guard_references[i],executed_transition_rows=len(samples)))
                if wave['split']=='train' and getattr(pilot,'jaw_behavior',None) is not None:
                    outcomes[-1]['collection_jaw_behavior']=pilot.jaw_behavior
                if wave['split']=='train' and getattr(pilot,'body_behavior',None) is not None:
                    outcomes[-1]['collection_body_behavior']=pilot.body_behavior
                if collection_mode is not None:
                    outcomes[-1]['collection_policy_mode']=collection_mode
                if wave['split']=='train' and getattr(pilot,'body_saturation',None) is not None:
                    outcomes[-1]['actor_body_regularization']=pilot.body_saturation
                if wave['split']=='train' and getattr(pilot,'jaw_saturation',None) is not None:
                    outcomes[-1]['actor_jaw_regularization']=pilot.jaw_saturation
                if wave['split']=='train' and getattr(pilot,'success_jaw_balance',None) is not None:
                    outcomes[-1]['successful_jaw_balance']=pilot.success_jaw_balance
            if not pilot.training and updates_before!=(pilot.actor_updates,pilot.critic_updates,pilot.replay.size):
                raise ValueError('Evaluation modified optimizer counters or replay')
            if backend_eval or workplace_eval:
                frozen_eval_contract=backend_eval or workplace_eval
                frozen_eval_contract['frozen_network_integrity']=verify_frozen_network_snapshot(pilot,frozen_integrity)
                frozen_eval_contract['actor_critic_updates_and_replay_size_unchanged']=True
                frozen_eval_contract['replay_rows_imported']=pilot.replay.size
            pilot.training=args.training
            if args.training and pilot.success_bank is not None:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import add_completed_training_wave
                add_completed_training_wave(pilot.success_bank,wave,outcomes[-n:],measured_goal_batches,source_run=output.name)
            if args.training and getattr(pilot, 'measured_credit_bank', None) is not None:
                pilot.add_measured_training_wave(wave, outcomes[-n:], measured_goal_batches,
                    source_run=output.name)
            if args.training:pilot.save(final=True)
            regression=baseline_failed=False
            if guard is not None and wave['split']=='validation':
                check=evaluate_development_wave(guard,wave['layouts'],last,wave_index,
                    actor_updates=pilot.actor_updates,completed=not stopped['value'])
                development_checks.append(dict(wave=wave_index,actor_updates=pilot.actor_updates,
                    critic_updates=pilot.critic_updates,**check))
                regression=check.get('regression',False)
                baseline_failed=check.get('baseline_failed',False)
            if not stopped['value']:completed_wave_count+=1
            (output/'metrics.json').write_text(json.dumps(dict(policy=pilot.artifact_type,outcomes=outcomes,
                frozen_physics_backend_evaluation=backend_eval,
                CPU_workplace_probe=workplace_eval,
                learner=pilot.report(),actual_rows=total_rows,seconds=time.monotonic()-start,
                development_checks=development_checks),indent=2)+'\n')
            status=('interrupted' if stopped['value'] else
                'physical_reproducibility_failed' if regression and check['regression_cause']=='physical_reproducibility_loss_without_actor_update' else
                'policy_regression' if regression else 'baseline_performance_failed' if baseline_failed else
                'training' if wave_index+1<len(waves) else 'complete')
            (output/'status.json').write_text(json.dumps(dict(status=status,
                completed_waves=completed_wave_count,recorded_waves=wave_index+1,
                total_waves=len(waves),interrupted=stopped['value']))+'\n')
            if regression or baseline_failed:
                print('[DEVELOPMENT PERFORMANCE STOP] '+json.dumps(development_checks[-1]),flush=True)
                return 1
        if stopped['value']:
            (output/'status.json').write_text(json.dumps(dict(status='interrupted',
                completed_waves=completed_wave_count,recorded_waves=len(outcomes)//n))+'\n')
        return 0
    except Exception:
        import traceback
        failure=traceback.format_exc()
        print(failure,flush=True)
        if pilot is not None and args.training:
            # Preserve previously measured rows on a runner error. A corrupt
            # learner must never replace its last finite checkpoint.
            try:
                if all(bool(torch.isfinite(v).all()) for v in pilot.agent.state_dict().values()):
                    pilot.save(final=True)
                    print('[FAILURE CHECKPOINT] finite learner and previous measured replay saved',flush=True)
            except Exception:
                print('[FAILURE CHECKPOINT] '+traceback.format_exc(),flush=True)
        output.mkdir(parents=True,exist_ok=True)
        if not (output/'manifest.json').exists():
            (output/'manifest.json').write_text(json.dumps(dict(artifact_type='batched_staged_runtime_failure',
                waves=waves,exception_before_collection=True))+'\n')
        (output/'failure.json').write_text(json.dumps(dict(traceback=failure))+'\n')
        (output/'status.json').write_text(json.dumps(dict(status='failed'))+'\n')
        raise
    finally:
        if base_substep_trace:base_substep_trace.close()
        if grasp_audit:grasp_audit.close()
        if recorder:recorder.close()
        if env:env.close()
        app.close()


if __name__=='__main__':raise SystemExit(main())
