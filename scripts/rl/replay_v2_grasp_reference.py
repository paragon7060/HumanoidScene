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
    parser.add_argument('--contact-diagnostics', action='store_true',
                        help='Record existing privileged per-jaw pinch checks before reset; no actor/reward changes.')
    parser.add_argument('--flap-pose-source',choices=('nominal','articulated'),default='nominal',
                        help='Explicit observation semantics; articulated panels use the existing simulator perception proxy.')
    parser.add_argument('--allow-nominal-flap-prior',action='store_true',
                        help='Articulated collection only: allow an approximate old reset/VR guide; never migrates its Q rows or panel poses.')
    parser.add_argument('--actor-checkpoint', type=Path,
                        help='Evaluate this deterministic actor instead of the VR/live-IK controller.')
    parser.add_argument('--pose-student-checkpoint',type=Path,
                        help='Evaluate the learned absolute-goal student with no live reference path; BC diagnostic, not SAC.')
    parser.add_argument('--pose-student-training',action=argparse.BooleanOptionalAction,default=False,
                        help='Continue the goal student with actual SAC. Evaluation remains the default.')
    parser.add_argument('--pose-student-native-seed',type=Path,action='append',
                        help='Repeat for measured current-environment successes; episode clocks/anchors stay separate.')
    parser.add_argument('--staged-base-waypoints',type=Path,
                        help='Neutral-arm approach then TRAIN-derived base hold. Default frozen diagnostic; new staged SAC has separate replay.')
    parser.add_argument('--staged-goal-sac',action='store_true',
                        help='Separate fresh-Q SAC on the 21 remaining goals after physical base settling.')
    parser.add_argument('--staged-goal-training',action=argparse.BooleanOptionalAction,default=False)
    parser.add_argument('--collect-train-goals',action='store_true',
                        help='Explicit TRAIN-data collection with frozen staged SAC; save exact21 goals, no optimizer or live teacher.')
    parser.add_argument('--train-collection-behavior',choices=('greedy','checkpoint-exploration'),default='greedy',
                        help='TRAIN collection only: keep deployed means or sample the checkpoint behavior while weights stay frozen.')
    parser.add_argument('--train-collection-seed',type=int,default=0)
    parser.add_argument('--staged-contact-ik-native-seed',type=Path,action='append',
                        help='Staged frozen diagnostic only: native successful TRAIN calibration for local contact IK. Not a SAC policy.')
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
    parser.add_argument('--residual-controller',choices=('delta','goal','retargeted-goal'),default='delta',
                        help='Goal mode anchors accumulated base/torso/joint goals to measured reference controller state.')
    parser.add_argument('--layout-json',type=Path,help='Explicit measured-target layout from a separate train/holdout split.')
    parser.add_argument('--layout-vr-teacher',action='store_true',help='Physical layout diagnostic using the original VR/live-contact IK guide, no SAC.')
    parser.add_argument('--vr-layout-retarget',action='store_true',
                        help='Layout VR diagnostic only: move its whole initial approach with the actually perceived selected box.')
    parser.add_argument('--residual-zero',action='store_true',help='Geometry-guide physical probe, no learned actions or optimizer updates.')
    parser.add_argument('--vr-orientation-mode',choices=('full','closing-axis'),default='full',
                        help='VR/live-IK diagnostic only: constrain the complete wrist or just its jaw closing axis.')
    parser.add_argument('--staged-contact-ik-mode',choices=('near-contact','after-base-hold'),
                        default='near-contact',help='Frozen teacher probe: local contact correction or front-stage IK after physical base settling.')
    parser.add_argument('--staged-contact-ik-orientation',choices=('full','closing-axis'),default='full')
    parser.add_argument('--staged-contact-ik-velocity-feedforward',action='store_true')
    parser.add_argument('--staged-contact-ik-lock-assignment',action='store_true')
    parser.add_argument('--vr-contact-torso-forward-m',type=float,default=0.,
                        help='VR/live-IK diagnostic only: bounded upright torso X assist during contact.')
    parser.add_argument('--vr-contact-torso-up-m',type=float,default=0.,
                        help='VR diagnostic only: additional upright height with a separate travel/MDP contract; never resumes an old policy.')
    parser.add_argument('--torso-extra-height-m',type=float,default=0.,
                        help='Explicit physical travel profile for any controller; matching recorded data/prior required, default unchanged.')
    parser.add_argument('--vr-close-distance-m',type=float,default=.035)
    parser.add_argument('--vr-coordinated-close',action='store_true',
                        help='VR/live-IK diagnostic only: wait for both hands before closing either jaw.')
    parser.add_argument('--vr-reference-grippers',action='store_true',
                        help='VR/live-IK diagnostic only: preserve the demonstrator\'s actual jaw timing.')
    parser.add_argument('--vr-handoff-distance-m',type=float,default=0.,
                        help='VR diagnostic only: start live geometry tracking when BOTH perceived hands approach within this distance; default reference timing.')
    parser.add_argument('--vr-contact-rest-mode',choices=('reference','current'),default='reference',
                        help='VR/live-IK diagnostic only: use the measured current posture as contact IK rest.')
    parser.add_argument('--vr-contact-goal',choices=('demo','center'),default='demo',
                        help='VR diagnostic only: calibrated demo offset or perceived panel midpoint for contact IK.')
    parser.add_argument('--vr-contact-base-forward-m',type=float,default=0.,
                        help='VR/live-IK diagnostic only: bounded base approach after handoff; stop advancing on first pinch.')
    parser.add_argument('--vr-arm-reach-fraction',type=float,default=.95,
                        help='VR/live-IK diagnostic only: gross URDF reach sphere fraction; joint/rate/collision limits remain active.')
    parser.add_argument('--steps', type=int, default=900)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_rack_roller_cli_args(parser)
    add_base_drive_cli_args(parser)
    parser.set_defaults(headless=True, robot_model='s63', gripper='leju-twofinger', rack_rollers=True)
    args = parser.parse_args()
    if (args.train_collection_behavior!='greedy' or args.train_collection_seed) and not args.collect_train_goals:
        parser.error('Frozen behavior sampling requires explicitly declared TRAIN goal collection')
    if not 0<=args.train_collection_seed<2**32:
        parser.error('TRAIN behavior seed must be in0..2**32-1')
    if args.collect_train_goals:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.training_goal_collection import validate_training_collection
        try:
            validate_training_collection(
                json.loads(args.layout_json.read_text()) if args.layout_json and args.layout_json.is_file() else None,
                staged_policy=args.staged_goal_sac,
                optimization=args.staged_goal_training or args.pose_student_training,
                live_teacher=bool(args.staged_contact_ik_native_seed or args.layout_vr_teacher
                                  or args.executed_actions or args.actor_reference_mix is not None))
        except ValueError as error:
            parser.error(str(error))
    if args.capture_every < 1 or args.episode_index < 0 or not 1 <= args.steps <= 900:
        parser.error('Capture interval must be positive, episode index nonnegative, and steps in1..900')
    if args.allow_nominal_flap_prior and args.flap_pose_source!='articulated':
        parser.error('Approximate nominal prior is only meaningful for articulated collection')
    if args.body_envelope and not args.actor_checkpoint:
        parser.error('--body-envelope requires --actor-checkpoint')
    if args.joint_offset_rad is not None and (not args.actor_checkpoint or args.body_envelope
                                             or args.actor_reference_mix is not None):
        parser.error('Joint-goal comparison requires a matching actor and excludes other mix/envelope variants')
    if args.actor_checkpoint and not args.actor_checkpoint.is_file():
        parser.error('Missing actor checkpoint')
    if args.pose_student_checkpoint and (not args.pose_student_checkpoint.is_file() or
            args.actor_checkpoint or args.executed_actions or args.residual_sac or args.layout_vr_teacher):
        parser.error('Pose student needs its own checkpoint and excludes reference/actor mixtures')
    if (args.pose_student_training or args.pose_student_native_seed) and not args.pose_student_checkpoint:
        parser.error('Goal SAC options require --pose-student-checkpoint')
    if args.staged_base_waypoints and (not args.staged_base_waypoints.is_file()
            or not args.pose_student_checkpoint or args.pose_student_training):
        parser.error('Staged base probe requires existing templates and a frozen pose policy; new matching SAC is a separate task')
    if args.staged_contact_ik_native_seed and (not args.staged_base_waypoints
            or not all(p.is_file() for p in args.staged_contact_ik_native_seed)):
        parser.error('Native contact IK requires staged frozen control and existing TRAIN success calibration')
    if (args.staged_contact_ik_mode!='near-contact' or args.staged_contact_ik_orientation!='full'
            or args.staged_contact_ik_velocity_feedforward or args.staged_contact_ik_lock_assignment) and not args.staged_contact_ik_native_seed:
        parser.error('A staged contact handoff mode requires native TRAIN contact calibration')
    if args.staged_goal_training and not args.staged_goal_sac:
        parser.error('Staged goal training requires its separate controller mode')
    if args.staged_goal_sac and (not args.staged_base_waypoints or args.staged_contact_ik_native_seed
                                or not args.pose_student_native_seed):
        parser.error('Staged SAC requires base waypoints and a frozen warm-start audit, and excludes live IK teachers')
    if args.pose_student_training and (not args.pose_student_native_seed or
            not all(path.is_file() for path in args.pose_student_native_seed)):
        parser.error('Goal SAC requires an existing measured current success seed')
    if args.executed_actions and (args.actor_checkpoint or not args.executed_actions.is_file()):
        parser.error('--executed-actions needs an existing native dataset and excludes --actor-checkpoint')
    if args.residual_sac and (not args.executed_actions or not 0 < args.residual_scale <= .2):
        parser.error('Residual pilot needs measured commands and a scale in (0,.2]')
    if args.residual_checkpoint and (not args.residual_sac or not args.residual_checkpoint.is_file()):
        parser.error('Residual checkpoint requires --residual-sac and an existing file')
    if args.residual_sac and not args.residual_training and not args.residual_checkpoint and not args.residual_zero:
        parser.error('Residual evaluation requires a learned residual checkpoint')
    if args.layout_json and not args.layout_vr_teacher and not args.pose_student_checkpoint and (not args.residual_sac or args.residual_controller!='retargeted-goal'):
        parser.error('Varied layouts require the geometry-conditioned residual controller')
    if args.layout_vr_teacher and (not args.layout_json or args.residual_sac or args.executed_actions or args.actor_checkpoint):
        parser.error('Layout VR teacher requires a layout and excludes policy/recorded action replay')
    if args.vr_layout_retarget and not args.layout_vr_teacher:
        parser.error('VR path retargeting requires --layout-vr-teacher')
    if args.residual_zero and (not args.residual_sac or args.residual_training):
        parser.error('Zero-residual probes require --no-residual-training')
    if not 0<=args.vr_contact_torso_forward_m<=.08:
        parser.error('VR contact torso assist must be within0..8cm')
    if not 0<=args.vr_contact_torso_up_m<=.08:
        parser.error('VR contact torso up diagnostic must be within0..8cm')
    if not 0<=args.torso_extra_height_m<=.08:
        parser.error('Extra torso travel must be within0..8cm')
    if args.torso_extra_height_m and args.vr_contact_torso_up_m and args.torso_extra_height_m!=args.vr_contact_torso_up_m:
        parser.error('VR assist and explicit travel profile must request the same extra height')
    if not 0<=args.vr_contact_base_forward_m<=.08:
        parser.error('VR contact base assist must be within0..8cm')
    if not .95<=args.vr_arm_reach_fraction<=1:
        parser.error('VR gross arm reach fraction must be within0.95..1')
    if not .003<=args.vr_close_distance_m<=.035:
        parser.error('VR closing gate must be within3..35mm')
    if args.vr_reference_grippers and args.vr_coordinated_close:
        parser.error('Choose reference timing or coordinated geometric closing')
    if args.vr_handoff_distance_m and not .05<=args.vr_handoff_distance_m<=.25:
        parser.error('VR approach handoff must be0 or within5..25cm')
    if (args.vr_orientation_mode!='full' or args.vr_contact_torso_forward_m or args.vr_contact_torso_up_m or
        args.vr_close_distance_m!=.035 or args.vr_coordinated_close or args.vr_reference_grippers or
        args.vr_contact_rest_mode!='reference' or args.vr_contact_base_forward_m or
        args.vr_arm_reach_fraction!=.95 or args.vr_handoff_distance_m or args.vr_contact_goal!='demo') and (args.actor_checkpoint or args.executed_actions or args.pose_student_checkpoint):
        parser.error('VR diagnostic options only apply to the live VR/IK guide')
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
            VRJointTracker, select_reference_episode, settle_reference_scene,configure_vr_torso_up_diagnostic,
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
        from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import require_current_lift_contract,frozen_prior_lift_contract
        require_current_lift_contract(contract)
        if (contract.get('task_family') != 'multi_box_v2' or contract.get('skill') != 'grasp'
                or contract.get('robot_model') != 's63' or contract.get('gripper') != 'leju-twofinger'
                or contract.get('flap_pose_source', 'nominal') != 'nominal'
                or contract.get('reward_profile', {}).get('weights') != asdict(MultiBoxRewardWeights())):
            raise ValueError('Reference replay needs the current nominal v2 grasp contract/rewards')
        cfg = MultiBoxGraspAssemblyEnvCfg(num_envs=1)
        cfg.episode_length_s = 30.
        cfg.multi_box = replace(cfg.multi_box, self_collision_enabled=contract['self_collision']['enabled'],
                                flap_pose_source=args.flap_pose_source)
        cfg.sim.device = args.device or 'cuda:0'
        if args.staged_goal_sac:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import configure_staged_physics
            configure_staged_physics(cfg,contract)
        elif contract.get('physics_dynamics'):
            raise ValueError('Explicit changed dynamics require the matching staged goal controller')
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
        contract=configure_vr_torso_up_diagnostic(cfg,contract,
            args.torso_extra_height_m or args.vr_contact_torso_up_m)
        if args.flap_pose_source=='articulated':
            from kuavo_isaaclab_scene.rl.multi_box.observations.contracts import flap_observation_contract
            contract=contract|flap_observation_contract(args.flap_pose_source)
        # The source manifest can describe a vectorized run. This replay is
        # actually one environment; preserve old archives and report new runs.
        contract=contract|{'num_envs':1}

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
        batch, demo_audit = load_v2_grasp_demonstrations(args.demo_dataset,
            self_collision_enabled=cfg.multi_box.self_collision_enabled,
            flap_pose_source=args.flap_pose_source,
            allow_nominal_flap_prior=args.allow_nominal_flap_prior)
        demo = select_reference_episode(batch, args.episode_index)
        observation, _ = env.reset(seed=42)
        observation, _ = _settle_initial_resets(env, observation)
        layout = None
        staged_layout_guard=None
        scene_demo=demo
        if args.layout_json:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import GraspLayout, layout_reset_observation
            from kuavo_isaaclab_scene.workcell.rack_rollers import resolve_rack_roller_settings
            layout=GraspLayout(**json.loads(args.layout_json.read_text())).validate()
            scene_demo={key:value.clone() for key,value in demo.items()}
            rollers=resolve_rack_roller_settings()
            scene_demo['actor_obs'][0]=layout_reset_observation(demo['actor_obs'][0],layout,cfg.multi_box,
                roller_clearance_m=rollers.box_clearance_m if rollers.enabled else 0.)
        if args.staged_goal_sac:
            # Match the trained staged runner's asset/controller/FK reset.
            # Legacy VR/other-policy replay keeps its own reset lifecycle.
            from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import settle_batched_layouts
            observation,settling_steps,staged_initial_valid,staged_layout_guard=settle_batched_layouts(
                env,scene_demo['actor_obs'][:1].to(env.device),allow_partial=False)
            rack=env.scene['rack'].data.root_pose_w.clone()
        elif layout:
            observation,rack,settling_steps=settle_reference_scene(env,scene_demo,settle_all=True)
        else:
            observation, rack, settling_steps = settle_reference_scene(env, demo)
        projection = GraspActionProjector(list(actions.items()))
        initial_base_pose=env.scene['robot'].data.root_pose_w[0].tolist()
        initial_rack_pose=env.scene['rack'].data.root_pose_w[0].tolist()
        initial_actor=observation['policy'][0].detach().cpu().clone()
        teacher = None
        agent = state = limits = executed = joint_goal = residual = pose_student = pose_sac = None
        initial_actor_error = None
        staged_base = None
        staged_contact_ik = None
        staged_goal_sac = None
        controller_name = 'VR_reference_plus_contact_confirmed_IK_NOT_SAC'
        if args.pose_student_checkpoint:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
            pose_state=torch.load(args.pose_student_checkpoint,map_location=env.device,weights_only=False)
            staged_type=pose_state.get('artifact_type')
            staged_resume=staged_type in ('staged_base_hold_remaining_goal_sac_v1',
                                         'staged_base_hold_remaining_hybrid_sac_v1')
            if staged_resume and not args.staged_goal_sac:
                raise ValueError('A staged checkpoint cannot run as the legacy whole-body goal controller')
            if staged_resume:
                pose_state=pose_state['frozen_warm_start']
            if args.pose_student_training or pose_state.get('artifact_type')=='pose_goal_sac_no_live_reference':
                if not args.pose_student_native_seed:raise ValueError('Goal SAC evaluation also requires the declared physical seed audit')
                from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
                base=env.action_manager.get_term('base');upper=env.action_manager.get_term('upper_body')
                head=env.action_manager.get_term('head');height=env.action_manager.get_term('height')
                from kuavo_isaaclab_scene.rl.multi_box.experiments.reference_residual import validate_goal_feedback_rates
                validate_goal_feedback_rates(base._scale,upper._scale,head._scale,height.cfg.speed_m_s,env.step_dt)
                pose_sac=PoseGoalSACPilot(pose_state if staged_resume else args.pose_student_checkpoint,
                                        args.pose_student_native_seed,
                                        frozen_prior_lift_contract(contract) if staged_resume else contract,
                                        args.output_dir,training=args.pose_student_training,device=env.device)
                controller_name='learned_pose_goal_SAC_NO_live_reference'
            else:
                pose_student=PoseStudent(pose_state,env.device)
                pose_student.validate_physical_contract(contract)
                controller_name='learned_absolute_pose_student_BC_NOT_SAC_NO_live_reference'
            if args.staged_base_waypoints:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
                waypoints=json.loads(args.staged_base_waypoints.read_text())
                if waypoints.get('physical_action_contract')!=contract['action_contract']:
                    raise ValueError('Staged workplace physical travel contract differs')
                staged_base=StagedBaseHoldDiagnostic((pose_sac or pose_student).coordinates,
                    waypoints,observation['policy'])
                controller_name='frozen_neural_grasp_with_analytic_base_staging_NOT_new_staged_SAC'
                if args.staged_contact_ik_native_seed:
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import merge_executed_successes
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_contact_ik import StagedContactIKDiagnostic
                    measured,audit=merge_executed_successes(args.staged_contact_ik_native_seed,contract)
                    token,_=target_token(measured['actor_obs'])
                    mask=(token[:,10:12].sum(-1)>.5)==(staged_base.shelf=='upper')
                    if not bool(mask.any()):raise ValueError('No native contact calibration for this shelf')
                    measured={k:v[mask].to(env.device) for k,v in measured.items()}
                    audit=audit|{'calibration_shelf':staged_base.shelf,
                                 'calibration_rows':int(mask.sum()),'used_for_Q':False}
                    staged_contact_ik=StagedContactIKDiagnostic(env,measured,audit,
                        handoff_mode=args.staged_contact_ik_mode,
                        orientation_mode=args.staged_contact_ik_orientation,
                        velocity_feedforward=args.staged_contact_ik_velocity_feedforward,
                        lock_assignment=args.staged_contact_ik_lock_assignment)
                    controller_name='frozen_neural_approach_with_staged_base_and_native_contact_IK_teacher_NOT_SAC'
                if args.staged_goal_sac:
                    if pose_sac is None:
                        raise ValueError('Staged SAC warm start must be the matching frozen goal-SAC network')
                    if not args.staged_goal_training and not staged_resume:
                        raise ValueError('Frozen staged SAC evaluation requires a trained staged checkpoint')
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalSACPilot
                    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_hybrid_goal_sac import StagedHybridGoalSACPilot
                    staged_class=(StagedHybridGoalSACPilot if staged_type==StagedHybridGoalSACPilot.artifact_type
                                  else StagedGoalSACPilot)
                    staged_goal_sac=staged_class(pose_sac,contract,args.output_dir,staged_base,
                        checkpoint=args.pose_student_checkpoint if staged_resume else None,
                        training=args.staged_goal_training,device=env.device)
                    controller_name='staged_base_hold_remaining_goal_SAC_NO_live_reference_or_IK_teacher'
        elif args.executed_actions:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import read_executed_successes
            measured, _ = read_executed_successes(args.executed_actions, contract)
            executed = select_reference_episode(measured, args.episode_index)
            initial_actor_error = float((observation['policy'][0].cpu()-executed['actor_obs'][0]).abs().max())
            if initial_actor_error > 1e-5 and not args.layout_json:
                raise ValueError(f'Recorded-command replay starts with a different actor observation: {initial_actor_error}')
            controller_name = 'recorded_current_GPU_actions_open_loop_NOT_SAC'
            if args.residual_sac:
                if args.residual_controller != 'delta':
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
                if args.residual_controller=='retargeted-goal':
                    residual.controller.retarget(executed['actor_obs'][0].to(env.device),observation['policy'][0])
                controller_name = 'fixed_scene_measured_reference_plus_SAC_residual_NOT_standalone_SAC'
                if layout:
                    controller_name='perceived_box_retargeted_reference_plus_SAC_residual'
                if args.residual_zero:
                    controller_name='perceived_box_retargeted_reference_ZERO_residual_NOT_learned_policy'
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
            vr_retarget=None
            if args.vr_layout_retarget:
                from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import retarget_reference_rack
                original_actor=demo['actor_obs'][0]
                actual_actor=observation['policy'][0].detach().cpu()
                retargeted={key:value.clone() for key,value in demo.items()}
                retargeted['actor_obs'],vr_retarget=retarget_reference_rack(demo['actor_obs'],original_actor,actual_actor)
                retargeted['next_actor_obs'],_=retarget_reference_rack(demo['next_actor_obs'],original_actor,actual_actor)
                demo=retargeted
            teacher = VRJointTracker(env, demo, rack,orientation_mode=args.vr_orientation_mode,
                                     contact_torso_forward_m=args.vr_contact_torso_forward_m,
                                     close_distance_m=args.vr_close_distance_m,coordinated_close=args.vr_coordinated_close,
                                     reference_grippers=args.vr_reference_grippers,contact_rest_mode=args.vr_contact_rest_mode,
                                     contact_base_forward_m=args.vr_contact_base_forward_m,
                                     arm_reach_fraction=args.vr_arm_reach_fraction,
                                     contact_torso_up_m=args.vr_contact_torso_up_m,
                                     handoff_distance_m=args.vr_handoff_distance_m,
                                     contact_goal=args.vr_contact_goal)
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
        if pose_sac:
            (output/'manifest.json').write_text(json.dumps(contract|{
                'artifact_type':pose_sac.artifact_type,'layout':layout.record() if layout else None,
                'policy':controller_name,'goal_contract':pose_sac.contract},indent=2)+'\n')
            (output/'env.yaml').write_text(json.dumps({'physical_contract':contract},indent=2)+'\n')
            (output/'agent.yaml').write_text(json.dumps({'goal_contract':pose_sac.contract,
                'sac_config':asdict(pose_sac.agent.config)},indent=2)+'\n')
            (output/'status.json').write_text(json.dumps({'status':'training' if pose_sac.training else 'evaluating'})+'\n')
        elif residual:
            artifact_type='layout_reference_residual_sac' if layout else 'fixed_scene_reference_residual_sac'
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type': artifact_type, 'layout':layout.record() if layout else None,
                'residual_contract': residual.contract}, indent=2)+'\n')
            (output/'env.yaml').write_text(json.dumps({'physical_contract':contract},indent=2)+'\n')
            (output/'agent.yaml').write_text(json.dumps({'residual_contract':residual.contract,
                'sac_config':asdict(residual.agent.config)},indent=2)+'\n')
            (output/'status.json').write_text(json.dumps({'status':'training' if args.residual_training else 'evaluating'})+'\n')
        elif args.actor_reference_mix is None:
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type':'pose_student_physical_evaluation_diagnostic' if pose_student else 'physical_reference_replay_diagnostic',
                'policy':controller_name,'episode_index':args.episode_index,
                'training':False,'vr_orientation_mode':args.vr_orientation_mode,
                'vr_contact_torso_forward_m':args.vr_contact_torso_forward_m,
                'vr_close_distance_m':args.vr_close_distance_m,'vr_coordinated_close':args.vr_coordinated_close,
                'vr_reference_grippers':args.vr_reference_grippers,
                'vr_contact_rest_mode':args.vr_contact_rest_mode,
                'vr_contact_base_forward_m':args.vr_contact_base_forward_m,
                'vr_arm_reach_fraction':args.vr_arm_reach_fraction,
                'vr_contact_torso_up_m':args.vr_contact_torso_up_m,
                'torso_extra_height_m':args.torso_extra_height_m,
                'vr_handoff_distance_m':args.vr_handoff_distance_m,
                'vr_contact_goal':args.vr_contact_goal,
                'source_flap_prior_audit':demo_audit,
                'layout':layout.record() if layout else None,
                'vr_layout_retarget':args.vr_layout_retarget,
                'vr_retarget':vr_retarget if args.vr_layout_retarget else None},indent=2)+'\n')
        meta = dict(task_family='multi_box_v2', skill='grasp', robot_model='s63',
            gripper='leju-twofinger', rack_rollers=True, controller_mapping='scaled',
            action_dim=sum(actions.values()), actor_obs_dim=dims['policy'][0],
            critic_obs_dim=dims['policy'][0]+dims['critic'][0], action_terms=list(map(list, actions.items())),
            control_dt=env.step_dt, multi_box=asdict(cfg.multi_box), episode_seconds=30.,
            collection_source=('pose_goal_sac_no_live_reference' if pose_sac else
                               'learned_pose_student_BC_no_live_reference' if pose_student else
                               'layout_reference_residual_sac' if residual and layout else
                               'fixed_scene_reference_residual_sac' if residual else
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
            residual_contract=residual.contract if residual else None,
            pose_student_checkpoint=str(args.pose_student_checkpoint.resolve()) if pose_student or pose_sac else None,
            layout=layout.record() if layout else None, zero_residual_probe=args.residual_zero)
        meta['pose_goal_contract']=pose_sac.contract if pose_sac else None
        if staged_base:
            meta['collection_source']=staged_base.collection_source
            meta['staged_base_contract']=staged_base.report()
            manifest=json.loads((output/'manifest.json').read_text())
            manifest.update(artifact_type=staged_base.collection_source,
                            staged_base_contract=staged_base.report(),training=False)
            if staged_contact_ik:
                meta['staged_contact_ik_contract']=staged_contact_ik.report()
                manifest['staged_contact_ik_contract']=staged_contact_ik.report()
            if staged_goal_sac:
                meta['collection_source']=staged_goal_sac.artifact_type
                meta['staged_goal_contract']=staged_goal_sac.contract
                meta['initial_layout_guard']=staged_layout_guard
                manifest.update(artifact_type=staged_goal_sac.artifact_type,
                    goal_contract=staged_goal_sac.contract,training=staged_goal_sac.training,
                    initial_layout_guard=staged_layout_guard,
                    wave_reset_controller_contract=staged_layout_guard['controller_reset']['contract'])
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        goal_collector = None
        if args.collect_train_goals:
            from kuavo_isaaclab_scene.rl.multi_box.experiments.training_goal_collection import TrainingGoalCollector, FORMAT
            behavior_seed=args.train_collection_seed if args.train_collection_behavior=='checkpoint-exploration' else None
            goal_collector = TrainingGoalCollector(layout.record(), staged_goal_sac.contract,
                behavior=args.train_collection_behavior,behavior_seed=behavior_seed)
            if behavior_seed is not None:
                torch.manual_seed(behavior_seed)
            collection_initial_counters = (staged_goal_sac.actor_updates, staged_goal_sac.critic_updates)
            manifest = json.loads((output/'manifest.json').read_text())
            manifest.update(training_data_collection=FORMAT, collection_phase='train',
                            optimizer_training=False, evaluation_data=False,
                            train_collection_behavior=args.train_collection_behavior,behavior_seed=behavior_seed)
            (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            meta.update(training_data_collection=FORMAT, collection_phase='train',
                        optimizer_training=False, evaluation_data=False,
                        train_collection_behavior=args.train_collection_behavior,behavior_seed=behavior_seed)
        meta['vr_orientation_mode']=args.vr_orientation_mode
        meta['vr_contact_torso_forward_m']=args.vr_contact_torso_forward_m
        meta['vr_close_distance_m']=args.vr_close_distance_m
        meta['vr_coordinated_close']=args.vr_coordinated_close
        meta['vr_reference_grippers']=args.vr_reference_grippers
        meta['vr_contact_rest_mode']=args.vr_contact_rest_mode
        meta['vr_contact_base_forward_m']=args.vr_contact_base_forward_m
        meta['vr_arm_reach_fraction']=args.vr_arm_reach_fraction
        meta['vr_contact_torso_up_m']=args.vr_contact_torso_up_m
        meta['torso_extra_height_m']=args.torso_extra_height_m
        meta['vr_layout_retarget']=args.vr_layout_retarget
        meta['vr_handoff_distance_m']=args.vr_handoff_distance_m
        meta['vr_contact_goal']=args.vr_contact_goal
        meta['source_flap_prior_audit']=demo_audit
        recorder = RlTransitionRecorder(output/'executed_transitions.hdf5', meta)
        recorder.start_episode(initial_state=capture_rl_initial_state(env, observation))
        renderer = None if args.no_video else SceneVideo(env,
            caption=(f'SAC TRAIN data | FROZEN weights, optimizer=0 | {args.train_collection_behavior}' if goal_collector else
                     f'Base hold + SAC21 goals | NO live reference/IK | train={staged_goal_sac.training}' if staged_goal_sac else
                     f'Base staging + FROZEN approach + local IK | TEACHER, NOT SAC' if staged_contact_ik else
                     f'Base staging + FROZEN grasp | DIAGNOSTIC, not new SAC' if staged_base else
                     f'Learned pose goals | SAC | NO live reference | train={pose_sac.training}' if pose_sac else
                     f'Learned pose student | BC, NOT SAC | NO live reference' if pose_student else
                     f'Reference + {"zero" if args.residual_zero else "SAC"} residual | layout={layout.seed if layout else "fixed"} | train={args.residual_training}' if residual else
                     f'Joint-goal BC actor | NOT trained SAC | offset {args.joint_offset_rad}rad' if joint_goal else
                     f'VR teacher + {args.actor_reference_mix:.0%} actor | NOT pure SAC' if args.actor_reference_mix is not None else
                     f'Recorded actual commands | NOT SAC | actual {env.device} PhysX' if executed is not None else
                     f'Actor | SAC updates={state.get("actor_updates", "unknown")} | envelope={args.body_envelope} | {env.device}' if agent else
                     f'VR reference + live IK | NOT SAC | actual {env.device} PhysX'))
        video_name = 'policy.mp4' if agent or pose_student or pose_sac else 'reference.mp4'
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
                grasp_conditions=dict(opposing_flaps=bool(grasp.success.opposing_flaps[0]),
                    stable_hands=grasp.stable_hands[0].tolist(),proof_lift=bool(grasp.success.proof_lift[0]),
                    hold_time_s=float(grasp.success.hold_time_s[0]),rack_clearance_m=float(grasp.rack_clearance_m[0]),
                    hand_flap_index=grasp.pinch.hand_flap_index[0].tolist()),
                flap_distances=grasp.matched_flap_distance_m[0].tolist(),
                box_pose=grasp.box_pose_world[0].tolist(), phase=int(teacher.phase[0]) if teacher else None,
                vr_handoff_step=teacher.handoff_index if teacher else None,
                ik_position_errors=[float(s.target_position_error()[0]) for s in teacher.solvers] if teacher else None,
                ik_status=[s.ik_status for s in teacher.solvers] if teacher and int(teacher.phase[0])>0 else None,
                action=action[0].tolist(), rack_peak_force=float(force.max()),
                rack_peak_body=V2_COLLISION_BODY_NAMES[int(force.argmax())],
                box_velocity=grasp.box_velocity_world[0].tolist(),
                base_pose_world=env.scene['robot'].data.root_pose_w[0].tolist(),
                unsafe_causes={key: bool(getattr(safety, key)[0]) for key in (
                    'invalid_box_pose', 'invalid_flap_pose', 'robot_rack_collision',
                    'self_collision', 'obstacle_collision', 'workspace_limit',
                    'box_drop', 'box_lift_limit', 'box_speed_limit')},
                **{key: bool(env.termination_manager.get_term(key)[0]) for key in counts})
            if staged_base:
                row['staged_base']=staged_base.report()
            if staged_contact_ik:
                row['staged_contact_ik']=staged_contact_ik.report()
            if staged_goal_sac:
                row['staged_goal_sac']=dict(actor_updates=staged_goal_sac.actor_updates,
                    critic_updates=staged_goal_sac.critic_updates,
                    radius=staged_goal_sac.radius,training=staged_goal_sac.training)
            if args.contact_diagnostics:
                from kuavo_isaaclab_scene.rl.multi_box.state.isaac_privileged_grasp import MIN_JAW_FORCE_N
                # These are the already measured success inputs, not new
                # sensors or deployable policy observations. Keep each jaw:
                # one strong contact cannot hide a missing opposing contact.
                row['contact_diagnostics']=dict(
                    axis_order='hand_L_R,flap_right_left,jaw_front_back',
                    force_n=grasp.contacts.force_n[0].tolist(),
                    in_region=grasp.contacts.in_region[0].tolist(),
                    opposed=grasp.contacts.opposed[0].tolist(),
                    available=bool(grasp.contacts.available[0]),
                    qualified_flaps=grasp.pinch.qualified_flaps[0].tolist(),
                    ambiguous_hands=grasp.pinch.ambiguous_hands[0].tolist(),
                    assigned_flap_index=grasp.assigned_flap_index[0].tolist(),
                    min_jaw_force_n=MIN_JAW_FORCE_N)
                # Read only the selected two cached body poses. No extra sensor
                # or actor input; preserve measured geometry before auto-reset.
                from kuavo_isaaclab_scene.rl.multi_box.debug.flap_geometry import compare_flap_centers
                adapter=env._multi_box_privileged_grasp
                if bool((grasp.invalid_box_pose|grasp.invalid_flap_pose)[0]):
                    row['flap_geometry_diagnostics']=dict(available=False,invalid_pose=True)
                else:
                    pool=int(grasp.target_pool_id[0]);logical=int(grasp.target_logical_id[0])
                    asset=env.scene[adapter.names[pool]];ids=adapter.flap_ids[pool]
                    panel_pose=torch.cat((asset.data.body_link_pos_w[:1,ids],
                                          asset.data.body_link_quat_w[:1,ids]),-1)
                    from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M
                    kind=env._multi_box_box_type_ids[:1,logical]
                    size=panel_pose.new_tensor(BOX_DIMENSIONS_M[('small','medium')[int(kind[0])]])[None]
                    comparison=compare_flap_centers(grasp.box_pose_world[:1],panel_pose,
                        adapter.flap_centers[pool:pool+1],adapter.flap_normal_axes[pool:pool+1],
                        size,kind,adapter.tcp.center_pose_w[:1])
                    row['flap_geometry_diagnostics']=dict(available=True,
                        axis_order='hand_L_R,flap_right_left',
                        **{key:value[0].tolist() for key,value in comparison.items()})
            history.append(row)
            for key in counts:
                counts[key] += int(row[key])
            if renderer and (row['step'] % args.capture_every == 1 or any(row[key] for key in counts)):
                frame = renderer.frame(env, row['step'], float(grasp.raw.matched_flap_distance_m[0]),
                                       row['success'])
                cv2.rectangle(frame,(10,113),(950,168),(35,35,35),-1)
                cv2.putText(frame, f"pinch L/R={row['pinching']} | success={int(row['success'])}",
                            (20, 140), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1)
                detail=row['grasp_conditions']
                cv2.putText(frame,f"stable={detail['stable_hands']} | opposing={detail['opposing_flaps']} | proof={detail['proof_lift']} | hold={detail['hold_time_s']:.2f}s",
                            (20,160),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
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
                policy_step=step
                staged_previous=None
                if staged_base:
                    robot=env.scene['robot']
                    staged_base.update(pre['policy'],robot.data.root_lin_vel_w,robot.data.root_ang_vel_w,step)
                    if staged_base.phase=='held_grasp':
                        policy_step=staged_base.manipulation_index(step)
                if staged_base and staged_base.phase=='approach':
                    action=staged_base.action(pre['policy'])
                elif staged_goal_sac:
                    action,staged_previous=staged_goal_sac.act(pre['policy'],
                        torch.cat((pre['policy'],pre['critic']),-1),policy_step,
                        sample_frozen_train_behavior=bool(goal_collector and args.train_collection_behavior=='checkpoint-exploration'))
                    if not torch.allclose(projection(pre['policy'],action),action,atol=1e-6,rtol=0):
                        raise ValueError('Staged goal and physical projection differ; replay prohibited')
                elif pose_sac:
                    action,pose_previous=pose_sac.act(pre['policy'],torch.cat((pre['policy'],pre['critic']),-1),policy_step)
                    if not torch.allclose(projection(pre['policy'],action),action,atol=1e-6,rtol=0):
                        raise ValueError('Goal-space and physical gripper projection differ; Q import prohibited')
                elif pose_student:
                    action=projection(pre['policy'],pose_student.act(pre['policy'],policy_step))
                elif residual:
                    raw_critic = torch.cat((pre['policy'], pre['critic']), -1)
                    action, residual_previous = residual.act(pre['policy'], raw_critic, step)
                    if args.residual_zero:
                        action=residual.controller.physical_commands(pre['policy'].new_zeros(1,22),step,pre['policy'])
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
                if staged_base and staged_base.phase=='held_grasp':
                    if staged_contact_ik:
                        action=staged_contact_ik.act(pre['policy'],action,step)
                    action=staged_base.action(pre['policy'],action)
                observation, reward, terminated, truncated, info = env.step(action)
                terminal = info['transition_next_observations']
                if bool(info['transition_numerical_failure'].any()):
                    raise ValueError('Numerical recovery during replay; attempt cannot enter Q')
                if not torch.allclose(reward, env._multi_box_grasp_reward_breakdown.total, atol=1e-5, rtol=1e-5):
                    raise ValueError('Current executed reward differs from reward breakdown')
                if pose_sac and not staged_base:
                    pose_sac.observe(pose_previous,terminal['policy'],torch.cat((terminal['policy'],terminal['critic']),-1),
                                     reward,terminated,step)
                    if pose_sac.training and pose_sac.actor_updates and pose_sac.actor_updates%512==0:pose_sac.save()
                if staged_goal_sac and staged_previous is not None:
                    if goal_collector:
                        following = staged_goal_sac.observations(terminal['policy'],
                            torch.cat((terminal['policy'],terminal['critic']),-1),policy_step+1)
                        goal_collector.append(staged_previous,following,reward,terminated)
                    staged_goal_sac.observe(staged_previous,terminal['policy'],
                        torch.cat((terminal['policy'],terminal['critic']),-1),reward,terminated,policy_step)
                    if staged_goal_sac.training and staged_goal_sac.critic_updates%1024==0:
                        staged_goal_sac.save()
                if residual:
                    residual.observe(residual_previous, terminal['policy'],
                        torch.cat((terminal['policy'], terminal['critic']), -1),
                        reward, terminated, step)
                    if args.residual_training and residual.actor_updates and residual.actor_updates % 512 == 0:
                        residual.save()
                row = history[-1]
                row['reward']=float(reward[0])
                row['reward_terms']={key:float(value[0]) for key,value in
                                     env._multi_box_grasp_reward_breakdown.terms.items()}
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
                    if frames == 0 or any(row[key] for key in counts):
                        cv2.imwrite(str(output/'preview.png'), frame)
                    frames += 1
                if step % 30 == 0:
                    (output/'progress.json').write_text(json.dumps(row))
                    print(f"[REFERENCE] step={step+1} pinch={row['pinching']} success={row['success']}", flush=True)
                    if residual:
                        print('[RESIDUAL SAC] '+json.dumps(residual.report()), flush=True)
                    if pose_sac and not staged_goal_sac:
                        print('[POSE GOAL SAC] '+json.dumps(pose_sac.report()),flush=True)
                    if staged_goal_sac:
                        print('[STAGED GOAL SAC] '+json.dumps(staged_goal_sac.report()),flush=True)
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
        if pose_sac and pose_sac.training:pose_sac.save(final=True)
        if staged_goal_sac and staged_goal_sac.training:staged_goal_sac.save(final=True)
        if residual and args.residual_training:
            residual.save(final=True)
            (output/'manifest.json').write_text(json.dumps(contract | {
                'artifact_type': artifact_type, 'layout':layout.record() if layout else None,
                'residual_contract': residual.contract}, indent=2)+'\n')
        video_encoding = None
        if writer is not None:
            writer.release()
            writer = None
            if frames:
                from browser_video import encode_browser_video
                video_encoding = encode_browser_video(output/video_name)
            else:
                # A normal stop during initialization can create an empty
                # container. There is no video to transcode or publish.
                (output/video_name).unlink(missing_ok=True)
        report = dict(policy=controller_name,
                      steps=len(history), outcomes=counts, frames=frames, history=history,
                      initial_settling_steps=settling_steps, sim_device=str(env.device),
                      interrupted=stopped['value'], completed_attempt=bool(sum(counts.values())),
                      video=str(output/video_name) if renderer and frames else None,
                      video_encoding=video_encoding,
                      body_envelope_diagnostic=args.body_envelope,
                      actor_reference_mix=args.actor_reference_mix, correction_label_rows=len(label_actions),
                      diagnostic_joint_offset_rad=args.joint_offset_rad,
                      initial_actor_error=initial_actor_error,
                      checkpoint_actor_updates=state.get('actor_updates') if state else None,
                      checkpoint_actor_refit=state.get('diagnostic_actor_refit') if state else None,
                      layout=layout.record() if layout else None, zero_residual_probe=args.residual_zero,
                      retarget=residual.controller.retarget_report if residual and args.residual_controller=='retargeted-goal' else None)
        report['initial_base_pose_world']=initial_base_pose
        if staged_base:report['staged_base_contract']=staged_base.report()
        if staged_contact_ik:report['staged_contact_ik_contract']=staged_contact_ik.report()
        report['initial_rack_pose_world']=initial_rack_pose
        report['initial_base_rack_observation']=initial_actor[68:77].tolist()
        report['initial_active_box_tokens']=initial_actor[86:350].reshape(12,22)[
            initial_actor[86:350].reshape(12,22)[:,0]>.5].tolist()
        report['reward_term_sums']={key:sum(row['reward_terms'][key] for row in history)
                                    for key in history[0]['reward_terms']} if history else {}
        report['vr_orientation_mode']=args.vr_orientation_mode
        report['vr_contact_torso_forward_m']=args.vr_contact_torso_forward_m
        report['vr_contact_torso_up_m']=args.vr_contact_torso_up_m
        report['torso_extra_height_m']=args.torso_extra_height_m
        report['vr_layout_retarget']=args.vr_layout_retarget
        report['vr_handoff_distance_m']=args.vr_handoff_distance_m
        report['vr_handoff_step']=teacher.handoff_index if teacher else None
        report['vr_contact_goal']=args.vr_contact_goal
        report['source_flap_prior_audit']=demo_audit
        report['vr_retarget']=vr_retarget if args.vr_layout_retarget else None
        report['physical_action_contract']=contract['action_contract']
        report['contact_diagnostics']=args.contact_diagnostics
        report['vr_close_distance_m']=args.vr_close_distance_m
        report['vr_coordinated_close']=args.vr_coordinated_close
        report['vr_reference_grippers']=args.vr_reference_grippers
        report['vr_contact_rest_mode']=args.vr_contact_rest_mode
        report['vr_contact_base_forward_m']=args.vr_contact_base_forward_m
        if pose_student:
            report['pose_student']=dict(actor_fit_steps=pose_student.state['actor_fit_steps'],
                sac_actor_updates=0,sac_critic_updates=0,runtime_reference_path_required=False,
                episode_clock_input=True,artifact_type=pose_student.artifact_type)
        if residual:
            report['residual_sac'] = residual.report()
        if pose_sac:
            report['frozen_goal_warm_start' if staged_goal_sac else 'pose_goal_sac']=pose_sac.report()
        if staged_goal_sac:report['staged_goal_sac']=staged_goal_sac.report()
        if goal_collector:
            if staged_goal_sac.training or collection_initial_counters != (
                    staged_goal_sac.actor_updates, staged_goal_sac.critic_updates):
                raise ValueError('Frozen TRAIN collection unexpectedly updated the policy or critic')
            last = history[-1] if history else {}
            conditions = last.get('grasp_conditions',{})
            outcome = dict(wave=0,split='train',environment=0,layout=layout.record(),
                complete=report['completed_attempt'] and not report['interrupted'],
                initial_layout_valid=bool(staged_initial_valid[0]),
                result=dict(success=bool(last.get('success')),unsafe=bool(last.get('unsafe')),
                    invalid_reset=bool(last.get('invalid_reset')),time_out=bool(last.get('time_out')),
                    unsafe_causes=last.get('unsafe_causes',{}),
                    pinching=last.get('pinching',[]),stable_hands=conditions.get('stable_hands',[]),
                    opposing_flaps=conditions.get('opposing_flaps',False),proof_lift=conditions.get('proof_lift',False),
                    hold_time_s=conditions.get('hold_time_s',0),rack_clearance_m=conditions.get('rack_clearance_m',0),
                    staged_base=staged_base.report()))
            report['training_goal_collection'] = goal_collector.save(output,outcome,
                actor_updates=staged_goal_sac.actor_updates,critic_updates=staged_goal_sac.critic_updates,
                completed=report['completed_attempt'],interrupted=report['interrupted'])
        (output/'metrics.json').write_text(json.dumps(report, indent=2)+'\n')
        (output/'status.json').write_text(json.dumps({'status':'stopped' if stopped['value'] else 'complete',
            'outcomes':counts,'actor_updates':staged_goal_sac.actor_updates if staged_goal_sac else
                pose_sac.actor_updates if pose_sac else residual.actor_updates if residual else None})+'\n')
        print(json.dumps({key: value for key, value in report.items() if key != 'history'}), flush=True)
    except BaseException as error:
        # Kit shutdown can replace Python's nonzero exit and suppress the
        # uncaught traceback. Preserve the failure before closing the app.
        import traceback
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'failure.json').write_text(json.dumps(
            {'phase': 'failed', 'error': type(error).__name__, 'reason': str(error)})+'\n')
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
