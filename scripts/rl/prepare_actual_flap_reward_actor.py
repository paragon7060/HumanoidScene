#!/usr/bin/env python3
"""Move an existing actual-flap actor to precise capture with fresh learning state."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training-manifest', 'waypoints', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--native-seed', type=Path, action='append', required=True)
    parser.add_argument('--initialization-seed', type=int, default=20261006)
    parser.add_argument('--body-controller', choices=('source', 'reanchored30'), default='source',
        help='Reanchored30 starts fresh .30 corrections around the learned actual-flap body goals')
    args = parser.parse_args()
    if args.output_dir.exists() or not all(p.is_file() for p in (
            args.checkpoint, args.training_manifest, args.waypoints, *args.native_seed)):
        parser.error('Matching existing inputs and a unique output directory are required')
    if not 0 <= args.initialization_seed < 2**32:
        parser.error('Initialization seed must be within0..2**32-1')
    import h5py
    import torch
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import (
        ReanchoredActualFlapSACPilot, actual_actor_anchor_state)
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_prior_lift_contract, require_current_lift_contract
    from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import frozen_actor_reward_contract
    from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import with_precision_capture_profile
    torch.set_num_threads(1)
    torch.manual_seed(args.initialization_seed)
    source_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    physical = json.loads(args.training_manifest.read_text())
    require_current_lift_contract(physical)
    if source.get('artifact_type') != ActualFlapResidualSACPilot.artifact_type:
        raise ValueError('Precise reward initialization requires an existing actual-flap actor')
    goal = source['goal_contract']
    if goal['physical_contract'] != {k: physical[k] for k in goal['physical_contract']}:
        raise ValueError('Source physical contract and manifest differ')
    updated = with_precision_capture_profile(physical)
    warm = PoseGoalSACPilot(source['frozen_warm_start'], args.native_seed,
        frozen_prior_lift_contract(frozen_actor_reward_contract(updated)),
        args.output_dir, training=False, device='cpu')
    templates = json.loads(args.waypoints.read_text())
    if templates.get('physical_action_contract') != updated['action_contract']:
        raise ValueError('Reward transfer must keep the measured waypoint travel contract')
    with h5py.File(args.native_seed[0], 'r') as stream:
        raw = torch.tensor(next(iter(stream['episodes'].values()))['transitions/actor_obs'][:1])
    stage = StagedBaseHoldDiagnostic(warm.coordinates, templates, raw)
    if stage.templates != goal['shelf_templates']:
        raise ValueError('Reward transfer must keep all measured regional workplaces')
    reanchored = args.body_controller == 'reanchored30'
    pilot_class = ReanchoredActualFlapSACPilot if reanchored else ActualFlapResidualSACPilot
    pilot = pilot_class(warm, updated, args.output_dir, stage,
        body_anchor_state=actual_actor_anchor_state(source) if reanchored else source['body_anchor_state'], device='cpu',
        replay_capacity=goal['replay_capacity'], actor_min_replay_rows=goal['actor_min_replay_rows'],
        exploration_correlation=goal['exploration_correlation'], train_success_retention=True)
    if asdict(pilot.agent.config) != source['config']:
        raise ValueError('Reward transfer must preserve the reviewed SAC configuration')
    fresh_learning = {k: v.clone() for k, v in pilot.agent.state_dict().items()
                      if not k.startswith(('actor.', 'actor_normalizer.'))}
    if reanchored:
        migration = dict(source_artifact_type=source['artifact_type'],
            destination_artifact_type=pilot.artifact_type, source_actor_only_imported=True,
            body_controller=args.body_controller, old_correction_radius=.15,
            new_correction_radius=.30, source_actual_body_goal_is_new_frozen_anchor=True,
            nineteen_body_mean_output_rows_zeroed=True,
            source_trunk_jaw_rows_logstd_and_actor_normalizer_preserved=True)
    else:
        migration = initialize_staged_actor_only(pilot, source, allow_contact_reward_change=True)
    # The independent comparator may read old Q tensors, but its replay and
    # optimizers are never imported into the destination learner.
    if reanchored:
        from export_eval_q_videos import restored_agent
        original, _ = restored_agent(source)
    else:
        original = pilot.agent_class(pilot.actor_dim, pilot.critic_dim, 21,
            SACConfig(**source['config']), 'cpu', action_projector=pilot.agent.action_projector,
            validated_jaw_prior_confidence=pilot.validated_jaw_prior_confidence,
            jaw_prior_residual_gain=pilot.jaw_prior_residual_gain)
        original.correction_radius = goal['body_correction_radius']
        original.executed_body_anchor = pilot.agent.executed_body_anchor
        original.validated_jaw_prior = pilot.agent.validated_jaw_prior
        original.restore(source, training=False)
    inputs = [e['rows']['actor_obs'][::20] for episodes in
              source.get('successful_train_transitions', {}).get('episodes', {}).values() for e in episodes]
    if not inputs:
        raise ValueError('Literal successful TRAIN inputs are required for control identity verification')
    inputs = torch.cat(inputs)
    new_inputs = inputs.clone()
    new_inputs[:, -1] = pilot.correction_radius
    with torch.no_grad():
        old, new = original.act(inputs, True), pilot.agent.act(new_inputs, True)
    if not torch.equal(old, new):
        raise ValueError('Fresh reward actor changed its executed source goals or jaws')
    if any(not torch.equal(v, pilot.agent.state_dict()[k]) for k, v in fresh_learning.items()) \
            or pilot.actor_updates or pilot.critic_updates or pilot.online_rows or pilot.replay.size \
            or pilot.success_bank.size or pilot.agent.critic_normalizer.count \
            or any(opt.state for opt in pilot.agent.optimizers):
        raise ValueError('Reward initialization imported old learning state')
    if not all(torch.isfinite(v).all() for v in pilot.agent.state_dict().values()) \
            or source_sha != hashlib.sha256(args.checkpoint.read_bytes()).hexdigest():
        raise ValueError('Nonfinite model or changed source checkpoint')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
    # Actor inputs are selected features, not the original464-D observation.
    # Decode uses only the recorded proprioception and pending-PD telemetry;
    # unpack those exact retained columns. Other sensor columns are unused by
    # this command comparison and are never fed to a policy or replay.
    features = warm.coordinates.features
    if features.mode != 'grasp_target_no_history' or features.tail_end != 440:
        raise ValueError('Unknown retained controller telemetry for command verification')
    telemetry = inputs.new_zeros(len(inputs), 464)
    telemetry[:, :features.token_start] = inputs[:, :features.token_start]
    tail_start = features.token_start + features.token_width + 38
    telemetry[:, features.tail_start:features.tail_end] = inputs[:,
        tail_start:tail_start + features.tail_end - features.tail_start]
    measured_stage = SimpleNamespace(phase='held_grasp', target_xy=inputs[:, -5:-3],
        target_yaw=torch.atan2(inputs[:, -3], inputs[:, -2]))
    physical_old = held_goal_coordinates(warm.coordinates, telemetry,
        pilot.center + pilot.scale*old, measured_stage)
    physical_new = held_goal_coordinates(warm.coordinates, telemetry,
        pilot.center + pilot.scale*new, measured_stage)
    if not torch.equal(physical_old, physical_new):
        raise ValueError('Fresh controller changed its decoded initial physical commands')
    proof = migration | dict(source_actor_updates=source['actor_updates'],
        source_critic_updates_not_imported=source['critic_updates'],
        source_checkpoint_SHA256=source_sha, actual_TRAIN_input_rows=len(inputs),
        greedy_goals_bit_identical=True, greedy_body_error_max=0., executed_jaw_disagreement_count=0,
        decoded_physical_commands_bit_identical=True,
        fresh_Q_target_Q_entropy_critic_normalizer_optimizers_replay_and_success_bank=True,
        fresh_non_actor_model_tensors_preserved=len(fresh_learning),
        original_waypoints_goal_coordinates_physics_randomization_and_safety_preserved=True,
        source_checkpoint_unchanged=True, initialized_not_trained=True, goal_not_complete=True)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pilot.save(final=True)
    for name, value in (
            ('training_manifest.json', updated), ('waypoints.json', templates),
            ('initialization_verification.json', proof),
            ('manifest.json', dict(artifact_type=pilot.artifact_type, training_contract=updated,
                goal_contract=pilot.contract, initialization_verification=proof,
                initialization_seed=args.initialization_seed,
                created_utc=datetime.now(timezone.utc).isoformat())),
            ('status.json', dict(status='complete', initialized_not_trained=True))):
        (args.output_dir / name).write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()), verification=proof)))


if __name__ == '__main__':
    main()
