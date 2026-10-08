#!/usr/bin/env python3
"""Prepare a fresh bounded URDF-goal SAC; reuse an actor-only regional source."""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from compare_actor_train_memory import owned_stable_bytes
from export_eval_q_videos import restored_agent
from prepare_actual_success_actor_tail import identical
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations
from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import BatchedBaseStages
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import GraspLayout, layout_reset_observation
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_prior_lift_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_regional_goal_sac import (
    URDFRegionalGoalSACPilot, regional_source_snapshot, validate_urdf_regional_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_servo_guard_sac import (
    URDFServoGuardSACPilot, validate_urdf_servo_guard_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_full_arm_sac import (
    URDFFullArmSACPilot, validate_full_arm_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import (
    URDFStrongSuccessSACPilot, validate_strong_success_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_perceived_contact_sac import (
    URDFPerceivedContactSACPilot, URDFSettledContactSACPilot, validate_perceived_contact_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import (
    GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import frozen_actor_reward_contract
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for key in ('source-checkpoint', 'training-manifest', 'waypoints', 'waves-json', 'demo-dataset', 'output-dir'):
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--native-seed', type=Path, action='append', required=True)
    parser.add_argument('--replay-capacity', type=int, default=2000000)
    parser.add_argument('--servo-retention-profile',
        choices=('uniform', 'guard-tail64', 'full-arm-tail64', 'full-arm-tail64-strong-servo',
                 'full-arm-contact-exploration','full-arm-settled-contact'),
        default='uniform')
    parser.add_argument('--body-behavior', choices=(GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT),
        default=GREEDY_REST_VARIANT, help='Fresh TRAIN collection bias; policy/evaluation and original defaults are preserved')
    args = parser.parse_args()
    if args.servo_retention_profile in ('full-arm-contact-exploration','full-arm-settled-contact') and args.body_behavior != GENTLE_GREEDY_REST_VARIANT:
        parser.error('Contact exploration requires explicit --body-behavior arm20-gentle-rest-greedy')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('Initialization is CPU-only; CUDA_VISIBLE_DEVICES must be empty')
    torch.set_num_threads(1)
    if args.output_dir.exists():
        raise ValueError('A unique nonexistent output directory is required')
    inputs = (args.source_checkpoint, args.training_manifest, args.waypoints, args.waves_json, args.demo_dataset, *args.native_seed)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_uid != os.getuid() for p in inputs):
        raise ValueError('Owned regular initialization inputs required')
    blob = owned_stable_bytes(args.source_checkpoint)
    source_SHA256 = hashlib.sha256(blob).hexdigest()
    source = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    packet = regional_source_snapshot(source, source_SHA256)
    physical = json.loads(args.training_manifest.read_text())
    waypoints = json.loads(args.waypoints.read_text())
    waves = json.loads(args.waves_json.read_text())
    if source['goal_contract']['physical_contract'] != {k: physical.get(k) for k in source['goal_contract']['physical_contract']}:
        raise ValueError('Original physical manifest differs')
    batch, _ = load_v2_grasp_demonstrations(args.demo_dataset, self_collision_enabled=False)
    references = {i: select_reference_episode(batch, i)['actor_obs'][0] for i in
        {r['episode_index'] for r in waves[0]['layouts']}}
    raw = torch.stack([layout_reset_observation(references[r['episode_index']], GraspLayout(**r['layout']), MultiBoxSpec())
        for r in waves[0]['layouts']])
    warm = PoseGoalSACPilot(source['frozen_warm_start'], args.native_seed,
        frozen_prior_lift_contract(frozen_actor_reward_contract(physical)), args.output_dir, training=False, device='cpu')
    stages = BatchedBaseStages(warm.coordinates, waypoints, raw)
    pilot_class, validator = {
        'uniform': (URDFRegionalGoalSACPilot, validate_urdf_regional_state),
        'guard-tail64': (URDFServoGuardSACPilot, validate_urdf_servo_guard_state),
        'full-arm-tail64': (URDFFullArmSACPilot, validate_full_arm_state),
        'full-arm-tail64-strong-servo': (URDFStrongSuccessSACPilot, validate_strong_success_state),
        'full-arm-contact-exploration': (URDFPerceivedContactSACPilot, validate_perceived_contact_state),
        'full-arm-settled-contact': (URDFSettledContactSACPilot,
            lambda state:validate_perceived_contact_state(state,settled_close=True)),
    }[args.servo_retention_profile]
    pilot = pilot_class(warm, physical, args.output_dir, stages.stages[0],
        regional_source=packet, body_anchor_state=source['body_anchor_state'], training=False, device='cpu',
        replay_capacity=args.replay_capacity, train_success_retention=True, exploration_correlation=.98,
        measured_train_credit='measured-episode-return', jaw_behavior='joint-epsilon30',
        jaw_saturation='logit4-soft-strong', body_behavior=args.body_behavior,
        body_saturation='mean3-soft', critic_episode_clock='task-remaining')
    assert pilot.actor_updates == pilot.critic_updates == pilot.online_rows == pilot.replay.size == 0
    assert pilot.success_bank.size == pilot.measured_credit_bank.size == 0
    assert all(not o.state for o in pilot.agent.optimizers)
    args.output_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    pilot.save(final=True)
    checkpoint = args.output_dir / 'checkpoint_00000000.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    validator(state)
    assert state['goal_contract'] == pilot.contract and identical(state['model'], pilot.agent.state_dict())
    agent, _ = restored_agent(state)
    assert identical(agent.state_dict(), pilot.agent.state_dict())
    # The real runtime restoration must consume the new empty replay and
    # contract, not an old normalized-action bank or learned source Q.
    resumed = pilot_class(warm, physical, args.output_dir, stages.stages[0],
        checkpoint=checkpoint, training=True, device='cpu')
    assert resumed.contract == pilot.contract and identical(resumed.agent.state_dict(), state['model'])
    assert resumed.replay.size == resumed.success_bank.size == resumed.measured_credit_bank.size == 0
    assert identical([o.state_dict() for o in resumed.agent.optimizers], state['optimizers'])
    assert hashlib.sha256(owned_stable_bytes(args.source_checkpoint)).hexdigest() == source_SHA256
    proof = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source_checkpoint_SHA256=source_SHA256,
        source_actor_only=True, source_Q_replay_rewards_entropy_optimizers_or_teacher_imported=False,
        actor518_critic578_action21_and_midpoint_perception=True,
        arm_URDF_goals_and_original_non_arm_coordinates=True,
        exact_initial_checkpoint_and_training_resume_model_optimizer_contract=True,
        frozen_Q_video_restoration_exact=True, actor_Q_online_replay_success_return_banks0=True,
        replay_capacity=pilot.replay.capacity, original_task_physics_randomization_reward_success_safety_preserved=True,
        physical_simulation_or_training_NOT_started=True, no_physical_success_or_improvement_claim=True,
        servo_retention_profile=args.servo_retention_profile,
        body_behavior_variant=args.body_behavior,
        independent_FINAL_unused=True, goal_not_complete=True)
    for name, value in (('training_manifest.json', physical), ('waypoints.json', waypoints),
            ('training_waves.json', waves), ('initialization_verification.json', proof),
            ('status.json', dict(status='complete', initialized_not_trained=True)),
            ('manifest.json', dict(artifact_type=state['artifact_type'], goal_contract=state['goal_contract'], initialization_verification=proof))):
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output_dir=str(args.output_dir), proof=proof)))


if __name__ == '__main__':
    main()
