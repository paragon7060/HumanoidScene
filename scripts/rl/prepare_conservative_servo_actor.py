#!/usr/bin/env python3
"""Fork a pristine retained-servo SAC, changing only its actor learning rate."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import torch

from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import require_pristine_learning_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetentionGentleSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import (
    ACTOR_LR, BASE_ACTOR_LR, CRITIC_ALPHA_LR, ConservativeServoRetentionSACPilot,
    conservative_actor_contract, validate_conservative_actor_state,
)


def prepare(initial, experience):
    require_pristine_learning_state(initial, experience)
    if initial.get('artifact_type') != ServoRetentionGentleSACPilot.artifact_type \
            or (initial['config'].get('actor_lr'), initial['config'].get('lr')) != (BASE_ACTOR_LR, CRITIC_ALPHA_LR):
        raise ValueError('Pristine retained-servo initialization with reviewed learning rates required')
    if any(not o.get('param_groups') or any(g.get('lr') != rate for g in o['param_groups'])
            for o, rate in zip(initial['optimizers'], (BASE_ACTOR_LR, *([CRITIC_ALPHA_LR] * 3)))):
        raise ValueError('Source optimizer learning rates differ')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import body_behavior_statistics
    if any(value.get('body_behavior_statistics') != body_behavior_statistics()
           for value in (initial, experience)):
        raise ValueError('Collection must not have started')
    state, replay = deepcopy(initial), deepcopy(experience)
    state['artifact_type'] = ConservativeServoRetentionSACPilot.artifact_type
    state['config']['actor_lr'] = ACTOR_LR
    for group in state['optimizers'][0]['param_groups']:
        group['lr'] = ACTOR_LR
    for value in (state, replay):
        value['goal_contract']['name'] = state['artifact_type']
        value['goal_contract']['actor_update_step'] = conservative_actor_contract()
    validate_conservative_actor_state(state)
    return state, replay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint', 'training-manifest', 'waypoints', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_experience = args.initial_checkpoint.parent / 'staged_goal_experience.pt'
    sources = (args.initial_checkpoint, source_experience, args.training_manifest, args.waypoints)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_uid != os.getuid() for p in sources):
        raise ValueError('Owned regular pristine input files required')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    original = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    experience = torch.load(source_experience, map_location='cpu', weights_only=True)
    physical = json.loads(args.training_manifest.read_text())
    contract = original['goal_contract']['physical_contract']
    if {k: physical.get(k) for k in contract} != contract:
        raise ValueError('Physical manifest differs')
    state, replay = prepare(original, experience)
    assert identical(state['model'], original['model'])
    assert identical(state['optimizers'][1:], original['optimizers'][1:])
    assert identical(replay['executed_goal_transitions'], experience['executed_goal_transitions'])
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, args.output_dir / 'checkpoint_00000000.pt')
    torch.save(replay, args.output_dir / 'staged_goal_experience.pt')
    assert identical(state, torch.load(args.output_dir / 'checkpoint_00000000.pt', weights_only=True))
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    proof = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        actor_learning_rate_before=BASE_ACTOR_LR, actor_learning_rate_after=ACTOR_LR,
        critic_alpha_rates_models_targets_normalizers_Gaussian_collection_unchanged=True,
        all_four_optimizer_states_and_online_success_return_banks_empty=True,
        physical_reward_safety_success_randomization_preserved=True,
        no_old_Q_reward_or_evaluation_data_imported=True, physical_training_NOT_started=True,
        goal_not_complete=True)
    for name, value in [('training_manifest.json', physical),
            ('waypoints.json', json.loads(args.waypoints.read_text())),
            ('initialization_verification.json', proof),
            ('manifest.json', dict(artifact_type=state['artifact_type'], goal_contract=state['goal_contract'],
                                  initialization_verification=proof))]:
        (args.output_dir / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(output_dir=str(args.output_dir), actor_lr=ACTOR_LR,
        all_initial_model_tensors_preserved=True, physical_training_NOT_started=True)))


if __name__ == '__main__':
    main()
