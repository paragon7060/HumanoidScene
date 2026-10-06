#!/usr/bin/env python3
"""Fork fresh actual-flap learning inputs, changing only future arm collection."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training-manifest', 'waypoints', 'verification-inputs-checkpoint', 'output-dir'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    experience = args.checkpoint.parent/'staged_goal_experience.pt'
    sources = [args.checkpoint, experience, args.training_manifest, args.waypoints,
        args.verification_inputs_checkpoint]
    if args.output_dir.exists() or not all(p.is_file() for p in sources):
        parser.error('Existing closed inputs and a unique output directory required')
    import torch
    from export_eval_q_videos import restored_agent
    from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import enable_body_behavior
    torch.set_num_threads(1)
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    actual = torch.load(experience, map_location='cpu', weights_only=True)
    physical = json.loads(args.training_manifest.read_text())
    goal = source['goal_contract']
    if goal['physical_contract'] != {k: physical[k] for k in goal['physical_contract']}:
        raise ValueError('Source checkpoint and physical manifest differ')
    if source['actor_updates'] or source['critic_updates'] \
            or len(actual['executed_goal_transitions']['reward']) \
            or any(opt['state'] for opt in source['optimizers']) \
            or any(source['successful_train_transitions']['episodes'].values()) \
            or source['model']['critic_normalizer.count']:
        raise ValueError('Controlled comparison requires fresh counters, optimizers, replay and success bank')
    destination, updated = enable_body_behavior(source, actual, source_checkpoint=args.checkpoint)
    # This closed historical checkpoint supplies measured states for identity
    # checks only. Its Q, replay and optimizers never enter the new learner.
    verification = torch.load(args.verification_inputs_checkpoint, map_location='cpu', weights_only=True)
    inputs = [e['rows']['actor_obs'][::20] for episodes in
        verification.get('successful_train_transitions', {}).get('episodes', {}).values() for e in episodes]
    if not inputs:
        raise ValueError('Measured TRAIN states required to verify control identity')
    inputs = torch.cat(inputs)
    original, _ = restored_agent(source)
    new, _ = restored_agent(destination)
    with torch.no_grad():
        before, after = original.act(inputs, True), new.act(inputs, True)
    if not torch.equal(before, after):
        raise ValueError('Greedy body/jaw commands changed')
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    target = args.output_dir/'checkpoint_00000000.pt'
    torch.save(destination, target)
    torch.save(updated, args.output_dir/'staged_goal_experience.pt')
    reloaded = torch.load(target, map_location='cpu', weights_only=True)

    def identical(a, b):
        if isinstance(a, torch.Tensor): return isinstance(b, torch.Tensor) and torch.equal(a, b)
        if isinstance(a, dict): return isinstance(b, dict) and a.keys()==b.keys() and all(identical(a[k],b[k]) for k in a)
        if isinstance(a, (tuple, list)): return type(a) is type(b) and len(a)==len(b) and all(identical(x,y) for x,y in zip(a,b))
        return a==b

    if any(not identical(value, reloaded[key]) for key, value in source.items()):
        raise ValueError('Original checkpoint fields changed during collection fork')
    old_replay = torch.load(args.output_dir/'staged_goal_experience.pt', weights_only=True)
    if any(not identical(value, old_replay[key]) for key, value in actual.items()):
        raise ValueError('Original replay fields changed during collection fork')
    if any(digest != hashlib.sha256(Path(path).read_bytes()).hexdigest() for path,digest in hashes.items()):
        raise ValueError('A source changed during verification')
    tensors = sum(len(d) for d in (source['model'], source['body_anchor_state']['model'], source['frozen_actor_prior']))
    all_command_tensors = tensors + len(source['frozen_warm_start']['model']) \
        + len(source['frozen_warm_start']['bc_prior']['model'])
    proof = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_SHA256=hashes,actual_TRAIN_input_rows=len(inputs),
        greedy_goals_bit_identical=True,greedy_body_error_max=0.,executed_jaw_disagreement_count=0,
        model_anchor_and_prior_tensors_bit_identical=tensors,
        all_command_network_and_normalizer_tensors_bit_identical=all_command_tensors,
        all_original_checkpoint_and_replay_fields_bit_identical=True,
        four_optimizer_states_preserved=True,new_actor_Q_updates_initially0=True,
        no_historical_Q_replay_optimizer_or_success_bank_import=True,
        only_future_TRAIN_arm_behavior_enabled=True,source_files_unchanged=True,
        initialized_not_trained=True,goal_not_complete=True)
    for name, value in (
        ('training_manifest.json',physical), ('waypoints.json',json.loads(args.waypoints.read_text())),
        ('initialization_verification.json',proof),
        ('manifest.json',dict(artifact_type=destination['artifact_type'],training_contract=physical,
            goal_contract=goal,TRAIN_body_behavior=destination['body_behavior'],initialization_verification=proof)),
        ('status.json',dict(status='complete',initialized_not_trained=True))):
        (args.output_dir/name).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir.resolve()),verification=proof)))


if __name__ == '__main__': main()
