#!/usr/bin/env python3
"""Change goal-SAC exploration while preserving mean policy and actual Q data."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def fork_checkpoint(checkpoint, output_dir, *, initial_std=.001, min_std=.0001,
                    max_std=.003, actor_lr=1e-6, demo_fade_updates=20000):
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    if (not all(math.isfinite(v) for v in (min_std, initial_std, max_std, actor_lr))
            or not 0 < min_std <= initial_std <= max_std <= 1
            or actor_lr <= 0 or demo_fade_updates < 1):
        raise ValueError('Invalid exploration or fade settings')

    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (state.get('artifact_type') != 'pose_goal_sac_no_live_reference'
            or state.get('format_version') != 1):
        raise ValueError('Only a native goal-SAC checkpoint may be forked')
    source_manifest = json.loads((checkpoint.parent / 'manifest.json').read_text())
    old_config, old_contract = dict(state['config']), dict(state['goal_contract'])
    state['config'].update(min_policy_std=min_std, initial_policy_std=initial_std,
                           max_policy_std=max_std, actor_lr=actor_lr)
    state['goal_contract'].update(actor_lr=actor_lr, demo_fade_updates=demo_fade_updates,
                                 prior_fade_updates=demo_fade_updates)

    # Only Gaussian variance-head rows change. The deterministic mean function,
    # decoder, Q networks and normalization remain exactly identical.
    width = state['action_dim']
    state['model']['actor.network.4.weight'][width:].zero_()
    state['model']['actor.network.4.bias'][width:].fill_(math.log(initial_std))
    state['optimizers'][0]['state'] = {}
    for group in state['optimizers'][0]['param_groups']:
        group['lr'] = actor_lr
    state['entropy_contract']['target_per_dim'] = min(
        -1., math.log(max_std) + .5 * math.log(2 * math.pi * math.e) - .5)
    audit = dict(source_checkpoint=str(checkpoint.resolve()),
                 source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 deterministic_mean_unchanged=True,
                 goal_decoder_and_Q_coordinates_unchanged=True,
                 critic_and_normalizers_unchanged=True,
                 actor_optimizer_moments_reset=True,
                 source_policy_config=old_config, new_policy_config=state['config'])
    state['policy_only_fork_audit'] = audit

    actual = None
    prior_experience = checkpoint.parent / 'pose_goal_experience.pt'
    if prior_experience.exists():
        actual = torch.load(prior_experience, map_location='cpu', weights_only=True)
        if actual['goal_contract'] != old_contract:
            raise ValueError('Source actual goal experience contract differs')
        # Keep the collection policy for provenance. State/action/reward tensors
        # are copied without any alteration or invented rows.
        actual['collection_goal_contract'] = old_contract
        actual['collection_policy_config'] = old_config
        actual['goal_contract'] = state['goal_contract']
        actual['policy_only_migration'] = True

    # Validate inputs before creating a unique destination or writing a model.
    output_dir.mkdir(parents=True, exist_ok=False)
    destination = save_checkpoint(output_dir, state, state['actor_updates'], keep=None)
    if actual is not None:
        torch.save(actual, output_dir / 'pose_goal_experience.pt')
    manifest = source_manifest | dict(goal_contract=state['goal_contract'],
                                     policy_only_fork_audit=audit)
    (output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output_dir / 'status.json').write_text(json.dumps(dict(
        status='complete', actor_updates=state['actor_updates'],
        physical_performance_after_exploration_change_not_yet_measured=True)) + '\n')
    return destination, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--initial-std', type=float, default=.001)
    parser.add_argument('--min-std', type=float, default=.0001)
    parser.add_argument('--max-std', type=float, default=.003)
    parser.add_argument('--actor-lr', type=float, default=1e-6)
    parser.add_argument('--demo-fade-updates', type=int, default=20000)
    args = parser.parse_args()
    try:
        _, audit = fork_checkpoint(args.checkpoint, args.output_dir,
                                  initial_std=args.initial_std, min_std=args.min_std,
                                  max_std=args.max_std, actor_lr=args.actor_lr,
                                  demo_fade_updates=args.demo_fade_updates)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(audit | {'source_policy_config': None, 'new_policy_config': None}))


if __name__ == '__main__':
    main()
