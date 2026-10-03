#!/usr/bin/env python3
"""Fork goal-SAC exploration or discount while preserving its actor and real data."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from kuavo_isaaclab_scene.rl.runners.storage import save_checkpoint


def fork_checkpoint(checkpoint, output_dir, *, initial_std=.001, min_std=.0001,
                    max_std=.003, actor_lr=None, demo_fade_updates=20000,
                    align_discount_with=None, critic_warmup_updates=2000,
                    preserve_exploration=False, prior_weight_floor=None,
                    max_prior_deviation=None, prior_initial_weight=None,
                    actor_update_interval=None):
    checkpoint, output_dir = Path(checkpoint), Path(output_dir)
    requested_actor_lr = actor_lr
    actor_lr = 1e-6 if actor_lr is None else actor_lr
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
    if actor_update_interval is not None:
        if type(actor_update_interval) is not int or not 1<=actor_update_interval<=256:
            raise ValueError('Actor update interval must be an integer within1..256')
        old_interval=old_contract.get('actor_update_interval',1)
        if actor_update_interval != old_interval:
            if actor_update_interval==1:
                state['goal_contract'].pop('actor_update_interval',None)
            else:
                state['goal_contract']['actor_update_interval']=actor_update_interval
                state['goal_contract'].setdefault('demo_fade_critic_offset',
                    state['critic_updates']-state['actor_updates'])
    initial_weight = old_contract.get('frozen_network_prior_initial_weight', 10.)
    if prior_initial_weight is not None:
        if not math.isfinite(prior_initial_weight) or prior_initial_weight < 0:
            raise ValueError('Prior initial weight must be finite and nonnegative')
        initial_weight = prior_initial_weight
        state['goal_contract']['frozen_network_prior_initial_weight'] = initial_weight
    floor = (old_contract.get('frozen_network_prior_weight_floor', 0.)
             if prior_weight_floor is None else prior_weight_floor)
    if not math.isfinite(floor) or not 0 <= floor <= initial_weight:
        raise ValueError('Prior weight floor must be finite and within the initial weight')
    if prior_weight_floor is not None:
        if prior_weight_floor:state['goal_contract']['frozen_network_prior_weight_floor']=prior_weight_floor
        else:state['goal_contract'].pop('frozen_network_prior_weight_floor',None)
    if max_prior_deviation is not None:
        if not math.isfinite(max_prior_deviation) or not 0<=max_prior_deviation<=1:
            raise ValueError('Prior goal radius must be finite and within0..1')
        from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
        projection=GoalGripperProjector.name
        if max_prior_deviation:
            projection += '_neural_prior_radius_'+format(max_prior_deviation,'.8g')
            state['goal_contract']['frozen_network_prior_radius']=max_prior_deviation
        else:state['goal_contract'].pop('frozen_network_prior_radius',None)
        state['goal_contract']['projection']=projection
        state['action_projection']=projection
    alignment=None
    if align_discount_with is not None:
        from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import PHYSICAL_KEYS
        from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import reward_discount
        physical=json.loads(Path(align_discount_with).read_text())
        if any(source_manifest.get(key)!=physical.get(key) for key in PHYSICAL_KEYS):
            raise ValueError('Discount alignment may not change the recorded physical/reward contract')
        discount=reward_discount(physical)
        if (not isinstance(critic_warmup_updates,int) or not 1<=critic_warmup_updates<=10000
                or not (checkpoint.parent/'pose_goal_experience.pt').is_file()):
            raise ValueError('Discount alignment needs actual replay and1..10000 critic warmup updates')
        if old_config['gamma']==discount:
            raise ValueError('Learner discount already matches the reward contract')
        alignment=dict(source_learner_discount=old_config['gamma'],learner_discount=discount,
            reward_discount=discount,critic_only_warmup_updates=critic_warmup_updates,
            recorded_rewards_unchanged=True,actor_unchanged_during_warmup=True)
        state['config']['gamma']=discount
        state['pending_discount_critic_warmup']=critic_warmup_updates
        state['discount_alignment']=alignment
        state['optimizers'][1]['state']={}
    if preserve_exploration:
        initial_std=old_config['initial_policy_std'];min_std=old_config['min_policy_std']
        max_std=old_config['max_policy_std']
        actor_lr=old_config['actor_lr'] if requested_actor_lr is None else requested_actor_lr
        demo_fade_updates=old_contract['demo_fade_updates']
    state['config'].update(min_policy_std=min_std, initial_policy_std=initial_std,
                           max_policy_std=max_std, actor_lr=actor_lr)
    state['goal_contract'].update(actor_lr=actor_lr, demo_fade_updates=demo_fade_updates,
                                 prior_fade_updates=demo_fade_updates)

    # Only Gaussian variance-head rows change. The deterministic mean function,
    # decoder, Q networks and normalization remain exactly identical.
    width = state['action_dim']
    if not preserve_exploration:
        state['model']['actor.network.4.weight'][width:].zero_()
        state['model']['actor.network.4.bias'][width:].fill_(math.log(initial_std))
        state['optimizers'][0]['state'] = {}
    for group in state['optimizers'][0]['param_groups']:
        group['lr'] = actor_lr
    if not preserve_exploration:
        state['entropy_contract']['target_per_dim'] = min(
            -1., math.log(max_std) + .5 * math.log(2 * math.pi * math.e) - .5)
    audit = dict(source_checkpoint=str(checkpoint.resolve()),
                 source_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                 deterministic_mean_unchanged=(state['goal_contract'].get('frozen_network_prior_radius',0.)==
                                               old_contract.get('frozen_network_prior_radius',0.)),
                 deterministic_network_mean_unchanged=True,
                 projected_mean_may_change=(state['goal_contract'].get('projection')!=old_contract.get('projection')),
                 goal_decoder_and_Q_coordinates_unchanged=True,
                 critic_and_normalizers_unchanged=True,
                 actor_optimizer_moments_reset=not preserve_exploration,
                 exploration_preserved=preserve_exploration,discount_alignment=alignment,
                 actor_learning_rate_changed=actor_lr!=old_config['actor_lr'],
                 actor_update_interval=state['goal_contract'].get('actor_update_interval',1),
                 demo_fade_uses_critic_progress='demo_fade_critic_offset' in state['goal_contract'],
                 prior_initial_weight=initial_weight,
                 prior_weight_floor=state['goal_contract'].get('frozen_network_prior_weight_floor',0.),
                 max_prior_deviation=state['goal_contract'].get('frozen_network_prior_radius',0.),
                 ongoing_imitation_disabled=initial_weight==0 and floor==0,
                 neural_prior_action_bound_disabled=not state['goal_contract'].get('frozen_network_prior_radius',0.),
                 BC_initialization_retained=True,
                 demo_replay_schedule_unchanged=state['goal_contract']['demo_fade_updates']==old_contract['demo_fade_updates'],
                 actual_replay_actions_rewards_unchanged=True,
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
        actual.setdefault('collection_goal_contract',old_contract)
        actual.setdefault('collection_policy_config',old_config)
        actual['goal_contract'] = state['goal_contract']
        actual['policy_only_migration'] = alignment is None
        actual['learning_horizon_migration'] = alignment is not None

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
    parser.add_argument('--actor-lr', type=float,
                        help='Explicit LR override; without it preserve the source LR with --preserve-exploration, otherwise use 1e-6.')
    parser.add_argument('--actor-update-interval',type=int,
                        help='Update the actor every N critic steps; keep demo replay fade tied to critic progress when delayed.')
    parser.add_argument('--demo-fade-updates', type=int, default=20000)
    parser.add_argument('--align-discount-with',type=Path,
                        help='Same physical manifest; explicitly fork legacy learner gamma and warm up only its critic.')
    parser.add_argument('--critic-warmup-updates',type=int,default=2000)
    parser.add_argument('--preserve-exploration',action='store_true')
    parser.add_argument('--prior-initial-weight',type=float,
                        help='Actor imitation weight before fade; set both this and --prior-weight-floor to0 to disable ongoing BC.')
    parser.add_argument('--prior-weight-floor',type=float,
                        help='Keep an actor-only frozen neural policy anchor after demo replay fades out.')
    parser.add_argument('--max-prior-deviation',type=float,
                        help='Bound generated normalized goals around that neural prior; historical actions stay measured.')
    args = parser.parse_args()
    try:
        _, audit = fork_checkpoint(args.checkpoint, args.output_dir,
                                  initial_std=args.initial_std, min_std=args.min_std,
                                  max_std=args.max_std, actor_lr=args.actor_lr,
                                  demo_fade_updates=args.demo_fade_updates,
                                  align_discount_with=args.align_discount_with,
                                  critic_warmup_updates=args.critic_warmup_updates,
                                  preserve_exploration=args.preserve_exploration,
                                  prior_initial_weight=args.prior_initial_weight,
                                  actor_update_interval=args.actor_update_interval,
                                  prior_weight_floor=args.prior_weight_floor,
                                  max_prior_deviation=args.max_prior_deviation)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(audit | {'source_policy_config': None, 'new_policy_config': None}))


if __name__ == '__main__':
    main()
