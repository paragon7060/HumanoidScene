#!/usr/bin/env python3
"""Add measured critic deadline coordinates only to pristine SAC inputs."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib, json, os
from pathlib import Path
import torch

from kuavo_isaaclab_scene.rl.multi_box.observations.task_timing import critic_episode_clock_config, VARIANT, CRITIC_REMAINING_INDEX
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from prepare_actual_success_actor_tail import identical


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('initial-checkpoint', 'training-manifest', 'waypoints', 'output-dir'):
        p.add_argument('--'+name, type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    experience = args.initial_checkpoint.parent/'staged_goal_experience.pt'
    sources = (args.initial_checkpoint, experience, args.training_manifest, args.waypoints)
    if any(x.is_symlink() or not x.is_file() or x.stat().st_uid != os.getuid() for x in sources):
        raise ValueError('Owned regular pristine input files are required')
    hashes = {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    state = torch.load(args.initial_checkpoint, map_location='cpu', weights_only=True)
    replay = torch.load(experience, map_location='cpu', weights_only=True)
    pilot_class = staged_policy_class(state.get('artifact_type'))
    if pilot_class is None or not issubclass(pilot_class, ActualFlapResidualSACPilot):
        raise ValueError('Only matching actual-flap SAC inputs support this critic clock')
    if (state['goal_contract']['actor_dim'], state['goal_contract']['critic_dim']) != (518, 577) \
            or state['actor_updates'] or state['critic_updates'] or state['model']['critic_normalizer.count'] \
            or len(state['optimizers']) != 4 or any(o['state'] for o in state['optimizers']) \
            or any(len(v) for v in replay['executed_goal_transitions'].values()) \
            or any(state['successful_train_transitions']['episodes'].values()) \
            or any(replay['measured_train_credit_bank']['episodes'].values()):
        raise ValueError('Fresh Q, four empty optimizers and all empty reward-bearing banks are required')
    if state['goal_contract'] != replay['goal_contract'] or any(
            x.get('critic_episode_clock') is not None or 'critic_episode_clock' in x['goal_contract']
            for x in (state, replay)):
        raise ValueError('Matching pristine original critic-clock contracts are required')
    physical = json.loads(args.training_manifest.read_text())
    contract = state['goal_contract']['physical_contract']
    if {k: physical.get(k) for k in contract} != contract:
        raise ValueError('Source physical manifest differs')
    original = deepcopy(state); clock = critic_episode_clock_config(VARIANT)
    widened = []
    for key, value in list(state['model'].items()):
        if key in ('critic_normalizer.mean', 'critic_normalizer.var'):
            fill = 1 if key.endswith('.var') else 0
            state['model'][key] = torch.cat((value[:CRITIC_REMAINING_INDEX],
                value.new_full((1,), fill), value[CRITIC_REMAINING_INDEX:]))
        elif key.startswith(('q1.', 'q2.', 'target1.', 'target2.')) and value.ndim == 2 \
                and value.shape[1] == 577+21:
            state['model'][key] = torch.cat((value[:, :CRITIC_REMAINING_INDEX],
                value.new_zeros(len(value), 1), value[:, CRITIC_REMAINING_INDEX:]), -1)
            widened.append(key)
    assert len(widened) == 4
    state['critic_obs_dim'] = 578
    for key in ('critic_obs', 'next_critic_obs'):
        assert replay['executed_goal_transitions'][key].shape == (0, 577)
        replay['executed_goal_transitions'][key] = replay['executed_goal_transitions'][key].new_empty(0, 578)
    for value in (state, replay):
        value['critic_episode_clock'] = deepcopy(clock)
        value['goal_contract']['critic_episode_clock'] = deepcopy(clock)
        value['goal_contract']['critic_dim'] = 578
    changed = set(widened) | {'critic_normalizer.mean', 'critic_normalizer.var'}
    assert all(torch.equal(value, original['model'][key]) for key, value in state['model'].items() if key not in changed)
    assert identical(state['optimizers'], original['optimizers'])
    assert state['config'] == original['config']
    assert all(torch.isfinite(v).all() for v in state['model'].values())
    args.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    torch.save(state, args.output_dir/'checkpoint_00000000.pt')
    torch.save(replay, args.output_dir/'staged_goal_experience.pt')
    assert identical(state, torch.load(args.output_dir/'checkpoint_00000000.pt', map_location='cpu', weights_only=True))
    assert identical(replay, torch.load(args.output_dir/'staged_goal_experience.pt', map_location='cpu', weights_only=True))
    assert hashes == {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in sources}
    proof = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), source_SHA256=hashes,
        actual_initial_actor_jaws_normalizer_four_optimizers_and_learning_config_unchanged=True,
        only_pristine_Q_input_weights_and_critic_normalizer_widened=True,
        new_Q_feature_initial_weight0_mean0_var1=True,
        original_held_phase_clock_and_last_six_context_preserved=True,
        all_actual_Q_updates_optimizers_replay_success_and_nstep_reward_banks0=True,
        only_future_critic_clock_coordinates_changed=clock,
        actor_features_controller_rewards_success_safety_randomization_unchanged=True,
        no_old_nonempty_Q_or_reward_rows_relabelled=True, source_files_preserved=True,
        new_physical_training_NOT_started=True, goal_not_complete=True)
    for name, value in [('training_manifest.json', physical), ('waypoints.json', json.loads(args.waypoints.read_text())),
            ('initialization_verification.json', proof), ('manifest.json', dict(
                artifact_type=state['artifact_type'], goal_contract=state['goal_contract'], initialization_verification=proof))]:
        (args.output_dir/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output_dir=str(args.output_dir),critic_episode_clock=clock,
        actor_policy_unchanged=True,initial_Q_functions_preserved_by_zero_new_column=True,
        all_reward_banks0=True,new_training_NOT_started=True)), flush=True)


if __name__ == '__main__':
    main()
