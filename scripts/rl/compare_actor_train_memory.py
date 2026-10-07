#!/usr/bin/env python3
"""Compare immutable actors on verified, past successful TRAIN observations.

This uses the production servo decoder, never Isaac, GPU, an optimizer or
evaluation transitions. Agreement on old observations is not physical success.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from export_eval_q_videos import restored_agent
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import (
    ActorTrainMemory, compatibility_contract, structure_sha256,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import servo_interval_loss
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS

GROUPS = dict(left_arm=list(range(1, 8)), right_arm=list(range(8, 15)),
              waist_head=[0, 15, 16], torso_fore_aft_height=[17, 18])


def owned_stable_bytes(path):
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise ValueError('Owned regular immutable input required')
    with path.open('rb') as f:
        before = os.fstat(f.fileno()); data = f.read(); after = os.fstat(f.fileno())
    identity = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns)
    if before.st_uid != os.getuid() or identity(before) != identity(after):
        raise ValueError('Input owner or state changed')
    return data


def statistics(raw, labels, greedy, logits, encoder, gate):
    recorded = encoder(raw, labels)
    predicted = encoder(raw, greedy)
    error = (predicted[:, :19] - recorded[:, :19]).abs()
    goal_error = (greedy[:, :19] - labels[:, :19]).abs()
    near = gate.entropy_mask(raw)[:, 19:].bool()
    closed = labels[:, 19:] > 0
    if bool((closed & ~near).any()):
        raise ValueError('Verified old closing labels must fit the current jaw gate')
    hands = []
    for i, name in enumerate(('left', 'right')):
        active = near[:, i]; closing = closed[:, i]
        hands.append(dict(hand=name, measured_gate_active_rows=int(active.sum()),
            recorded_closed_rows=int(closing.sum()),
            greedy_agreement_on_active_rows=float((greedy[active, 19+i] == labels[active, 19+i]).float().mean())
                if bool(active.any()) else None,
            greedy_closed_fraction_on_recorded_closed_rows=float((greedy[closing, 19+i] > 0).float().mean())
                if bool(closing.any()) else None,
            policy_close_probability_on_recorded_closed_rows=float(logits[closing, i].sigmoid().mean())
                if bool(closing.any()) else None))
    both = closed.all(-1)
    result = dict(rows=len(raw), normalized_absolute_goal_MAE=float(goal_error.mean()),
        actual_next_body_servo_command_MAE=float(error.mean()),
        servo_equivalent_interval_Huber=float(servo_interval_loss(encoder, raw, greedy, labels)),
        recorded_both_jaws_closed_rows=int(both.sum()),
        greedy_both_closed_fraction_on_recorded_both_closed_rows=float((greedy[both, 19:] > 0).all(-1).float().mean())
            if bool(both.any()) else None, hands=hands, body_groups={})
    for name, columns in GROUPS.items():
        difference = error[:, columns]; goals = goal_error[:, columns]
        result['body_groups'][name] = dict(next_servo_command_MAE=float(difference.mean()),
            normalized_goal_MAE=float(goals.mean()),
            identical_servo_command_different_goal_fraction=float(
                ((difference <= 1e-6) & (goals > 1e-4)).float().mean()),
            opposite_saturated_command_fraction=float(
                ((predicted[:, columns] * recorded[:, columns] < -0.99999)).float().mean()))
    return result


@torch.no_grad()
def compare(proof, memory):
    path = Path(proof['protected_checkpoint']); blob = owned_stable_bytes(path)
    checkpoint_hash = hashlib.sha256(blob).hexdigest()
    if checkpoint_hash != proof['checkpoint_SHA256']:
        raise ValueError('Checkpoint does not match its immutable identity proof')
    state = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    if (state['actor_updates'], state['critic_updates']) != (proof['actor_updates'], proof['critic_updates']):
        raise ValueError('Checkpoint update counters differ from its proof')
    bank = ActorTrainMemory(memory, compatibility=compatibility_contract(state['goal_contract']),
        frozen_anchor_SHA256=structure_sha256(state['body_anchor_state']))
    agent, _ = restored_agent(state)
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    reports = []
    for region in REGIONS:
        for episode in bank.episodes[region]:
            raw, labels = episode['actor_obs'], episode['action']
            greedies, jaws = [], []
            for chunk in raw.split(512):
                normal = agent.actor_normalizer(agent.actor_features(chunk))
                greedies.append(agent.act(chunk, True))
                jaws.append(agent.parameters_at(normal)[2])
            greedy, logits = torch.cat(greedies), torch.cat(jaws)
            if not torch.isfinite(greedy).all() or not torch.isfinite(logits).all():
                raise ValueError('Restored actor output is nonfinite')
            reports.append(dict(region=region, identity=episode['identity'],
                source_TRAIN_seed=episode['outcome']['layout']['seed'],
                whole_path=statistics(raw, labels, greedy, logits,
                    agent.goal_servo_critic_encoder, agent.action_projector),
                final64=statistics(raw[-64:], labels[-64:], greedy[-64:], logits[-64:],
                    agent.goal_servo_critic_encoder, agent.action_projector)))
    by_region = {}
    for region in REGIONS:
        values = [r for r in reports if r['region'] == region]
        by_region[region] = dict(episodes=len(values), rows=sum(r['whole_path']['rows'] for r in values))
        for phase in ('whole_path', 'final64'):
            fields = ('normalized_absolute_goal_MAE', 'actual_next_body_servo_command_MAE',
                      'servo_equivalent_interval_Huber',
                      'greedy_both_closed_fraction_on_recorded_both_closed_rows')
            by_region[region][phase] = {}
            for field in fields:
                measured = [r[phase][field] for r in values if r[phase][field] is not None]
                by_region[region][phase][field+'_episode_mean'] = sum(measured)/len(measured) if measured else None
    if not all(torch.equal(v, agent.state_dict()[k]) for k, v in before.items()):
        raise ValueError('Offline comparison altered model or normalizer state')
    if hashlib.sha256(owned_stable_bytes(path)).hexdigest() != checkpoint_hash:
        raise ValueError('Immutable source checkpoint changed during comparison')
    role = proof.get('role') or Path(proof.get('source_run', path.parent)).name
    return dict(role=role, source_commit=proof.get('source_commit'),
        checkpoint_SHA256=checkpoint_hash, actor_updates=state['actor_updates'],
        critic_updates=state['critic_updates'], by_region=by_region, episodes=reports,
        aggregation='uniform mean per successful path within region; not row-weighted',
        source_models_normalizers_and_files_unchanged=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--actor-memory', type=Path, required=True)
    p.add_argument('--matching-memory-proof', type=Path, required=True)
    p.add_argument('--checkpoint-proof', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); torch.set_num_threads(1)
    blob = owned_stable_bytes(args.actor_memory)
    memory_hash = hashlib.sha256(blob).hexdigest()
    audit = json.loads(owned_stable_bytes(args.matching_memory_proof))
    if memory_hash != audit['actor_only_dataset_SHA256'] or not all(audit.get(k) is True for k in (
            'all27_successful_TRAIN_paths_and12476_rows_own_HDF_and_manifest_exact',
            'current_stricter_reset_relative_drop10cm_checked_entire_attempt',
            'source_TRAIN_and_new_TRAIN_and_all_DEV_FINAL_seeds_disjoint',
            'no_active_HDF_or_replay_read')):
        raise ValueError('Verified closed physical TRAIN memory audit required')
    memory = torch.load(io.BytesIO(blob), map_location='cpu', weights_only=True)
    reports = [compare(json.loads(owned_stable_bytes(proof)), memory) for proof in args.checkpoint_proof]
    if hashlib.sha256(owned_stable_bytes(args.actor_memory)).hexdigest() != memory_hash:
        raise ValueError('Source actor-only memory changed')
    result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(), actor_memory_SHA256=memory_hash,
        source_checkpoint_SHA256=memory['source_checkpoint_SHA256'], models=reports,
        all27_verified_past_successful_TRAIN_paths12476_rows=True,
        uses_production_servo_decoder_and_jaw_gate=True,
        old_observation_agreement_NOT_new_physical_success=True,
        different_goals_same_one_step_servo_NOT_same_future_trajectory=True,
        actual_lift_contact_collision_and_generalization_require_new_full_physical_evaluation=True,
        no_optimizer_update_or_GPU_or_Isaac_or_active_HDF_replay_read=True,
        no_Q_reward_or_evaluation_transition_import=True, independent_FINAL_unused=True,
        goal_not_complete=True)
    with args.output.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps(dict(models=[dict(role=r['role'], actor_updates=r['actor_updates'],
        final64={k:v['final64'] for k,v in r['by_region'].items()}) for r in reports],
        offline_TRAIN_only=True, new_physical_success_NOT_claimed=True)))


if __name__ == '__main__':
    main()
