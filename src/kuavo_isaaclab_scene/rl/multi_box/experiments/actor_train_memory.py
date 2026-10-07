"""Actor-only memory of verified successful SAC TRAIN paths.

This bank cannot supply critic observations, rewards, targets or replay rows.
Its old paths share the same actor/controller contract and were checked against
the current safety guard before export. New successes still enter ordinary Q
replay through the existing, separate TrainSuccessBank.
"""
from copy import deepcopy
import hashlib
import json

import torch

from .staged_train_success import REGIONS, validate_success_outcome

FORMAT = 'actual_safe_TRAIN_actor_memory_v1'


def actor_memory_contract():
    return dict(name='verified_successful_TRAIN_actor_only_memory_v1',
        format=FORMAT, source='completed_safe_SAC_TRAIN_only',
        fields=['actor_obs', 'action'], critic_or_reward_import_allowed=False,
        actor_batch=64, region_balance=True, episode_uniform_within_region=True,
        tail_fraction=.5, tail_steps=64, current_successes_in_same_actor_sampler=True,
        existing_body_goal_servo_and_jaw_loss_weights_unchanged=True,
        online_Q_replay_and_measured_return_bank_unchanged=True,
        evaluation_or_VR_teacher_import_allowed=False)


def compatibility_contract(goal):
    keys = ('actor_dim', 'goal_center', 'goal_scale', 'action_columns',
        'action_coordinates', 'shelf_templates', 'supplemental_perception',
        'phase_context', 'context_order', 'clock_horizon', 'time_harmonics',
        'condition_on_shelf', 'actor_clock_limit', 'body_controller',
        'body_correction_radius', 'jaw_proximity_gate')
    physical = {k: deepcopy(v) for k, v in goal['physical_contract'].items()
                if k not in ('reward_profile', 'discount')}
    actor={k:deepcopy(goal.get(k)) for k in keys}
    from .size_workplaces import FORMAT as SIZE_FORMAT,validate_size_workplaces
    if isinstance(actor['shelf_templates'],dict) and actor['shelf_templates'].get('name')==SIZE_FORMAT:
        typed=validate_size_workplaces(actor['shelf_templates'])
        # Old commands are actor-only references at their own original held
        # x/y/yaw observations. Keep their stored contract/outcomes untouched.
        # Neither rewards nor old Q/replay are compatible with new targets.
        if actor['context_order']!=['held_phase','held_x_rack_m','held_y_rack_m',
                'sin_held_yaw','cos_held_yaw','policy_radius']:
            raise ValueError('Old actor references require the explicit measured held-pose context')
        actor['shelf_templates']=deepcopy(typed['source_region_workplaces'])
    return dict(actor=actor,
                validated_current_physics_control_success_and_safety=physical)


def structure_sha256(value):
    """Hash frozen coordinates/weights without torch archive container details."""
    digest = hashlib.sha256()
    def add(x):
        if isinstance(x, torch.Tensor):
            v = x.detach().cpu().contiguous()
            digest.update(json.dumps(['tensor', str(v.dtype), list(v.shape)]).encode())
            digest.update(v.numpy().tobytes())
        elif isinstance(x, dict):
            digest.update(b'{')
            for k in sorted(x):
                digest.update(json.dumps(k).encode()); add(x[k])
            digest.update(b'}')
        elif isinstance(x, (tuple, list)):
            digest.update(b'[')
            for v in x: add(v)
            digest.update(b']')
        else:
            digest.update(json.dumps(x, allow_nan=False).encode())
    add(value)
    return digest.hexdigest()


class ActorTrainMemory:
    def __init__(self, state, *, compatibility, frozen_anchor_SHA256):
        expected = {'format', 'actor_dim', 'episodes', 'source_checkpoint_SHA256',
                    'compatibility', 'frozen_body_anchor_SHA256'}
        if set(state) != expected or state['format'] != FORMAT \
                or state['actor_dim'] != 518 or state['compatibility'] != compatibility \
                or state['frozen_body_anchor_SHA256'] != frozen_anchor_SHA256:
            raise ValueError('Actor-only TRAIN memory coordinates or fields differ')
        if len(state['source_checkpoint_SHA256']) != 64:
            raise ValueError('Verified source checkpoint identity is missing')
        self.episodes = {r: [] for r in REGIONS}
        identities = set()
        for e in state['episodes']:
            if set(e) != {'identity', 'outcome', 'actor_obs', 'action'}:
                raise ValueError('Actor memory must not contain critic/reward/replay fields')
            o = e['outcome']; validate_success_outcome('train', o)
            suffix = f"/wave{o['wave']}/env{o['environment']}/seed{o['layout']['seed']}"
            if not e['identity'].endswith(suffix) or e['identity'] in identities:
                raise ValueError('Actor memory TRAIN identity differs or is duplicated')
            identities.add(e['identity'])
            raw, action = e['actor_obs'], e['action']; n = len(action)
            region = o['layout']['target_region']; index = REGIONS.index(region)
            if not 1 <= n <= 900 or raw.shape != (n, 518) or action.shape != (n, 21) \
                    or not torch.isfinite(raw).all() or not torch.isfinite(action).all() \
                    or not raw.is_floating_point() or not action.is_floating_point() \
                    or (action.abs() > 1.00001).any() or not (action[:, 19:].abs() == 1).all() \
                    or not (raw[:, -6] == 1).all() \
                    or not (raw[:, 94:98].argmax(-1) == index).all() \
                    or not (raw[:, 94 + index] > .5).all():
                raise ValueError('Malformed measured actor memory path')
            self.episodes[region].append(e)
        if not all(self.episodes.values()):
            raise ValueError('Verified actor memory needs successful TRAIN from all four regions')
        self._state = deepcopy(state)
        # Checkpoint map_location may put historical paths on CUDA, while the
        # online success bank deliberately stores new paths on CPU. Keep both
        # banks on CPU and move only the sampled batch to the learner device.
        for e in self._state['episodes']:
            for key in ('actor_obs', 'action'):
                e[key] = e[key].detach().cpu().contiguous()
        self.episodes = {r: [e for e in self._state['episodes']
                            if e['outcome']['layout']['target_region'] == r] for r in REGIONS}

    @property
    def size(self):
        return sum(len(e['action']) for e in self._state['episodes'])

    def state(self):
        return self._state

    def report(self):
        return dict(rows=self.size, by_region={r: dict(episodes=len(es),
            rows=sum(len(e['action']) for e in es)) for r, es in self.episodes.items()},
            actor_only=True, critic_rows=0, reward_rows=0, evaluation_rows=0)

    def sample_actor(self, count, device, current_success_bank):
        if type(count) is not int or count < 1:
            raise ValueError('Positive actor batch size required')
        selected = []; old_rows = tail_rows = 0
        for i in range(count):
            region = REGIONS[i % len(REGIONS)]
            old = self.episodes[region]
            current = current_success_bank.episodes[region] if current_success_bank else []
            j = int(torch.randint(len(old) + len(current), ()))
            previous = j < len(old)
            e = old[j] if previous else current[j - len(old)]['rows']
            n = len(e['action']); tail = i < round(count * .5)
            row = int(torch.randint(max(0, n - 64) if tail else 0, n, ()))
            selected.append({k: e[k][row] for k in ('actor_obs', 'action')})
            old_rows += int(previous); tail_rows += int(tail)
        order = torch.randperm(count)
        batch = {k: torch.stack([e[k] for e in selected])[order].to(device)
                 for k in ('actor_obs', 'action')}
        return batch, dict(successful_train_actor_rows=count,
            successful_train_actor_designated_tail_rows=tail_rows,
            previous_success_actor_memory_rows=old_rows,
            new_success_actor_memory_rows=count-old_rows,
            previous_success_memory_rows_in_Q_batch=0)
