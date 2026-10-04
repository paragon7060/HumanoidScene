"""Record exact goals chosen by a frozen policy on explicitly declared TRAIN cases.

This is training-data collection without optimization. A past evaluation cannot
be converted into this archive, and physical delta commands are never inverted.
"""
from copy import deepcopy
from pathlib import Path

import torch

from .staged_train_success import KEYS, TrainSuccessBank
from .staged_physics import require_current_lift_contract

FORMAT = 'actual_frozen_policy_TRAIN_goal_collection_v1'
FILENAME = 'actual_train_goal_collection.pt'


def validate_training_collection(layout, *, staged_policy, optimization, live_teacher):
    if not staged_policy or optimization or live_teacher or not isinstance(layout, dict) \
            or layout.get('split') != 'train':
        raise ValueError('Goal collection requires an explicit TRAIN layout, frozen staged SAC and no teacher')


class TrainingGoalCollector:
    def __init__(self, layout, goal_contract):
        validate_training_collection(layout, staged_policy=True, optimization=False, live_teacher=False)
        if goal_contract.get('name') != 'staged_base_hold_remaining_hybrid_sac_v1' \
                or goal_contract.get('actor_dim') != 480 or goal_contract.get('critic_dim') != 539:
            raise ValueError('Training goal collection requires the current hybrid21 context')
        require_current_lift_contract(goal_contract.get('physical_contract', {}))
        self.layout = deepcopy(layout)
        self.contract = deepcopy(goal_contract)
        self.rows = []

    def append(self, previous, following, reward, terminated):
        ao, co, action = previous
        na, nc = following
        row = dict(actor_obs=ao, critic_obs=co, action=action,
                   next_actor_obs=na, next_critic_obs=nc, reward=reward, terminated=terminated)
        widths = dict(actor_obs=480, critic_obs=539, action=21,
                      next_actor_obs=480, next_critic_obs=539)
        if any(row[k].shape != (1, width) for k, width in widths.items()) \
                or reward.shape != (1,) or terminated.shape != (1,) \
                or terminated.dtype != torch.bool or not all(torch.isfinite(v).all() for v in row.values()):
            raise ValueError('Malformed measured single-environment goal transition')
        if action.abs().max() > 1.00001 or not (action[:, 19:21].abs() == 1).all() \
                or not all((row[k][:, -6] == 1).all() for k in ('actor_obs', 'critic_obs', 'next_actor_obs', 'next_critic_obs')):
            raise ValueError('Goals must retain actual binary jaws and the confirmed held phase')
        if self.rows and bool(self.rows[-1]['terminated'][0]):
            raise ValueError('A training goal path cannot cross an episode reset')
        if self.rows and not torch.equal(self.rows[-1]['next_critic_obs'], co.detach().cpu()):
            raise ValueError('Recorded training goals must form one continuous measured path')
        self.rows.append({k: v.detach().cpu().contiguous().clone() for k, v in row.items()})

    def save(self, directory, outcome, *, actor_updates, critic_updates, completed, interrupted):
        if outcome.get('layout') != self.layout or outcome.get('split') != 'train':
            raise ValueError('Goal archive must retain its originally declared TRAIN identity')
        rows = {k: torch.cat([r[k] for r in self.rows]) for k in KEYS} if self.rows else None
        bank = TrainSuccessBank(480, 539)
        if (outcome.get('result') or {}).get('success'):
            if not completed or interrupted or rows is None:
                raise ValueError('A successful goal archive must be a complete physical attempt')
            bank.add_episode(rows, outcome, source_run=Path(directory).name, split='train')
        artifact = dict(artifact_type=FORMAT, collection_phase='train',
            collection_declared_before_rollout=True, frozen_policy=True, optimizer_updates=0,
            source_actor_updates=actor_updates, source_critic_updates=critic_updates,
            source_goal_contract=self.contract, outcome=deepcopy(outcome),
            completed=bool(completed), interrupted=bool(interrupted),
            evaluation_or_probe_imported=False, physical_delta_actions_inverted=False,
            goal_transitions=rows, successful_train_transitions=bank.state())
        path = Path(directory) / FILENAME
        if path.exists():
            raise ValueError('Refusing to replace an existing TRAIN goal archive')
        pending = path.with_suffix('.pending')
        torch.save(artifact, pending)
        pending.replace(path)
        return dict(filename=FILENAME, collection_phase='train', frozen_policy=True,
                    optimizer_updates=0, held_goal_rows=len(self.rows), success_bank=bank.report(),
                    evaluation_or_probe_imported=False, physical_delta_actions_inverted=False)
