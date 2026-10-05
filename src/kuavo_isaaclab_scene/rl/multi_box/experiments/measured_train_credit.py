"""Opt-in behavior n-step critic credit from matching real held TRAIN paths.

Successful and failed episodes share this bank. Uncorrected intermediate
behavior actions make this an experimental auxiliary objective, not an unbiased
current-policy return target. Online one-step rewards and replay stay unchanged.
"""
from copy import deepcopy
import math

import torch

from .physical_train_credit import discounted_episode_windows
from .staged_train_success import KEYS, REGIONS

VARIANT = 'measured-nstep16'


def measured_credit_config(variant):
    if variant in (None, 'one-step'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown measured TRAIN credit variant')
    return dict(name='matching_actual_held_TRAIN_nstep16_v1', horizon=16,
        capacity_per_region=8192, batch_size=64, critic_weight=.1,
        retained_successful_episodes_per_region=1,
        source='completed_matching_successful_and_failed_TRAIN_paths',
        sampling='uniform_rows_within_each_available_region',
        online_target='unchanged_one_step_SAC', off_policy_correction=False,
        intermediate_entropy_included=False, evaluation_import_allowed=False,
        episode_crossing_allowed=False, reward_relabeling=False)


def validate_measured_credit_batch(agent, batch, weight):
    if not math.isfinite(weight) or weight < 0 or (weight and batch is None):
        raise ValueError('Measured multi-step critic requires finite nonnegative weight and real rows')
    if batch is None:
        return
    n = len(batch['reward'])
    shapes = dict(actor_obs=(n, agent.actor_obs_dim), next_actor_obs=(n, agent.actor_obs_dim),
        critic_obs=(n, agent.critic_obs_dim), next_critic_obs=(n, agent.critic_obs_dim),
        action=(n, 21), reward=(n,), terminated=(n,), bootstrap_discount=(n,), n_steps=(n,))
    if not n or set(batch) != set(shapes) or any(
            batch[k].shape != shape or not torch.isfinite(batch[k]).all() for k, shape in shapes.items()) \
            or batch['terminated'].dtype != torch.bool or batch['n_steps'].dtype != torch.int64 \
            or (batch['n_steps'] < 1).any() or (batch['n_steps'] > 16).any() \
            or (batch['action'].abs() > 1.00001).any() or not (batch['action'][:, 19:].abs() == 1).all():
        raise ValueError('Malformed actual-goal multi-step critic rows')
    expected = (agent.config.gamma ** batch['n_steps'].to(batch['reward'])).masked_fill(batch['terminated'], 0.)
    if not torch.allclose(batch['bootstrap_discount'], expected, atol=1e-7, rtol=1e-6):
        raise ValueError('Measured multi-step discount must match its actual horizon and terminal')


class MeasuredTrainCreditBank:
    """Whole physical paths, with no DEV, altered controller, or broken chains."""
    def __init__(self, actor_dim, critic_dim, gamma, config):
        if config != measured_credit_config(VARIANT) or not math.isfinite(gamma) or not 0 < gamma <= 1:
            raise ValueError('Unknown measured TRAIN credit configuration')
        self.actor_dim, self.critic_dim, self.gamma = actor_dim, critic_dim, gamma
        self.config = deepcopy(config)
        self.episodes = {region: [] for region in REGIONS}
        self._windows = {}

    @property
    def size(self):
        return sum(len(e['rows']['reward']) for es in self.episodes.values() for e in es)

    def add_episode(self, rows, outcome, *, source_run):
        result = outcome.get('result') or {}
        stage = result.get('staged_base') or {}
        if outcome.get('split') != 'train' or not outcome.get('initial_layout_valid') \
                or not outcome.get('complete') or result.get('numerical_failure') or result.get('invalid_reset') \
                or stage.get('phase') != 'held_grasp' or stage.get('manipulation_start') is None \
                or 'waypoint_probe' in stage:
            raise ValueError('Measured credit requires completed matching TRAIN; DEV/FINAL/probes are excluded')
        n = len(rows['reward'])
        shapes = dict(actor_obs=(n, self.actor_dim), next_actor_obs=(n, self.actor_dim),
            critic_obs=(n, self.critic_dim), next_critic_obs=(n, self.critic_dim),
            action=(n, 21), reward=(n,), terminated=(n,))
        if not 1 <= n <= 900 or set(rows) != set(KEYS) or any(
                rows[k].shape != shape or not torch.isfinite(rows[k]).all() for k, shape in shapes.items()) \
                or (rows['action'].abs() > 1.00001).any() or not (rows['action'][:, 19:].abs() == 1).all():
            raise ValueError('Malformed measured TRAIN path')
        if not all(bool((rows[k][:, -6] == 1).all()) for k in
                ('actor_obs', 'critic_obs', 'next_actor_obs', 'next_critic_obs')):
            raise ValueError('Measured credit excludes unconfirmed base approach')
        region = outcome['layout']['target_region']
        if region not in REGIONS or not bool((rows['actor_obs'][:, 94:98].argmax(-1) == REGIONS.index(region)).all()):
            raise ValueError('Measured credit region differs from perceived state')
        # This also rejects intermediate terminals and any missing/reset row.
        discounted_episode_windows(rows, horizon=self.config['horizon'], gamma=self.gamma)
        identity = f"{source_run}/wave{outcome['wave']}/env{outcome['environment']}/seed{outcome['layout']['seed']}"
        if any(e['identity'] == identity for es in self.episodes.values() for e in es):
            raise ValueError('Duplicate measured TRAIN path')
        episodes = self.episodes[region]
        episodes.append(dict(identity=identity, outcome=deepcopy(outcome),
            rows={k: v.detach().cpu().contiguous().clone() for k, v in rows.items()}))
        while sum(len(e['rows']['reward']) for e in episodes) > self.config['capacity_per_region']:
            successes = sum(bool(e['outcome']['result']['success']) for e in episodes)
            removable = next(i for i, e in enumerate(episodes)
                if not e['outcome']['result']['success'] or successes > 1)
            self._windows.pop(episodes.pop(removable)['identity'], None)

    def sample(self, count, device):
        regions = [r for r in REGIONS if self.episodes[r]]
        if count < 1 or not regions:
            raise ValueError('Cannot sample empty measured TRAIN credit')
        groups = {r: [] for r in regions}
        for i in range(count):
            groups[regions[i % len(regions)]].append(i)
        selected = []
        for region, positions in groups.items():
            episodes = self.episodes[region]
            lengths = torch.tensor([len(e['rows']['reward']) for e in episodes])
            indices = torch.randint(int(lengths.sum()), (len(positions),))
            ends = lengths.cumsum(0)
            for index in indices:
                ep = int(torch.searchsorted(ends, index, right=True))
                row = int(index - (ends[ep - 1] if ep else 0))
                episode = episodes[ep]
                identity = episode['identity']
                if identity not in self._windows:
                    self._windows[identity] = discounted_episode_windows(episode['rows'],
                        horizon=self.config['horizon'], gamma=self.gamma)
                selected.append({k: v[row] for k, v in self._windows[identity].items()})
        order = torch.randperm(count)
        return {k: torch.stack([s[k] for s in selected])[order].to(device) for k in selected[0]}

    def state(self):
        return dict(config=self.config, gamma=self.gamma, episodes=self.episodes)

    def restore(self, state):
        if state.get('config') != self.config or state.get('gamma') != self.gamma \
                or set(state.get('episodes', {})) != set(REGIONS):
            raise ValueError('Saved measured TRAIN credit differs')
        for region, episodes in state['episodes'].items():
            for episode in episodes:
                if episode['outcome']['layout']['target_region'] != region:
                    raise ValueError('Saved measured path region differs')
                self.add_episode(episode['rows'], episode['outcome'],
                    source_run=episode['identity'].split('/wave')[0])

    def report(self):
        return dict(rows=self.size, by_region={r: dict(episodes=len(es),
            successes=sum(bool(e['outcome']['result']['success']) for e in es),
            failures=sum(not bool(e['outcome']['result']['success']) for e in es),
            rows=sum(len(e['rows']['reward']) for e in es)) for r, es in self.episodes.items()},
            TRAIN_only=True, evaluation_rows=0)


def add_measured_training_wave(bank, wave, outcomes, batches, *, source_run):
    if wave['split'] != 'train':
        if batches:
            raise ValueError('Evaluation cannot contribute measured credit')
        return dict(added=0, skipped={})
    parts = {}
    for ids, batch in batches:
        if len(ids) != len(batch['reward']) or len(ids.unique()) != len(ids):
            raise ValueError('Measured environment identities differ')
        for j, env in enumerate(ids.tolist()):
            parts.setdefault(env, []).append({k: v[j:j + 1] for k, v in batch.items()})
    added, skipped = 0, {}
    for outcome in outcomes:
        path = parts.get(outcome['environment'])
        if path is None:
            skipped['no_held_rows'] = skipped.get('no_held_rows', 0) + 1
            continue
        try:
            bank.add_episode({k: torch.cat([p[k] for p in path]) for k in KEYS},
                outcome, source_run=source_run)
        except ValueError as error:
            reason = str(error)
            skipped[reason] = skipped.get(reason, 0) + 1
        else:
            added += 1
    return dict(added=added, skipped=skipped)
