"""Opt-in behavior n-step critic credit from matching real held TRAIN paths.

Successful and failed episodes share this bank. Uncorrected intermediate
behavior actions make this an experimental auxiliary objective, not an unbiased
current-policy return target. Online one-step rewards and replay stay unchanged.
"""
from copy import deepcopy
import math

import torch

from .physical_train_credit import discounted_episode_windows, completed_episode_returns
from .staged_train_success import KEYS, REGIONS, validate_success_outcome

VARIANT = 'measured-nstep16'
TERMINAL_VARIANT = 'measured-nstep16-terminal25'
EPISODE_RETURN_VARIANT = 'measured-episode-return'
BALANCED_RETURN_VARIANT = 'measured-episode-return-balanced50'
VARIANTS = (VARIANT, TERMINAL_VARIANT, EPISODE_RETURN_VARIANT, BALANCED_RETURN_VARIANT)


def measured_credit_config(variant):
    if variant in (None, 'one-step'):
        return None
    if variant not in VARIANTS:
        raise ValueError('Unknown measured TRAIN credit variant')
    config = dict(name='matching_actual_held_TRAIN_nstep16_v1', horizon=16,
        capacity_per_region=8192, batch_size=64, critic_weight=.1,
        retained_successful_episodes_per_region=1,
        source='completed_matching_successful_and_failed_TRAIN_paths',
        sampling='uniform_rows_within_each_available_region',
        online_target='unchanged_one_step_SAC', off_policy_correction=False,
        intermediate_entropy_included=False, evaluation_import_allowed=False,
        episode_crossing_allowed=False, reward_relabeling=False)
    if variant == TERMINAL_VARIANT:
        config.update(name='matching_actual_held_TRAIN_nstep16_terminal25_v1',
            sampling='75percent_uniform_rows25percent_final_terminal_per_available_region',
            terminal_batch_fraction=.25, terminal_episode_sampling='uniform_episodes_within_region',
            terminal_target='actual_last_reward_no_bootstrap_no_entropy',
            critic_weight_unchanged=True)
    if variant in (EPISODE_RETURN_VARIANT, BALANCED_RETURN_VARIANT):
        config.update(name='matching_actual_held_TRAIN_episode_return_v1', horizon=900,
            sampling='uniform_episodes_then_uniform_rows_within_available_region',
            auxiliary_target='actual_discounted_rewards_through_real_terminal_no_bootstrap',
            episode_length_bias_removed=True, bootstrap_allowed=False,
            terminal_and_timeout_rewards_unchanged=True, critic_weight_unchanged=True)
    if variant == BALANCED_RETURN_VARIANT:
        config.update(name='matching_actual_held_TRAIN_episode_return_balanced50_v1',
            sampling='uniform_regions_then_half_safe_success_half_failure_then_uniform_episode_and_row',
            success_fraction_per_region_when_both_classes=.5,
            missing_class_fallback='available_actual_TRAIN_class_only',
            success_evidence='safe_opposing_bilateral_pinch_hold_and_proof_lift')
    return config


def validate_measured_credit_batch(agent, batch, weight):
    if not math.isfinite(weight) or weight < 0 or (weight and batch is None):
        raise ValueError('Measured multi-step critic requires finite nonnegative weight and real rows')
    if batch is None:
        return
    n = len(batch['reward'])
    config = getattr(agent, 'measured_train_credit_config', None)
    if config is not None and config not in tuple(measured_credit_config(v) for v in VARIANTS):
        raise ValueError('Measured critic sampling configuration differs')
    max_steps = 900 if config is not None and config.get('bootstrap_allowed') is False else 16
    shapes = dict(actor_obs=(n, agent.actor_obs_dim), next_actor_obs=(n, agent.actor_obs_dim),
        critic_obs=(n, agent.critic_obs_dim), next_critic_obs=(n, agent.critic_obs_dim),
        action=(n, 21), reward=(n,), terminated=(n,), bootstrap_discount=(n,), n_steps=(n,))
    if not n or set(batch) != set(shapes) or any(
            batch[k].shape != shape or not torch.isfinite(batch[k]).all() for k, shape in shapes.items()) \
            or batch['terminated'].dtype != torch.bool or batch['n_steps'].dtype != torch.int64 \
            or (batch['n_steps'] < 1).any() or (batch['n_steps'] > max_steps).any() \
            or (batch['action'].abs() > 1.00001).any() or not (batch['action'][:, 19:].abs() == 1).all():
        raise ValueError('Malformed actual-goal multi-step critic rows')
    if max_steps == 900 and not bool(batch['terminated'].all()):
        raise ValueError('Completed episode returns must reach the real terminal without bootstrap')
    expected = (agent.config.gamma ** batch['n_steps'].to(batch['reward'])).masked_fill(batch['terminated'], 0.)
    if not torch.allclose(batch['bootstrap_discount'], expected, atol=1e-7, rtol=1e-6):
        raise ValueError('Measured multi-step discount must match its actual horizon and terminal')


class MeasuredTrainCreditBank:
    """Whole physical paths, with no DEV, altered controller, or broken chains."""
    def __init__(self, actor_dim, critic_dim, gamma, config):
        if config not in tuple(measured_credit_config(v) for v in VARIANTS) \
                or not math.isfinite(gamma) or not 0 < gamma <= 1:
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
        if self.config.get('success_fraction_per_region_when_both_classes') is not None and result.get('success'):
            validate_success_outcome('train', outcome)
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
        discounted_episode_windows(rows, horizon=1 if self.config.get('bootstrap_allowed') is False
            else self.config['horizon'], gamma=self.gamma)
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
        if self.config.get('bootstrap_allowed') is False:
            return self._sample_episode_returns(count, device)
        terminal_count = int(count * self.config.get('terminal_batch_fraction', 0.))
        if not terminal_count:
            return self._sample_uniform_rows(count, device)
        regions = [r for r in REGIONS if self.episodes[r]]
        if count < 1 or not regions:
            raise ValueError('Cannot sample empty measured TRAIN credit')
        uniform = self._sample_uniform_rows(count - terminal_count, 'cpu')
        terminal = []
        for index in range(terminal_count):
            episodes = self.episodes[regions[index % len(regions)]]
            episode = episodes[int(torch.randint(len(episodes), (1,)))]
            identity = episode['identity']
            if identity not in self._windows:
                self._windows[identity] = discounted_episode_windows(episode['rows'],
                    horizon=self.config['horizon'], gamma=self.gamma)
            terminal.append({k: v[-1] for k, v in self._windows[identity].items()})
        order = torch.randperm(count)
        return {k: torch.cat((uniform[k], torch.stack([r[k] for r in terminal])))[order].to(device)
            for k in uniform}

    def _sample_episode_returns(self, count, device):
        regions = [r for r in REGIONS if self.episodes[r]]
        if count < 1 or not regions:
            raise ValueError('Cannot sample empty measured TRAIN credit')
        selected = []
        for i in range(count):
            episodes = self.episodes[regions[i % len(regions)]]
            if self.config.get('success_fraction_per_region_when_both_classes') is not None:
                # Alternate within each region. A region with no real success
                # contributes failures only; never manufacture a successful row.
                want_success = (i // len(regions)) % 2 == 0
                selected_class = [e for e in episodes
                                  if bool(e['outcome']['result']['success']) == want_success]
                episodes = selected_class or episodes
            episode = episodes[int(torch.randint(len(episodes), (1,)))]
            identity = episode['identity']
            if identity not in self._windows:
                self._windows[identity] = completed_episode_returns(episode['rows'], gamma=self.gamma)
            row = int(torch.randint(len(episode['rows']['reward']), (1,)))
            selected.append({k: v[row] for k, v in self._windows[identity].items()})
        order = torch.randperm(count)
        return {k: torch.stack([s[k] for s in selected])[order].to(device) for k in selected[0]}

    def _sample_uniform_rows(self, count, device):
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
        report = dict(rows=self.size, by_region={r: dict(episodes=len(es),
            successes=sum(bool(e['outcome']['result']['success']) for e in es),
            failures=sum(not bool(e['outcome']['result']['success']) for e in es),
            rows=sum(len(e['rows']['reward']) for e in es)) for r, es in self.episodes.items()},
            TRAIN_only=True, evaluation_rows=0)
        if 'terminal_batch_fraction' in self.config:
            report.update(terminal_batch_fraction=self.config['terminal_batch_fraction'],
                terminal_target=self.config['terminal_target'])
        if self.config.get('bootstrap_allowed') is False:
            report.update(auxiliary_target=self.config['auxiliary_target'], bootstrap_allowed=False,
                episode_length_bias_removed=True)
        if 'success_fraction_per_region_when_both_classes' in self.config:
            report.update(success_fraction_per_region_when_both_classes=.5,
                          missing_class_fallback=self.config['missing_class_fallback'])
        return report


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
