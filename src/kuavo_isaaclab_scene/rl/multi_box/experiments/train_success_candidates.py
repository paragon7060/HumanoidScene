"""Bounded model candidates at safe TRAIN terminals, before learner updates.

This is a model archive, not a frozen rollout success or continuation replay.
Only the Drive uploader may prune checksum-verified old candidate checkpoints.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .staged_train_success import validate_success_outcome
from .actor_train_memory import structure_sha256


def collection_mode(pilot, environment):
    sampler = getattr(pilot, 'body_behavior_sampler', None)
    if not getattr(sampler, 'greedy_unselected_policy', False):
        return 'policy_collection_unspecified'
    if not bool(sampler.initialized[environment]):
        return 'no_held_mode_draw'
    if not bool(sampler.selected[environment]):
        return 'greedy_current_policy'
    return ('perceived_contact_exploration' if hasattr(pilot, 'contact_explorer')
            else 'coherent_arm_exploration')


class TrainSuccessCandidates:
    def __init__(self, directory, *, maximum=12, interval_seconds=300., clock=time.monotonic,
                 expected_scopes=()):
        if type(maximum) is not int or not 1 <= maximum <= 24:
            raise ValueError('Candidate limit must be 1..24 per unique run')
        if not math.isfinite(interval_seconds) or interval_seconds < 300:
            raise ValueError('Repeated scope candidates must be at least300 seconds apart')
        self.directory = Path(directory)
        if any(self.directory.glob('success_candidate_*')):
            raise ValueError('Candidate archive requires a new unique run directory')
        self.maximum, self.interval, self.clock = maximum, interval_seconds, clock
        self.expected_scopes = set(expected_scopes)
        if len(self.expected_scopes) > maximum:
            raise ValueError('Candidate limit must reserve one slot per requested TRAIN scope')
        self.scopes, self.identities, self.events = set(), set(), []
        self.last_saved_at = -math.inf

    def capture(self, pilot, outcomes, *, control_step, action_counters):
        if not pilot.training:
            raise ValueError('Only online TRAIN terminals may produce model candidates')
        if tuple(action_counters) != (pilot.actor_updates, pilot.critic_updates):
            raise ValueError('Candidate capture must precede terminal-step learner updates')
        eligible = []
        for outcome in outcomes:
            validate_success_outcome('train', outcome)
            if outcome['result']['steps'] != control_step:
                raise ValueError('Candidate terminal and action control steps differ')
            identity = (outcome['wave'], outcome['environment'], outcome['layout']['seed'])
            if identity in self.identities:
                continue
            eligible.append((identity, outcome))
        if not eligible:
            return None
        scopes = {(o['layout']['target_region'], o['layout'].get('target_box_type')) for _, o in eligible}
        if self.expected_scopes and not scopes <= self.expected_scopes:
            raise ValueError('Success scope was not requested by this run')
        now = self.clock()
        if len(self.events) >= self.maximum or (
                scopes <= self.scopes and now - self.last_saved_at < self.interval):
            return None
        # Repeated common successes must not consume slots needed by a later
        # first success in a previously unseen region/size.
        if scopes <= self.scopes and len(self.events) >= self.maximum - len(self.expected_scopes-self.scopes):
            return None
        number = len(self.events) + 1
        path = self.directory / f'success_candidate_{number:08d}.pt'
        evidence = [deepcopy(o) | dict(collection_policy_mode=collection_mode(pilot, o['environment']))
                    for _, o in eligible]
        state = pilot.checkpoint_state()
        metadata = dict(format='pre_update_safe_TRAIN_terminal_model_candidate_v1',
            candidate=number, checkpoint=path.name, control_step=control_step,
            actor_updates=pilot.actor_updates, critic_updates=pilot.critic_updates,
            evidence=evidence, capture_boundary='after_env_step_before_learner_observe',
            model_tensor_SHA256=structure_sha256(state['model']),
            model_current_at_terminal_action=True,
            terminal_action_from_greedy_actor_only=all(
                o['collection_policy_mode']=='greedy_current_policy' for o in evidence),
            terminal_action_may_include_exploration_override=any(
                o['collection_policy_mode'] in ('perceived_contact_exploration','coherent_arm_exploration')
                for o in evidence),
            entire_episode_used_one_frozen_model=False,
            fixed_DEV_success_verified=False, direct_training_resume_allowed=False,
            replay_snapshot_included=False, independent_FINAL_used=False)
        pending = path.with_suffix('.pt.pending')
        sidecar = path.with_suffix('.json')
        if path.exists() or pending.exists() or sidecar.exists():
            raise ValueError('Refusing to overwrite an immutable candidate')
        torch.save(state | dict(TRAIN_success_candidate=metadata,
            format_version=1,iteration=pilot.critic_updates), pending)
        digest = hashlib.sha256()
        with pending.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        event = metadata | dict(SHA256=digest.hexdigest(), bytes=pending.stat().st_size)
        sidecar_pending = sidecar.with_suffix('.json.pending')
        sidecar_pending.write_text(json.dumps(event, indent=2, allow_nan=False)+'\n')
        # The sidecar is complete before the uploader can see the .pt file.
        sidecar_pending.replace(sidecar)
        pending.replace(path)
        self.events.append(event)
        self.identities.update(identity for identity, _ in eligible)
        self.scopes.update(scopes)
        self.last_saved_at = now
        return deepcopy(event)

    def report(self):
        return deepcopy(dict(maximum_per_run=self.maximum, repeated_scope_interval_seconds=self.interval,
            reserved_unseen_scopes=[list(x) for x in sorted(self.expected_scopes-self.scopes)],
            saved=len(self.events), events=self.events, fixed_DEV_required=True,
            only_verified_Drive_uploader_prunes=True, no_raw_replay_HDF_archive=True))
