"""Opt-in jaw coverage for real TRAIN collection, outside the SAC policy.

Reuse the two existing correlated Gaussian jaw noises. At a fixed state
their stationary uniform marginals give (1-epsilon)*policy + epsilon*U4.
Feedback/history still make collection off-policy; neither Q targets nor
actor expectations use this behavior distribution. The usual physical
projector merges branches when a jaw is outside its production near gate.
"""
from copy import deepcopy
import math

import torch


VARIANT = 'joint-epsilon10'


def jaw_behavior_config(variant):
    if variant in (None, 'policy'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown TRAIN jaw behavior variant')
    return dict(variant=VARIANT, format_version=1, uniform_joint_fraction=.1,
        scope='actual_flap_real_TRAIN_collection_only',
        noise_source='existing_correlated_two_jaw_gaussians_no_extra_rng',
        learned_branch_left_uniform='conditional_rescale_after_mixture_selection',
        branches=['open_open', 'open_close', 'close_open', 'close_close'],
        projection='unchanged_production_nominal_near_gate',
        body_policy_actor_Q_targets_and_eval_unchanged=True)


class JointJawBehaviorExploration:
    def __init__(self, config, statistics=None):
        if config != jaw_behavior_config(VARIANT):
            raise ValueError('TRAIN jaw behavior configuration differs')
        self.config = deepcopy(config)
        self.statistics = dict(rows=0, uniform_joint_rows=0,
            projected_branch_counts=[0, 0, 0, 0])
        if statistics is not None:
            if set(statistics) != set(self.statistics) or any(
                    type(statistics[key]) is not int or statistics[key] < 0
                    for key in ('rows', 'uniform_joint_rows')) or \
                    len(statistics['projected_branch_counts']) != 4 or any(
                        type(v) is not int or v < 0 for v in statistics['projected_branch_counts']) or \
                    sum(statistics['projected_branch_counts']) != statistics['rows'] or \
                    statistics['uniform_joint_rows'] > statistics['rows']:
                raise ValueError('Malformed TRAIN jaw behavior statistics')
            self.statistics = deepcopy(statistics)

    @torch.no_grad()
    def act(self, agent, observation, noise, *, body_latent_offset=None):
        if noise.shape != (len(observation), 21) or not torch.isfinite(noise).all():
            raise ValueError('One finite21-D behavior noise per environment required')
        normalized = agent.actor_normalizer(agent.actor_features(observation))
        body, _, logits = agent.continuous_sample(normalized, noise=noise[:, :19],
            body_latent_offset=body_latent_offset, raw=observation)
        uniform = .5*(1+torch.erf(noise[:, 19:21]/math.sqrt(2)))
        epsilon = self.config['uniform_joint_fraction']
        explore = uniform[:, 0] < epsilon
        # The selection event truncates uL. Undo that truncation for the
        # learned branch; otherwise its left-jaw marginal would be biased.
        policy_uniform = torch.stack(((uniform[:, 0]-epsilon)/(1-epsilon), uniform[:, 1]), -1)
        policy_closed = policy_uniform < logits.sigmoid()
        branch = (uniform[:, 1]*4).long().clamp(0, 3)
        uniform_closed = torch.stack((branch >= 2, branch.remainder(2).bool()), -1)
        closed = torch.where(explore[:, None], uniform_closed, policy_closed)
        action = agent.projected_command(observation, body, closed)
        index = (action[:, 19] > 0).long()*2+(action[:, 20] > 0).long()
        counts = torch.bincount(index, minlength=4).tolist()
        self.statistics['rows'] += len(observation)
        self.statistics['uniform_joint_rows'] += int(explore.sum())
        self.statistics['projected_branch_counts'] = [
            a+b for a, b in zip(self.statistics['projected_branch_counts'], counts)]
        return action

    def report(self):
        return deepcopy(self.statistics)
