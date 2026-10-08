"""Coherent arm coverage in real actual-flap TRAIN collection only.

An episode keeps one bias direction and smoothly enters it. This is an
off-policy collection distribution: actor/Q targets, greedy evaluation and
the affine physical goal envelope do not use this sampler.
"""
from copy import deepcopy

import torch


VARIANT = 'ramped-arm-bias20'
GREEDY_REST_VARIANT = 'arm20-explore-rest-greedy'
GENTLE_GREEDY_REST_VARIANT = 'arm20-gentle-rest-greedy'
VARIANTS = (VARIANT, GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT)


def body_behavior_config(variant):
    if variant in (None, 'off'):
        return None
    if variant not in VARIANTS:
        raise ValueError('Unknown TRAIN body behavior variant')
    result = dict(variant=variant, format_version=1,
        scope='actual_flap_real_TRAIN_collection_only', episode_fraction=.2,
        arm_goal_columns=list(range(1, 15)), latent_bias_std=.8,
        max_abs_latent_bias=1.6, ramp_held_steps=90,
        actor_observation_dim=518, held_phase_column=-6,
        unselected_episodes_keep_original_behavior_distribution=True,
        actor_Q_targets_and_greedy_evaluation_unchanged=True,
        affine_goal_radius_and_physical_projection_unchanged=True,
        base_head_torso_bias_zero=True, privileged_inputs=False,
        scene_curriculum=False, active_episode_biases_reset_between_waves=True)
    if variant in (GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT):
        result.update(unselected_episodes_keep_original_behavior_distribution=False,
            unselected_policy_sampling='greedy_current_learned_body_and_binary_jaws',
            selected_policy_sampling='existing_AR1_Gaussian_arm_bias_and_joint_jaw_behavior',
            selection_shared_with_existing_arm_bias_not_an_independent20_percent_mask=True,
            greedy_and_exploring_rows_are_actual_executed_TRAIN_not_demo_or_teacher=True)
    if variant == GENTLE_GREEDY_REST_VARIANT:
        # Full-arm URDF goals span many production servo steps. Measured
        # initial-policy TRAIN states showed that a .2 bias still changed
        # most arm commands by over half their step range. Keep the learned
        # goal envelope intact and reduce only the episode collection bias.
        result.update(latent_bias_std=.01, max_abs_latent_bias=.02)
    return result


def body_behavior_statistics(statistics=None):
    result = dict(episodes_drawn=0, biased_episodes=0,
        held_collection_rows=0, biased_episode_rows=0)
    if statistics is not None:
        if not isinstance(statistics, dict) or set(statistics) != set(result) \
                or any(type(v) is not int or v < 0 for v in statistics.values()) \
                or statistics['biased_episodes'] > statistics['episodes_drawn'] \
                or statistics['biased_episode_rows'] > statistics['held_collection_rows']:
            raise ValueError('Malformed TRAIN body behavior statistics')
        result = deepcopy(statistics)
    return result


def enable_body_behavior(state, experience, *, source_checkpoint):
    """Add future collection metadata without transforming any existing tensor."""
    goal = state.get('goal_contract', {})
    if state.get('artifact_type') != 'staged_actual_flap_bounded_actor_correction_hybrid_sac_v1' \
            or goal != experience.get('goal_contract') \
            or (goal.get('actor_dim'), goal.get('critic_dim')) != (518, 577) \
            or goal.get('body_correction_radius') != .15 or not goal.get('exploration_correlation') \
            or goal.get('gripper_prior_bound') is not False or goal.get('episode_arm_exploration') is not None \
            or state.get('body_behavior') is not None or experience.get('body_behavior') is not None:
        raise ValueError('Matching unmodified actual-flap checkpoint/replay and correlated collection required')
    rows = experience.get('executed_goal_transitions', {})
    if 'reward' not in rows or any(len(v) != len(rows['reward']) for v in rows.values()):
        raise ValueError('Matching actual executed replay required')
    origin = dict(source_checkpoint=str(source_checkpoint),
        actor_updates_at_activation=state['actor_updates'],critic_updates_at_activation=state['critic_updates'],
        old_replay_rows_at_activation=len(rows['reward']),
        model_Q_normalizers_and_four_optimizer_states_kept=True,
        old_replay_kept_with_original_behavior=True,old_rows_not_relabelled=True,
        scope='future_real_TRAIN_collection')
    extras = dict(body_behavior=body_behavior_config(VARIANT), body_behavior_origin=origin,
        body_behavior_statistics=body_behavior_statistics())
    return state | deepcopy(extras), experience | deepcopy(extras)


class RampedArmBehaviorExploration:
    training_only = True

    def __init__(self, num_envs, device, config, statistics=None):
        if (not isinstance(config,dict) or config.get('variant') not in VARIANTS
                or config != body_behavior_config(config['variant'])
                or type(num_envs) is not int or num_envs < 1):
            raise ValueError('Declared body behavior and positive environment count required')
        self.config = deepcopy(config)
        self.greedy_unselected_policy = config['variant'] in (
            GREEDY_REST_VARIANT, GENTLE_GREEDY_REST_VARIANT)
        self.statistics = body_behavior_statistics(statistics)
        self.bias = torch.zeros(num_envs, 14, device=device)
        self.selected = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.initialized = torch.zeros_like(self.selected)
        self.last_clock = torch.full((num_envs,), -1., device=device)

    def _validate_ids(self, ids):
        if not isinstance(ids, torch.Tensor) or ids.ndim != 1 or ids.dtype != torch.long \
                or ids.device != self.bias.device or len(ids.unique()) != len(ids) \
                or (ids < 0).any() or (ids >= len(self.bias)).any():
            raise ValueError('Distinct valid global environment IDs on the learner device required')

    def reset(self, ids):
        """Explicit reset also covers a new episode whose first clock is zero."""
        self._validate_ids(ids)
        self.initialized[ids] = False
        self.selected[ids] = False
        self.bias[ids] = 0
        self.last_clock[ids] = -1

    @torch.no_grad()
    def offset(self, observation, clocks, ids):
        self._validate_ids(ids)
        if observation.shape != (len(ids), 518) or observation.device != self.bias.device \
                or not observation.is_floating_point() or not torch.isfinite(observation).all():
            raise ValueError('Body behavior requires finite measured518-D held observations')
        if not (observation[:, -6] == 1).all():
            raise ValueError('Body behavior requires a physically confirmed held phase')
        if not isinstance(clocks, torch.Tensor):
            clocks = observation.new_full((len(ids),), clocks)
        if clocks.shape != (len(ids),) or clocks.device != observation.device \
                or not torch.isfinite(clocks).all() or (clocks < 0).any():
            raise ValueError('One measured nonnegative held clock per active environment required')
        # Global IDs remain stable when other environments finish. A clock
        # restart denotes a new episode; repeated calls at zero do not redraw.
        new = ids[~self.initialized[ids] | (clocks < self.last_clock[ids])]
        if len(new):
            chosen = torch.rand(len(new), device=self.bias.device) < self.config['episode_fraction']
            bias = (torch.randn(len(new), 14, device=self.bias.device)
                * self.config['latent_bias_std']).clamp(
                    -self.config['max_abs_latent_bias'], self.config['max_abs_latent_bias'])
            self.bias[new] = bias * chosen[:, None]
            self.selected[new] = chosen
            self.initialized[new] = True
            self.statistics['episodes_drawn'] += len(new)
            self.statistics['biased_episodes'] += int(chosen.sum())
        self.last_clock[ids] = clocks.to(self.last_clock)
        u = (clocks.to(observation) / self.config['ramp_held_steps']).clamp(0, 1)
        result = observation.new_zeros(len(ids), 19)
        result[:, 1:15] = self.bias[ids].to(observation) * (u.square() * (3 - 2*u))[:, None]
        self.statistics['held_collection_rows'] += len(ids)
        self.statistics['biased_episode_rows'] += int(self.selected[ids].sum())
        return result

    def report(self):
        return deepcopy(self.statistics)
