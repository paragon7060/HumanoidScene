"""Check the actual Adam proposal against completed successful TRAIN commands.

This is a local update safeguard, not a guarantee of rollout success. No Q
targets or recorded actions are changed. Rejection restores both parameters
and optimizer moments; accepted interpolation retains the proposed moments.
"""
from copy import deepcopy

import torch
from torch.nn import functional as F

from .common import optimize

VARIANT = 'train-success-Adam-backtrack'


def success_update_guard_config(variant):
    if variant is None:
        return None
    from .success_cohort_guard import VARIANT as COHORT_VARIANT
    if variant not in (VARIANT, COHORT_VARIANT):
        raise ValueError('Unknown successful TRAIN update guard')
    result = dict(name=VARIANT, format_version=1,
        source='same_completed_safe_TRAIN_actor_batch_only',
        grouping='each_available_rack_region',
        protect_body_and_jaw_losses_separately=True,
        protect_actual_greedy_jaw_label_agreement=True,
        absolute_loss_tolerance=1e-9, relative_loss_tolerance=1e-6,
        project_actual_Adam_displacement_against_regional_body_and_jaw_gradients=True,
        per_hand_near_jaw_gradients_also_projected=True,
        projection_cycles=4,
        candidate_parameter_scales=[1., .5, .25, .125, .0625, .03125, .015625],
        reject_restores_parameters_and_Adam_moments=True,
        accepted_interpolation_keeps_proposed_Adam_moments=True,
        reject_skips_actor_counter_and_temperature_updates=True,
        critic_and_replay_updates_unchanged=True,
        freeze_actor_normalizer_required=True,
        absent_success_batch='ordinary_actor_update',
        pilot_waits_for_first_completed_safe_TRAIN_success=True,
        local_batch_protection_is_not_rollout_success_guarantee=True)
    if variant == COHORT_VARIANT:
        result.update(name=variant, format_version=2,
            source='persistent_first_two_safe_TRAIN_paths_per_region_and_size',
            grouping='each_episode_approach_and_tail64_with_separate_arms_and_jaws',
            absolute_loss_tolerance=1e-9, relative_loss_tolerance=1e-6,
            loss_ceiling='lowest_accepted_loss_on_same_fixed_cohort_not_previous_random_batch',
            protected_correct_jaws='each_row_hand_correct_label_cannot_become_wrong',
            first_success_paths_survive_replay_eviction=True,
            actor_training_minibatch_and_Q_replay_unchanged=True,
            absent_success_batch='pilot_waits_for_first_completed_safe_TRAIN_success')
    return result


@torch.no_grad()
def success_metrics(agent, successful):
    raw, labels = successful['actor_obs'], successful['action']
    regions = raw[:, 94:98].argmax(-1)
    result = {}
    for region in regions.unique().tolist():
        selected = regions == region
        obs, actions = raw[selected], labels[selected]
        normalized = agent.actor_normalizer(agent.actor_features(obs))
        mean, _, logits = agent.continuous_parameters(normalized, obs)
        body_loss = agent.success_body_loss(obs, mean.tanh(), actions)
        jaw_loss, _ = agent.successful_jaw_loss(obs, logits, actions)
        executed = agent.act(obs, deterministic=True)
        result[str(region)] = dict(body=body_loss.item(), jaw=jaw_loss.item(),
            jaw_errors=int((executed[:, 19:] != actions[:, 19:]).sum()))
    return result


def retention_gradients(agent, successful, parameters):
    """Local retention constraints, evaluated before the Adam proposal."""
    raw, labels = successful['actor_obs'], successful['action']
    regions = raw[:, 94:98].argmax(-1)
    gradients = []
    for region in regions.unique().tolist():
        chosen = regions == region
        obs, actions = raw[chosen], labels[chosen]
        normalized = agent.actor_normalizer(agent.actor_features(obs))
        mean, _, logits = agent.continuous_parameters(normalized, obs)
        body = agent.success_body_loss(obs, mean.tanh(), actions)
        jaw, _ = agent.successful_jaw_loss(obs, logits, actions)
        near = agent.action_projector.entropy_mask(obs)[:, 19:].bool()
        per_hand = [F.binary_cross_entropy_with_logits(logits[near[:, hand], hand],
            (actions[near[:, hand], 19+hand]+1)/2)
            for hand in range(2) if bool(near[:, hand].any())]
        for loss in (body, jaw, *per_hand):
            if not loss.requires_grad:
                continue
            parts = torch.autograd.grad(loss, parameters, allow_unused=True, retain_graph=True)
            gradient = torch.cat([(torch.zeros_like(p) if g is None else g).detach().flatten()
                                  for p, g in zip(parameters, parts)])
            if not torch.isfinite(gradient).all():
                raise FloatingPointError('Non-finite successful TRAIN constraint gradient')
            if gradient.square().sum() > 1e-20:
                gradients.append(gradient)
    return gradients


def acceptable(before, after, config):
    if before.keys() != after.keys():
        return False
    for region, initial in before.items():
        candidate = after[region]
        if candidate['jaw_errors'] > initial['jaw_errors']:
            return False
        for key in ('body', 'jaw'):
            tolerance = config['absolute_loss_tolerance'] + config['relative_loss_tolerance'] * abs(initial[key])
            if not torch.isfinite(torch.tensor(candidate[key])) or candidate[key] > initial[key] + tolerance:
                return False
    return True


def guarded_actor_step(agent, loss, successful):
    config = getattr(agent, 'actor_success_guard_config', None)
    from . import success_cohort_guard as cohort
    persistent = cohort.enabled(agent)
    if config is None or (successful is None and not persistent):
        optimize(agent.actor_optimizer, loss, agent.actor.parameters())
        return True, {}
    if config != success_update_guard_config(config['name']) or not agent.config.freeze_actor_normalizer:
        raise ValueError('Success guard requires its exact contract and frozen actor normalization')
    if agent.actor_obs_dim != 518 or (not persistent and not len(successful['action'])):
        raise ValueError('Success guard requires measured held TRAIN actor states')
    previous_servo_statistics = deepcopy(getattr(agent, '_success_servo_statistics', None))
    if persistent:
        successful, groups = cohort.cohort_batch(agent)
        before, _ = cohort.metrics(agent, successful, groups)
    else:
        before = success_metrics(agent, successful)
    parameters = list(agent.actor.parameters())
    previous = [p.detach().clone() for p in parameters]
    optimizer = deepcopy(agent.actor_optimizer.state_dict())
    constraints = (cohort.gradient_constraints(agent, successful, groups, parameters)
                   if persistent else retention_gradients(agent, successful, parameters))
    # Metrics may update diagnostic attributes, but never parameters or RNG.
    accepted_scale = 0.
    after = before
    try:
        optimize(agent.actor_optimizer, loss, parameters)
        proposal = [p.detach().clone() for p in parameters]
        delta = torch.cat([(new-old).flatten() for new, old in zip(proposal, previous)])
        projected = 0
        for _ in range(config['projection_cycles']):
            for gradient in constraints:
                ascent = torch.dot(gradient, delta)
                if ascent > 0:
                    delta = delta - ascent / gradient.square().sum() * gradient
                    projected += 1
        # Remove float32 projection residue, not a policy/logit deadband.
        # This is below 1e-6 of the production actor learning rate.
        delta = torch.where(delta.abs() < 1e-12, 0., delta)
        offset = 0
        for index, old in enumerate(previous):
            count = old.numel()
            proposal[index] = old + delta[offset:offset+count].reshape_as(old)
            offset += count
        for scale in config['candidate_parameter_scales']:
            with torch.no_grad():
                for p, old, new in zip(parameters, previous, proposal):
                    p.copy_(new if scale == 1. else old + scale * (new - old))
            if persistent:
                candidate, correct = cohort.metrics(agent, successful, groups)
                allowed = cohort.within_ceilings(candidate, correct, agent.success_guard_memory, config)
            else:
                candidate = success_metrics(agent, successful)
                allowed = acceptable(before, candidate, config)
            if allowed:
                accepted_scale, after = scale, candidate
                if persistent:
                    cohort.tighten(agent.success_guard_memory, candidate, correct)
                break
    finally:
        if not accepted_scale:
            with torch.no_grad():
                for p, old in zip(parameters, previous):
                    p.copy_(old)
            agent.actor_optimizer.load_state_dict(optimizer)
        if previous_servo_statistics is not None:
            agent._success_servo_statistics = previous_servo_statistics
    stats = dict(actor_success_guard_checked=True,
        actor_success_guard_accepted=bool(accepted_scale),
        actor_success_guard_parameter_scale=accepted_scale,
        actor_success_guard_projected_constraints=projected,
        actor_success_guard_regions=len({e['outcome']['layout']['target_region']
            for e in agent.success_guard_memory['entries']}) if persistent else len(before),
        actor_success_guard_before=before, actor_success_guard_after=after)
    counters = getattr(agent, 'success_guard_statistics', None)
    if not counters:
        counters = agent.success_guard_statistics = dict(attempted=0, accepted=0, rejected=0,
            projected_constraints=0, parameter_scale_counts={})
    counters['attempted'] += 1
    counters['accepted' if accepted_scale else 'rejected'] += 1
    counters['projected_constraints'] += projected
    key = str(accepted_scale)
    counters['parameter_scale_counts'][key] = counters['parameter_scale_counts'].get(key, 0)+1
    if persistent:
        stats['actor_success_guard_groups'] = len(before)
        stats['actor_success_guard_cohort'] = cohort.report(agent)
    return bool(accepted_scale), stats


def validate_success_update_guard_state(state):
    config = state.get('actor_success_guard')
    if (config is not None and config != success_update_guard_config(config.get('name'))) \
            or state.get('goal_contract', {}).get('actor_success_guard') != config \
            or state.get('hybrid_contract', {}).get('actor_success_guard') != config:
        raise ValueError('Saved actor success guard contract differs')
    if config is not None and state['config'].get('freeze_actor_normalizer') is not True:
        raise ValueError('Saved guarded actor cannot change its observation normalization')
