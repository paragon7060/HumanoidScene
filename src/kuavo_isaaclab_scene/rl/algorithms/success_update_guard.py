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
    from .success_cohort_guard import VARIANT as COHORT_VARIANT, STABLE_VARIANT, TUBE_VARIANT
    if variant not in (VARIANT, COHORT_VARIANT, STABLE_VARIANT, TUBE_VARIANT):
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
    if variant in (COHORT_VARIANT, STABLE_VARIANT, TUBE_VARIANT):
        result.update(name=variant, format_version=2,
            source='persistent_first_two_safe_TRAIN_paths_per_region_and_size',
            grouping='each_episode_approach_and_tail64_with_separate_arms_and_jaws',
            absolute_loss_tolerance=1e-9, relative_loss_tolerance=1e-6,
            loss_ceiling='lowest_accepted_loss_on_same_fixed_cohort_not_previous_random_batch',
            protected_correct_jaws='each_row_hand_correct_label_cannot_become_wrong',
            first_success_paths_survive_replay_eviction=True,
            actor_training_minibatch_and_Q_replay_unchanged=True,
            absent_success_batch='pilot_waits_for_first_completed_safe_TRAIN_success')
    if variant in (STABLE_VARIANT, TUBE_VARIANT):
        result.update(format_version=3,
            guard_forward_batch='fixed_each_episode_phase_independent_of_added_cohort_paths',
            loss_reduction='float64_from_unchanged_float32_production_outputs',
            guard_loss_tolerances_unchanged=True,
            candidate_rejection_reasons_recorded=True,
            candidate_parameter_scales=[2.**-i for i in range(11)],
            independent_region_heads_backtracked_separately=True,
            rejected_region_restores_its_Adam_moments=True,
            final_whole_cohort_check_required=True,
            parameter_scale_statistic='minimum_nonzero_accepted_region_scale')
    if variant == TUBE_VARIANT:
        result.update(format_version=4, guard_loss_tolerances_unchanged=False,
            loss_ceiling='fixed_per_row_joint_executed_TRAIN_command_error_limits',
            normalized_goal_error_budget=.001,
            normalized_servo_error_budget=dict(approach=.02, tail64=.01),
            jaw_loss_budget=.002,
            scalar_body_metric='maximum_row_joint_goal_or_bounded_servo_error_ratio',
            projection='actual_Adam_displacement_with_remaining_fixed_constraint_headroom',
            budgets_do_not_accumulate_or_tighten=True,
            task_success_safety_and_randomization_unchanged=True)
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


def repair_scaled_body_candidate(agent, batch, groups, parameters, previous, proposal,
                                 trust_norm, candidate_metrics):
    """Repair the actual rounded, scaled candidate without scaling away repair.

    Called only after ordinary and full-proposal quantized backtracking fail.
    Acceptance retains every original hard predicate and projected Adam budget.
    """
    from . import success_cohort_guard as cohort
    return _repair_scaled_output_candidate(agent, batch, groups, parameters, previous,
        proposal, trust_norm, candidate_metrics, cohort.quantized_body_output_correction)


def repair_scaled_jaw_candidate(agent, batch, groups, parameters, previous, proposal,
                                trust_norm, candidate_metrics):
    """Repair only final jaw rows of an already rejected rounded candidate.

    A body violation forbids this correction. Body repair and its statistics
    remain separate; the existing Adam budget and whole-cohort checks decide.
    """
    from ..multi_box.experiments.jaw_output_repair import jaw_output_correction
    return _repair_scaled_output_candidate(agent, batch, groups, parameters, previous,
        proposal, trust_norm, candidate_metrics, jaw_output_correction)


def _repair_scaled_output_candidate(agent, batch, groups, parameters, previous, proposal,
                                    trust_norm, candidate_metrics, corrector):
    from . import success_cohort_guard as cohort
    details = dict(attempted=False, accepted=False, trials=[],
                   candidate_origin_scales=[2.**-6, 2.**-8, 2.**-10])
    if not cohort.tube(agent) or not bool(trust_norm > 0):
        return 0., details
    for scale in details['candidate_origin_scales']:
        with torch.no_grad():
            for p, old, new in zip(parameters, previous, proposal):
                p.copy_(old + scale*(new-old))
        for cycle in range(4):
            values, jaws = candidate_metrics()
            if cohort.within_ceilings(values, jaws, agent.success_guard_memory, agent.actor_success_guard_config):
                break
            correction, kinds = corrector(agent, batch, groups, parameters)
            if correction is None:
                break
            details['attempted'] = True
            current = torch.cat([(p.detach()-old).flatten() for p, old in zip(parameters, previous)])
            best_score = cohort.restoration_score(values, jaws, agent.success_guard_memory, agent.actor_success_guard_config)
            best = [p.detach().clone() for p in parameters]; selected_gain = 0.
            for gain in (1., 2., 4., 8.):
                adjusted = current + gain*correction
                adjusted *= torch.minimum(adjusted.new_tensor(1.), trust_norm/adjusted.norm().clamp_min(1e-30))
                offset = 0
                with torch.no_grad():
                    for p, old in zip(parameters, previous):
                        count = old.numel(); p.copy_(old+adjusted[offset:offset+count].reshape_as(old)); offset += count
                trial, correct = candidate_metrics()
                score = cohort.restoration_score(trial, correct, agent.success_guard_memory, agent.actor_success_guard_config)
                if score < best_score:
                    best_score = score; best = [p.detach().clone() for p in parameters]; selected_gain = gain
                if cohort.within_ceilings(trial, correct, agent.success_guard_memory, agent.actor_success_guard_config):
                    break
            with torch.no_grad():
                for p, value in zip(parameters, best):
                    p.copy_(value)
            details['trials'].append(dict(origin_scale=scale, cycle=cycle, selected_gain=selected_gain,
                                         restoration_score=list(best_score), **kinds))
        values, correct = candidate_metrics()
        moved = any(not torch.equal(p.detach(), old) for p, old in zip(parameters, previous))
        actual_norm = torch.cat([(p.detach()-old).flatten() for p, old in zip(parameters, previous)]).norm()
        in_budget = bool(actual_norm <= trust_norm*(1+1e-5)+1e-10)
        if moved and in_budget and cohort.within_ceilings(values, correct, agent.success_guard_memory, agent.actor_success_guard_config):
            details.update(accepted=True, origin_scale=scale, actual_delta_norm=float(actual_norm),
                           original_projected_Adam_norm=float(trust_norm))
            return scale, details
    with torch.no_grad():
        for p, old in zip(parameters, previous):
            p.copy_(old)
    return 0., details


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
        before, before_correct = cohort.metrics(agent, successful, groups)
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
    checks=[]
    region_scales={}
    region_trust_norms={}
    repairs=[]
    quantum_fallback_regions={}
    postscale_fallback_regions={}
    rounded_jaw_fallback_regions={}
    final_cohort_check_executed=False
    baseline_allowed=(cohort.within_ceilings(before,before_correct,agent.success_guard_memory,config)
                      if cohort.stable(agent) else None)
    try:
        optimize(agent.actor_optimizer, loss, parameters)
        proposal = [p.detach().clone() for p in parameters]
        delta = torch.cat([(new-old).flatten() for new, old in zip(proposal, previous)])
        projected = 0
        for _ in range(config['projection_cycles']):
            for constraint in constraints:
                gradient, headroom = constraint if isinstance(constraint, tuple) else (constraint, 0.)
                ascent = torch.dot(gradient, delta)
                if ascent > headroom:
                    delta = delta - (ascent-headroom) / gradient.square().sum() * gradient
                    projected += 1
        # Remove float32 projection residue, not a policy/logit deadband.
        # This is below 1e-6 of the production actor learning rate.
        delta = torch.where(delta.abs() < 1e-12, 0., delta)
        offset = 0
        for index, old in enumerate(previous):
            count = old.numel()
            proposal[index] = old + delta[offset:offset+count].reshape_as(old)
            offset += count
        if cohort.stable(agent) and hasattr(agent.actor.network, 'heads'):
            from ..multi_box.experiments.independent_actor_heads import parameter_partitions
            partitions=parameter_partitions(agent.actor)
        else:
            partitions={'all':list(range(len(parameters)))}
        metric_cache=None
        if cohort.stable(agent) and 'all' not in partitions:
            metric_cache=cohort.RegionalMetricCache(agent,successful,groups,before,before_correct)
        with torch.no_grad():
            for p,old in zip(parameters,previous):p.copy_(old)
        for region,indices in partitions.items():
            local_scale=0.
            initial_delta=torch.cat([(proposal[i]-previous[i]).flatten() for i in indices])
            trust_norm=initial_delta.norm()
            region_trust_norms[region]=float(trust_norm)
            def candidate_metrics():
                return (metric_cache.evaluate(region) if metric_cache is not None else
                        cohort.metrics(agent,successful,groups))
            def repair_candidate(corrector, reserve_ULPs=0):
                attempted=False
                local_parameters=[parameters[i] for i in indices]
                if trust_norm > 0:
                    with torch.no_grad():
                        for i in indices:parameters[i].copy_(proposal[i])
                    for cycle in range(config['projection_cycles']):
                        candidate,correct=candidate_metrics()
                        if cohort.within_ceilings(candidate,correct,agent.success_guard_memory,config):
                            break
                        correction,kinds=corrector(agent,successful,
                            metric_cache.groups[region] if metric_cache is not None else groups,
                            local_parameters)
                        repairs.append(dict(region=region,cycle=cycle,
                            internal_body_reserve_ULPs=reserve_ULPs,**kinds))
                        if correction is None:
                            break
                        attempted=True
                        current=torch.cat([(parameters[i].detach()-previous[i]).flatten() for i in indices])
                        # An exact-bound correction can stall when the max
                        # row changes or a float32 drive goal does not move.
                        # Search inward only for body-only violations. Every
                        # trial keeps the original projected Adam trust norm;
                        # the original whole-cohort hard checks still decide.
                        gains=(1.,2.,4.) if kinds['body'] and not kinds['jaw_loss'] and not kinds['jaw_label'] else (1.,)
                        best_score=cohort.restoration_score(candidate,correct,agent.success_guard_memory,config)
                        best=[parameters[i].detach().clone() for i in indices]
                        selected_gain=0.
                        for trial_count,gain in enumerate(gains,1):
                            adjusted=current+gain*correction
                            adjusted=adjusted*torch.minimum(adjusted.new_tensor(1.),
                                trust_norm/adjusted.norm().clamp_min(1e-30))
                            offset=0
                            with torch.no_grad():
                                for i in indices:
                                    count=previous[i].numel()
                                    parameters[i].copy_(previous[i]+adjusted[offset:offset+count].reshape_as(previous[i]))
                                    offset+=count
                            trial,jaws=candidate_metrics()
                            score=cohort.restoration_score(trial,jaws,agent.success_guard_memory,config)
                            if len(gains)==1 or score<best_score:
                                best_score=score;selected_gain=gain
                                best=[parameters[i].detach().clone() for i in indices]
                            if cohort.within_ceilings(trial,jaws,agent.success_guard_memory,config):
                                break
                        with torch.no_grad():
                            for i,value in zip(indices,best):parameters[i].copy_(value)
                        repairs[-1].update(restoration_trials=trial_count,
                            selected_restoration_gain=selected_gain,
                            restoration_score_after=list(best_score))
                    for i in indices:proposal[i]=parameters[i].detach().clone()
                    with torch.no_grad():
                        for i in indices:parameters[i].copy_(previous[i])
                return attempted

            def backtrack_candidate(reserve_ULPs=0):
                for scale in config['candidate_parameter_scales']:
                    with torch.no_grad():
                        for i in indices:
                            parameters[i].copy_(proposal[i] if scale==1. else previous[i]+scale*(proposal[i]-previous[i]))
                    if persistent:
                        candidate, correct = candidate_metrics()
                        allowed = cohort.within_ceilings(candidate, correct, agent.success_guard_memory, config)
                        if cohort.stable(agent):
                            checks.append(dict(region=region,scale=scale,allowed=allowed,
                                internal_body_reserve_ULPs=reserve_ULPs,
                                **cohort.violations(candidate,correct,agent.success_guard_memory,config)))
                    else:
                        candidate = success_metrics(agent, successful)
                        allowed = acceptable(before, candidate, config)
                    if allowed:
                        moved=any(not torch.equal(parameters[i].detach(),previous[i]) for i in indices)
                        return (scale if moved or not cohort.stable(agent) else 0.),candidate
                return 0.,None

            if cohort.tube(agent) and baseline_allowed:
                repair_candidate(cohort.violation_correction)
            local_scale,checked=backtrack_candidate()
            if checked is not None:after=checked
            if not local_scale and cohort.tube(agent) and baseline_allowed and trust_norm > 0:
                # Keep an accepted ordinary proposal exactly. Only an already
                # rejected region gets a body-only, quantization-aware inward
                # repair; every original fixed hard check still applies.
                attempted=repair_candidate(cohort.quantized_body_correction, reserve_ULPs=2)
                if attempted:
                    local_scale,checked=backtrack_candidate(reserve_ULPs=2)
                    if checked is not None:after=checked
                quantum_fallback_regions[region]=dict(attempted=attempted,accepted=bool(local_scale))
            if not local_scale and cohort.tube(agent) and baseline_allowed and trust_norm > 0:
                local_scale,details=repair_scaled_body_candidate(agent,successful,
                    metric_cache.groups[region] if metric_cache is not None else groups,
                    [parameters[i] for i in indices],[previous[i] for i in indices],
                    [proposal[i] for i in indices],trust_norm,candidate_metrics)
                postscale_fallback_regions[region]=details
                if local_scale:
                    after,_=candidate_metrics()
            if not local_scale and cohort.tube(agent) and baseline_allowed and trust_norm > 0 \
                    and getattr(agent, 'actor_rounded_jaw_repair', False):
                local_scale,details=repair_scaled_jaw_candidate(agent,successful,
                    metric_cache.groups[region] if metric_cache is not None else groups,
                    [parameters[i] for i in indices],[previous[i] for i in indices],
                    [proposal[i] for i in indices],trust_norm,candidate_metrics)
                rounded_jaw_fallback_regions[region]=details
                if local_scale:
                    after,_=candidate_metrics()
            region_scales[region]=local_scale
            if not local_scale:
                with torch.no_grad():
                    for i in indices:parameters[i].copy_(previous[i])
                # A rejected region must not advance its Adam momentum just
                # because another independent region had an accepted step.
                for i in indices:
                    pid=optimizer['param_groups'][0]['params'][i]
                    if pid in optimizer['state']:
                        agent.actor_optimizer.state[parameters[i]]=deepcopy(optimizer['state'][pid])
                    else:
                        agent.actor_optimizer.state.pop(parameters[i],None)
            elif metric_cache is not None:
                metric_cache.commit(region)
        accepted_scale=min((s for s in region_scales.values() if s),default=0.)
        if persistent and accepted_scale:
            final_cohort_check_executed=True
            after,correct=cohort.metrics(agent,successful,groups)
            if not cohort.within_ceilings(after,correct,agent.success_guard_memory,config):
                accepted_scale=0.
            else:cohort.tighten(agent.success_guard_memory,after,correct,fixed=cohort.tube(agent))
    finally:
        if not accepted_scale:
            with torch.no_grad():
                for p, old in zip(parameters, previous):
                    p.copy_(old)
            agent.actor_optimizer.load_state_dict(optimizer)
            region_scales={region:0. for region in region_scales}
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
    if cohort.stable(agent):
        stats.update(actor_success_guard_baseline_allowed=baseline_allowed,
                     actor_success_guard_candidate_checks=checks,
                     actor_success_guard_region_parameter_scales=region_scales,
                     actor_success_guard_region_projected_Adam_norms=region_trust_norms,
                     actor_success_guard_region_actual_delta_norms={region:float(torch.cat([
                         (parameters[i].detach()-previous[i]).flatten() for i in indices]).norm())
                         for region,indices in partitions.items()},
                     actor_success_guard_parameter_delta_norm=float(torch.cat([
                         (p.detach()-old).flatten() for p,old in zip(parameters,previous)]).norm()))
        if metric_cache is not None:
            stats['actor_success_guard_candidate_metric_cache']=dict(
                independent_heads_only=True,fixed_phase_forward_shapes=True,
                evaluations=metric_cache.evaluations,phase_forwards=metric_cache.phase_forwards,
                final_whole_cohort_check_required_for_acceptance=True,
                final_whole_cohort_check_executed=final_cohort_check_executed)
        counters['baseline_outside_fixed_ceiling']=counters.get('baseline_outside_fixed_ceiling',0)+int(not baseline_allowed)
        counts=counters.setdefault('region_parameter_scale_counts',{})
        for region,scale in region_scales.items():
            local=counts.setdefault(region,{})
            key=str(scale);local[key]=local.get(key,0)+1
    if cohort.tube(agent):
        if getattr(agent, 'actor_rounded_jaw_repair', False):
            stats['actor_success_guard_rounded_jaw_fallback_regions']={
                region:dict(value,accepted=value['accepted'] and bool(accepted_scale))
                for region,value in rounded_jaw_fallback_regions.items()}
            for name in ('attempted','accepted'):
                key='rounded_jaw_fallback_regions_'+name
                counters[key]=counters.get(key,0)+sum(value[name] for value in
                    stats['actor_success_guard_rounded_jaw_fallback_regions'].values())
        stats['actor_success_guard_postscale_body_fallback_regions']={
            region:dict(value,accepted=value['accepted'] and bool(accepted_scale))
            for region,value in postscale_fallback_regions.items()}
        stats['actor_success_guard_repaired_scale_semantics']='origin scale before internal correction; use actual_delta_norm for displacement'
        for name in ('attempted','accepted'):
            key='postscale_body_fallback_regions_'+name
            counters[key]=counters.get(key,0)+sum(value[name] for value in
                stats['actor_success_guard_postscale_body_fallback_regions'].values())
        stats['actor_success_guard_candidate_repairs']=repairs
        stats['actor_success_guard_quantized_body_fallback_regions']={
            region:dict(attempted=value['attempted'],accepted=value['accepted'] and bool(accepted_scale))
            for region,value in quantum_fallback_regions.items()}
        for name in ('attempted','accepted'):
            key='quantized_body_fallback_regions_'+name
            counters[key]=counters.get(key,0)+sum(value[name] for value in
                stats['actor_success_guard_quantized_body_fallback_regions'].values())
        counters['candidate_repair_cycles']=counters.get('candidate_repair_cycles',0)+len(repairs)
        counts=counters.setdefault('candidate_repair_constraint_counts',{})
        for repair in repairs:
            for kind in ('body','jaw_loss','jaw_label'):
                counts[kind]=counts.get(kind,0)+repair[kind]
            counters['restoration_trials']=counters.get('restoration_trials',0)+repair.get('restoration_trials',0)
            selected=counters.setdefault('restoration_gain_counts',{})
            gain=str(repair.get('selected_restoration_gain',0.))
            selected[gain]=selected.get(gain,0)+1
    return bool(accepted_scale), stats


def validate_success_update_guard_state(state):
    config = state.get('actor_success_guard')
    if (config is not None and config != success_update_guard_config(config.get('name'))) \
            or state.get('goal_contract', {}).get('actor_success_guard') != config \
            or state.get('hybrid_contract', {}).get('actor_success_guard') != config:
        raise ValueError('Saved actor success guard contract differs')
    if config is not None and state['config'].get('freeze_actor_normalizer') is not True:
        raise ValueError('Saved guarded actor cannot change its observation normalization')
