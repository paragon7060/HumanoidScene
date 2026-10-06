"""Explicit full-distribution frozen policy comparison across PhysX devices."""

from collections import Counter


SOURCE = 'frozen_physics_backend_policy_eval_NOT_matching_Q_replay'
REGIONS = ('shelf_2_left', 'shelf_2_right', 'shelf_3_left', 'shelf_3_right')
INCOMPATIBLE_FLAGS = frozenset('--' + name for name in (
    'contact-stability-probe', 'tgs-zero-velocity-probe', 'contact-last-probe', 'pgs-probe',
    'gripper-drive-probe', 'centered-world-probe', 'packed-background-probe', 'base-waypoint-probe',
    'reset-solver-probe', 'passive-bearing-probe-layer', 'reset-independent-scene-probe',
    'zero-passive-roller-velocities-probe', 'rear5-support-gap-probe-m', 'grasp-observation-audit',
    'full-distribution-grasp-observation-audit', 'cpu-workplace-probe', 'measured-train-credit', 'jaw-behavior',
    'jaw-saturation-penalty', 'success-jaw-balance', 'stop-on-validation-regression',
))


def validate_frozen_backend_policy_eval(waves, *, enabled, device, training, steps,
                                       explicit_frozen=True, other_probe=False):
    if not enabled:
        return None
    if device not in ('cpu', 'cuda:0') or training or not explicit_frozen or steps != 900 or other_probe:
        raise ValueError('Backend policy evaluation requires explicit frozen CPU/GPU,900steps and unchanged other physics')
    if not isinstance(waves, list) or len(waves) != 1 or not isinstance(waves[0], dict):
        raise ValueError('Backend policy evaluation requires one complete original DEV128 wave')
    wave = waves[0]
    rows = wave.get('layouts')
    if wave.get('split') != 'validation' or wave.get('background_placement', 'original') != 'original' \
            or not isinstance(rows, list) or len(rows) != 128:
        raise ValueError('Backend policy evaluation requires original DEV128; TRAIN/FINAL and changed backgrounds excluded')
    layouts = [row.get('layout', {}) if isinstance(row, dict) else {} for row in rows]
    if not all(isinstance(layout, dict) for layout in layouts):
        raise ValueError('Backend policy evaluation requires explicit layout dictionaries')
    seeds = [layout.get('seed') for layout in layouts]
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds) \
            or len(set(seeds)) != 128 or any(layout.get('split') != 'holdout' for layout in layouts) \
            or Counter(layout.get('target_region') for layout in layouts) != Counter(dict.fromkeys(REGIONS, 32)):
        raise ValueError('Backend policy evaluation requires128 distinct DEV seeds and32 original requests per region')
    return dict(name='frozen_full_DEV_physics_backend_policy_eval_v1', source=SOURCE,
        physics_device=device, training=False, Q_import_eligible=False,
        original_DEV_requested=128, requested_per_region=32, rollout_steps=900,
        requested_box_base_background_randomization_preserved=True,
        physical_parameters_success_and_safety_preserved=True,
        constructor_and_contact_solver_history_not_matched=True,
        exact_flap_randomization_draws_not_matched=True,
        independent_FINAL_used=False)


def frozen_network_snapshot(pilot):
    """Include learned nets/normalizers and every frozen command actor."""
    modules = dict(agent=pilot.agent, warm_agent=pilot.warm_start.agent)
    command_prior = getattr(pilot, 'command_prior', None)
    if command_prior is not None:
        modules.update(prior_actor=command_prior.actor, prior_normalizer=command_prior.normalizer)
    goal_prior = getattr(pilot, 'prior', None)
    if goal_prior is not None:
        modules['goal_prior_agent'] = goal_prior.agent
    actor_prior = getattr(pilot, 'frozen_actor_prior', None)
    if actor_prior is not None:
        modules['frozen_actor_prior'] = actor_prior
    for name, module in getattr(pilot, 'body_anchor', {}).items():
        modules['body_anchor_' + name] = module
    return {name: {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
            for name, module in modules.items()}


def verify_frozen_network_snapshot(pilot, before):
    import torch
    after = frozen_network_snapshot(pilot)
    if before.keys() != after.keys() or any(before[name].keys() != after[name].keys() or any(
            not torch.equal(value, after[name][key]) for key, value in state.items())
            for name, state in before.items()):
        raise ValueError('Frozen backend evaluation changed model or normalizer tensors')
    return dict(all_model_and_normalizer_tensors_bit_identical=True,
        compared_tensor_count=sum(len(state) for state in before.values()))
