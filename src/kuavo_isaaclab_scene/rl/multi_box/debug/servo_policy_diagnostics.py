"""Read-only production step clipping on measured held SAC actor states."""
from collections import defaultdict

import torch


@torch.no_grad()
def measured_policy_servo_diagnostics(agent, previous, environment_ids, layouts):
    raw, _, executed = previous
    encoder = getattr(agent, 'goal_servo_critic_encoder', None)
    if encoder is None or raw.ndim != 2 or raw.shape[1] != 518 \
            or executed.shape != (len(raw), 21) or environment_ids.shape != (len(raw),) \
            or environment_ids.dtype != torch.long or len(environment_ids.unique()) != len(raw):
        raise ValueError('Servo diagnostics require matching held actor518/goals21/unique IDs and production encoder')
    groups = defaultdict(list)
    for row, index in enumerate(environment_ids.tolist()):
        if index < 0 or index >= len(layouts):
            raise ValueError('Diagnostic environment ID is outside the requested wave')
        layout = layouts[index]['layout']
        groups[layout['target_region'] + '/' + layout['target_box_type']].append(row)
    if not len(raw):
        return dict(states=0, by_region_size={}, read_only=True)
    greedy = agent.act(raw, deterministic=True)
    _, radius = agent.anchor_and_scale(raw)
    active = radius > 1e-8
    greedy_delta = encoder.unclipped_body(raw, greedy)
    executed_delta = encoder.unclipped_body(raw, executed)
    greedy_clip = greedy_delta.abs() > 1
    executed_clip = executed_delta.abs() > 1
    result = {}
    for key, indices in sorted(groups.items()):
        rows = torch.tensor(indices, device=raw.device)
        group = dict(states=len(rows))
        for name, columns in [('body', slice(0, 19)), ('arms', slice(1, 15))]:
            available = active[rows, columns]
            n = int(available.sum())
            mean_blocked = greedy_clip[rows, columns] & available
            behavior_blocked = executed_clip[rows, columns] & available
            group[name] = dict(active_coordinates=n,
                greedy_step_clip_zero_gradient_coordinates=int(mean_blocked.sum()),
                greedy_step_clip_zero_gradient_fraction=float(mean_blocked.sum()) / n if n else None,
                executed_step_clipped_coordinates=int(behavior_blocked.sum()),
                executed_step_clipped_fraction=float(behavior_blocked.sum()) / n if n else None,
                greedy_all_active_step_gradients_zero_states=int(((available.sum(-1) > 0)
                    & (mean_blocked.sum(-1) == available.sum(-1))).sum()))
        result[key] = group
    return dict(name='measured_policy_production_step_clip_v1', states=len(raw),
        by_region_size=result, actor_distribution='actual_deterministic_continuous_sampler_and_binary_jaws',
        state='same_pre_action_actor_input_as_actual_collection',
        production_decoder='actual_pending_targets_and_velocity_step_limits',
        gradient_scope='zero_from_production_step_clip_only_not_total_actor_or_Q_gradient',
        tanh_underflow_inactive_coordinates_and_Q_value_gradient_NOT_measured=True,
        no_optimization_no_random_draw_no_replay_or_policy_change=True, read_only=True)
