"""Optional critic deadline input measured by the actual task timeout clock."""
import torch

TASK_TIMING_GROUP = 'task_time_remaining'
CRITIC_HELD_CLOCK_INDEX = 530
CRITIC_REMAINING_INDEX = 571
VARIANT = 'task-remaining'


def critic_episode_clock_config(variant):
    if variant is None:
        return None
    if variant != VARIANT:
        raise ValueError('Unknown critic episode clock')
    return dict(name='v2_added_settled_task_remaining_fraction_v1', observation_group=TASK_TIMING_GROUP,
        critic_feature_index=CRITIC_REMAINING_INDEX, critic_dimension=578,
        original_held_phase_clock_preserved=True, last_six_held_context_preserved=True,
        actor_features_and_controller_unchanged=True,
        source='same_reset_settling_ready_steps_and_max_episode_length_as_task_time_out',
        value='clamp(1-ready_steps/max_episode_length,0,1)',
        before_autoreset_terminal_capture_required=True, old_Q_or_reward_banks_import_allowed=False)


def resolve_critic_episode_clock(saved, requested=None):
    config = critic_episode_clock_config(requested)
    stored = saved.get('critic_episode_clock') if saved is not None else None
    if stored is not None and stored != critic_episode_clock_config(VARIANT):
        raise ValueError('Saved critic episode clock differs')
    if saved is not None and requested is not None and stored != config:
        raise ValueError('Prepare fresh Q and empty replay before changing critic time coordinates')
    return stored if stored is not None else config


def task_remaining_fraction(ready_steps, max_episode_length):
    if ready_steps.ndim != 1 or not torch.isfinite(ready_steps).all() or (ready_steps < 0).any() \
            or type(max_episode_length) is not int or max_episode_length < 1:
        raise ValueError('Task timing needs finite nonnegative ready steps and a positive horizon')
    return (1 - ready_steps.to(torch.float32) / max_episode_length).clamp(0, 1)[:, None]


def add_critic_task_time(critic, remaining):
    if critic.ndim != 2 or critic.shape[1] != 577 \
            or remaining is None or remaining.shape != (len(critic), 1) \
            or remaining.device != critic.device or not torch.isfinite(remaining).all() \
            or ((remaining < 0) | (remaining > 1)).any():
        raise ValueError('Measured pre-autoreset task time is required for this critic')
    return torch.cat((critic[:, :CRITIC_REMAINING_INDEX], remaining.to(critic),
        critic[:, CRITIC_REMAINING_INDEX:]), -1)
