"""Soft actor mean recovery when tanh hides both exploration and Q gradients.

This changes only an opt-in TRAIN actor loss. It neither clips a mean nor
narrows physical goals, changes rewards, supplies labels or samples actions.
"""
from copy import deepcopy

import torch


VARIANT = 'mean3-soft'


def body_saturation_config(variant):
    if variant in (None, 'off'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown body mean saturation penalty variant')
    return dict(variant=variant,format_version=1,mean_limit=3.,weight=.001,
        scope='actual_flap_TRAIN_actor_updates_only',
        loss='mean_rows_sum_active_body_relu_abs_mean_minus_limit_squared_over_active_count',
        active_coordinates='positive_available_affine_goal_radius',
        symmetric_positive_negative=True,hard_mean_clipping=False,
        physical_goal_envelope_and_projection_unchanged=True,
        reward_success_labels_Q_targets_and_behavior_distribution_unchanged=True)


def body_saturation_penalty(mean, active, config):
    if config != body_saturation_config(VARIANT):
        raise ValueError('Body saturation penalty configuration differs')
    if mean.ndim != 2 or mean.shape[1] != 19 or not len(mean) \
            or active.shape != mean.shape or active.dtype != torch.bool \
            or active.device != mean.device or not torch.isfinite(mean).all():
        raise ValueError('Finite19-D means and matching active-coordinate mask required')
    excess=(mean.abs()-config['mean_limit']).clamp_min(0)
    per_row=(excess.square()*active).sum(-1)/active.sum(-1).clamp_min(1)
    loss=config['weight']*per_row.mean()
    return loss,dict(body_saturation_loss=loss.detach().item(),
        body_saturation_weight=config['weight'],body_saturation_limit=config['mean_limit'],
        body_saturation_active_coordinates=int(active.sum()),
        body_saturation_saturated_coordinates=int(((excess>0)&active).sum()),
        body_active_abs_mean_max=torch.where(active,mean.abs(),0.).max().detach().item())


def enable_body_saturation(state,experience,*,source_checkpoint,variant=VARIANT):
    if state.get('artifact_type')!='staged_actual_flap_bounded_actor_correction_hybrid_sac_v1' \
            or state.get('goal_contract')!=experience.get('goal_contract') \
            or state.get('body_saturation') is not None or experience.get('body_saturation') is not None:
        raise ValueError('Matching actual-flap checkpoint/replay without body regularization required')
    config=body_saturation_config(variant)
    if config is None:raise ValueError('Explicit enabled body saturation variant required')
    origin=dict(source_checkpoint=str(source_checkpoint),actor_updates_at_activation=state['actor_updates'],
        critic_updates_at_activation=state['critic_updates'],
        old_replay_rows_at_activation=len(experience['executed_goal_transitions']['reward']),
        model_Q_normalizers_and_four_optimizer_states_kept=True,old_replay_kept_without_relabeling=True,
        scope='future_real_TRAIN_actor_updates')
    extras=dict(body_saturation=config,body_saturation_origin=origin)
    return state|deepcopy(extras),experience|deepcopy(extras)
