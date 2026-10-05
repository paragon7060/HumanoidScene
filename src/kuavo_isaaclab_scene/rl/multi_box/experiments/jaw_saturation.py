"""Optional actor-only recovery signal for saturated, eligible binary jaws.

This is a soft TRAIN loss, not a clipped logit, probability floor, prescribed
open/close action, reward change or behavior sampler. The physical projector
and exact learned four-branch expectation keep their original distributions.
"""
import torch


VARIANT = 'logit4-soft'


def jaw_saturation_config(variant):
    if variant in (None, 'off'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown jaw saturation penalty variant')
    return dict(variant=VARIANT, format_version=1, logit_limit=4., weight=.0001,
        scope='actual_flap_TRAIN_actor_updates_only',
        loss='mean_rows_sum_eligible_jaws_relu_abs_logit_minus_limit_squared_over_active_count',
        eligible_jaws='unchanged_production_nominal_near_gate',
        symmetric_open_close=True, reward_or_success_labels_used=False,
        hard_logit_clipping=False, probability_floor=False,
        Q_target_and_behavior_distribution_unchanged=True)


def jaw_saturation_penalty(logits, near, config):
    if config != jaw_saturation_config(VARIANT):
        raise ValueError('Jaw saturation penalty configuration differs')
    if logits.ndim != 2 or logits.shape[1] != 2 or not len(logits) \
            or near.shape != logits.shape or near.dtype != torch.bool \
            or not torch.isfinite(logits).all():
        raise ValueError('Expected finite two-jaw logits and matching eligibility mask')
    excess = (logits.abs()-config['logit_limit']).clamp_min(0)
    per_row = (excess.square()*near).sum(-1)/near.sum(-1).clamp_min(1)
    loss = config['weight']*per_row.mean()
    return loss, dict(jaw_saturation_loss=loss.detach().item(),
        jaw_saturation_weight=config['weight'], jaw_saturation_limit=config['logit_limit'],
        jaw_saturation_active_hands=int(near.sum()),
        jaw_saturation_saturated_hands=int(((excess>0)&near).sum()),
        jaw_active_abs_logit_max=torch.where(near,logits.abs(),0.).max().detach().item())
