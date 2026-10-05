"""Balance observed successful jaw labels without inventing closing commands."""
import torch
from torch.nn import functional as F


VARIANT = 'region-hand-class'


def success_jaw_balance_config(variant):
    if variant in (None, 'off'):
        return None
    if variant != VARIANT:
        raise ValueError('Unknown successful TRAIN jaw balance variant')
    return dict(variant=VARIANT, format_version=1,
        scope='actual_flap_successful_TRAIN_actor_jaw_NLL_only',
        reduction='mean_present_regions_then_eligible_hands_then_present_open_close_classes',
        labels='unchanged_actual_executed_binary_jaw_commands',
        eligibility='unchanged_production_nominal_near_gate',
        absent_classes_not_invented=True, body_loss_and_Q_replay_unchanged=True)


def balanced_success_jaw_loss(logits, near, labels, regions, config):
    if config != success_jaw_balance_config(VARIANT):
        raise ValueError('Successful TRAIN jaw balance configuration differs')
    if logits.ndim != 2 or logits.shape[1] != 2 or not len(logits) \
            or near.shape != logits.shape or near.dtype != torch.bool \
            or labels.shape != logits.shape or not bool((labels.abs() == 1).all()) \
            or regions.shape != (len(logits),) or regions.dtype != torch.long \
            or bool(((regions < 0) | (regions > 3)).any()) \
            or not bool(torch.isfinite(logits).all()):
        raise ValueError('Expected finite two-jaw logits, eligibility, binary labels and rack regions')
    terms = F.binary_cross_entropy_with_logits(logits, (labels+1)/2, reduction='none')
    region_losses = []
    groups = 0
    for region in range(4):
        hand_losses = []
        for hand in range(2):
            eligible = (regions == region) & near[:, hand]
            classes = []
            for label in (-1., 1.):
                selected = eligible & (labels[:, hand] == label)
                if bool(selected.any()):
                    classes.append(terms[selected, hand].mean())
                    groups += 1
            if classes:
                hand_losses.append(torch.stack(classes).mean())
        if hand_losses:
            region_losses.append(torch.stack(hand_losses).mean())
    loss = torch.stack(region_losses).mean() if region_losses else logits.sum()*0
    return loss, dict(success_jaw_balanced_regions=len(region_losses),
        success_jaw_balanced_groups=groups, success_jaw_eligible_hand_rows=int(near.sum()),
        success_jaw_closed_label_rows=int((near & (labels > 0)).sum()),
        success_jaw_open_label_rows=int((near & (labels < 0)).sum()))
