"""Sparse physical closing labels must survive long successful approach paths."""
import pytest
import torch
from torch.nn import functional as F

from kuavo_isaaclab_scene.rl.multi_box.experiments.success_jaw_balance import (
    balanced_success_jaw_loss, success_jaw_balance_config,
)


def test_longer_open_approach_does_not_overwhelm_observed_close_signal():
    logits = torch.zeros(100, 2, requires_grad=True)
    labels = torch.full_like(logits, -1); labels[-1] = 1
    near = torch.ones_like(logits, dtype=torch.bool)
    regions = torch.zeros(100, dtype=torch.long)
    loss, metrics = balanced_success_jaw_loss(logits, near, labels, regions,
        success_jaw_balance_config('region-hand-class'))
    balanced = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    ordinary = torch.autograd.grad(F.binary_cross_entropy_with_logits(logits, (labels+1)/2), logits)[0]
    assert ordinary.sum() > .4  # Shared zero-logit bias would be driven open.
    assert balanced.sum().abs() < 1e-6
    assert balanced[-1].sum() < -.2 and balanced[:-1].sum() > .2
    assert metrics['success_jaw_closed_label_rows'] == 2
    assert metrics['success_jaw_balanced_groups'] == 4


def test_replication_and_region_frequency_do_not_change_balanced_loss():
    config = success_jaw_balance_config('region-hand-class')
    logits = torch.tensor([[-2., 1.], [3., -1.], [.5, 2.], [-.5, -2.]])
    labels = torch.tensor([[-1., 1.], [1., -1.], [-1., 1.], [1., -1.]])
    near = torch.ones_like(logits, dtype=torch.bool)
    regions = torch.tensor([0, 0, 3, 3])
    first = balanced_success_jaw_loss(logits, near, labels, regions, config)[0]
    ids = torch.tensor([0]*100+[1, 2, 3])
    repeated = balanced_success_jaw_loss(logits[ids], near[ids], labels[ids], regions[ids], config)[0]
    torch.testing.assert_close(first, repeated)


def test_missing_closed_class_is_not_invented_and_far_hands_have_no_gradient():
    logits = torch.tensor([[-5., 4.], [-3., 8.]], requires_grad=True)
    labels = torch.tensor([[-1., 1.], [-1., 1.]])
    near = torch.tensor([[True, False], [True, False]])
    config = success_jaw_balance_config('region-hand-class')
    loss, stats = balanced_success_jaw_loss(logits, near, labels, torch.zeros(2, dtype=torch.long), config)
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    assert (gradient[:,0] > 0).all() and gradient[:,1].eq(0).all()
    assert stats['success_jaw_closed_label_rows'] == 0
    zero, _ = balanced_success_jaw_loss(logits, torch.zeros_like(near), labels, torch.zeros(2, dtype=torch.long), config)
    assert zero == 0 and torch.autograd.grad(zero, logits)[0].eq(0).all()


def test_unknown_contract_or_invented_nonbinary_labels_fail():
    assert success_jaw_balance_config(None) is success_jaw_balance_config('off') is None
    with pytest.raises(ValueError, match='Unknown'):
        success_jaw_balance_config('unknown')
    config = success_jaw_balance_config('region-hand-class')
    with pytest.raises(ValueError, match='binary'):
        balanced_success_jaw_loss(torch.zeros(1,2), torch.ones(1,2,dtype=torch.bool),
            torch.zeros(1,2), torch.zeros(1,dtype=torch.long), config)
