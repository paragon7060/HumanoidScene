"""Approach diagnostics must latch on BOTH perceived hands and retain defaults."""
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import vr_handoff_ready


def test_reference_handoff_is_unchanged_without_the_diagnostic():
    assert not vr_handoff_ready(399,514,False)
    assert vr_handoff_ready(514,514,False)
    assert vr_handoff_ready(515,514,True)


def test_early_handoff_requires_both_valid_near_hands_and_then_latches():
    assert not vr_handoff_ready(350,514,False,torch.tensor([[.01,.30]]),.12)
    assert not vr_handoff_ready(350,514,False,torch.tensor([[.01,float('nan')]]),.12)
    assert vr_handoff_ready(350,514,False,torch.tensor([[.10,.11]]),.12)
    assert vr_handoff_ready(351,514,True,torch.tensor([[.50,.60]]),.12)
    # Original reference handoff remains a fallback, without changing grasp
    # success, contact thresholds or reset timer.
    assert vr_handoff_ready(514,514,False,torch.tensor([[.50,.60]]),.12)


@pytest.mark.parametrize('distance',[-.1,.01,.26,float('nan'),float('inf')])
def test_invalid_handoff_bounds_are_rejected(distance):
    with pytest.raises(ValueError):vr_handoff_ready(0,514,False,torch.ones(1,2),distance)
