"""CPU-only checks for non-renewable grasp and signed alignment/lift shaping."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.flap_progress import FlapProgress


def step(p, *, a=.5, d=.05, held=False, height=0., target=0, enabled=True,
         candidate=0, update=True, prime=False):
    n = len(p.height)
    p.advance(torch.full((n, 2), a), torch.full((n, 2), d),
              torch.full((n, 2), candidate, dtype=torch.long),
              torch.full((n, 2), held), torch.full((n,), held),
              torch.full((n,), height), torch.full((n,), target, dtype=torch.long),
              torch.full((n,), enabled), torch.full((n,), update), prime=prime)


def test_alignment_signed_static_approach_and_candidate_boundaries():
    p = FlapProgress(1, "cpu")
    step(p)
    assert p.orientation_delta.eq(0).all()
    step(p, a=1.)
    assert p.orientation_delta.eq(.75).all()
    step(p, a=1., d=.02)  # pure approach earns no alignment reward
    assert p.orientation_delta.eq(0).all()
    step(p, a=.5, d=.02)
    assert p.orientation_delta.eq(-.75).all()
    step(p, a=1., candidate=1)  # new nearest flap must not generate a jump
    assert p.orientation_delta.eq(0).all()
    step(p, a=.5, candidate=1, held=True)
    assert p.orientation_delta.eq(0).all()
    step(p, a=1., candidate=1)  # lost grasp rebases
    assert p.orientation_delta.eq(0).all()


def test_grasp_awarded_once_and_no_reward_for_static_shelf_or_partial_lift():
    p = FlapProgress(1, "cpu")
    step(p)
    step(p, held=True)
    assert p.grasp_bonus.item() == 1
    assert p.lift_delta.item() == 0  # acquisition baseline
    for _ in range(3):
        step(p, held=True)
        assert p.grasp_bonus.item() == p.lift_delta.item() == 0
    step(p, held=True, height=.5)
    assert p.lift_delta.item() == .5
    step(p, held=True, height=.5)
    assert p.lift_delta.item() == 0
    step(p, held=True, height=.2)
    assert p.lift_delta.item() == pytest.approx(-.3)
    step(p, height=0.)  # released; no lift shaping
    step(p, held=True, height=.7)
    assert p.grasp_bonus.item() == p.lift_delta.item() == 0
    step(p, held=True, height=1.2)
    assert p.lift_delta.item() == pytest.approx(.3)
    step(p, held=True, height=1.4)
    assert p.lift_delta.item() == 0  # above goal cannot accrue more lift credit


def test_wait_held_reset_target_switch_partial_reset_and_cached_reads():
    p = FlapProgress(2, "cpu")
    step(p, enabled=False)
    step(p)  # waiting completed, baseline
    step(p, a=1., held=True)
    assert p.grasp_bonus.eq(1).all()
    before = p.observation().clone()
    step(p, update=False)
    torch.testing.assert_close(before, p.observation())
    assert p.grasp_bonus.eq(1).all()  # cached, not re-evaluated by readers
    p.reset(torch.tensor([0]))
    step(p)  # env 0 rebases; env 1 retains consumed grasp credit
    step(p, held=True)
    assert p.grasp_bonus.tolist() == [1., 0.]
    step(p, target=1, held=True, height=.8)
    assert p.grasp_bonus.eq(0).all() and p.lift_delta.eq(0).all()
    p.reset(torch.arange(2))
    step(p, held=True, prime=True)  # already held initial state cannot earn bonus
    step(p)
    step(p, held=True)
    assert p.grasp_bonus.eq(0).all()
    p.reset(torch.arange(2))
    step(p, enabled=False, held=True)
    step(p, held=True)  # first valid held sample is also not an acquisition
    assert p.grasp_bonus.eq(0).all()


def test_invalid_input_has_finite_cached_outputs_and_rebases():
    p = FlapProgress(1, "cpu")
    step(p)
    step(p, a=float("nan"), d=float("nan"), height=float("nan"))
    assert torch.isfinite(p.observation()).all()
    assert torch.isfinite(p.orientation_distance).all()
    assert p.orientation_delta.eq(0).all() and p.lift_delta.eq(0).all()
    step(p, a=1.)
    assert p.orientation_delta.eq(0).all()
