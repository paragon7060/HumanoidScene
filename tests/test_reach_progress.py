"""CPU-only approach progress/reset/isolation tests."""

import torch

from kuavo_isaaclab_scene.rl.mdp.reach_progress import ReachProgress


def test_hold_and_round_trip_have_symmetric_signed_reward():
    progress = ReachProgress(1, "cpu")
    target = torch.zeros(1, dtype=torch.long)
    active = torch.ones(1, dtype=torch.bool)
    def step(distance):
        progress.advance(torch.full((1, 2), distance), target, active, active)
        return progress.delta.clone()
    assert step(.1).eq(0).all()  # baseline
    first = step(.05)
    assert first.gt(0).all()
    assert step(.05).eq(0).all()  # holding
    torch.testing.assert_close(step(.10), -first)  # retreat
    torch.testing.assert_close(step(.05), first)  # re-approach earns again
    second = step(.04)
    torch.testing.assert_close(first + second, torch.full((1, 2), torch.exp(torch.tensor(-.48)) - torch.exp(torch.tensor(-1.2))))


def test_partial_reset_wait_and_box_switch_rebase_without_windfall():
    p = ReachProgress(2, "cpu")
    active = torch.ones(2, dtype=torch.bool)
    target = torch.zeros(2, dtype=torch.long)
    p.advance(torch.full((2, 2), .2), target, active, active)
    p.advance(torch.full((2, 2), .1), target, active, active)
    other_previous = p.previous[1].clone()
    p.reset(torch.tensor([0]))
    p.advance(torch.full((2, 2), .01), target, active, torch.tensor([True, False]))
    assert p.delta[0].eq(0).all()
    torch.testing.assert_close(p.previous[1], other_previous)
    target[:] = 1
    p.advance(torch.zeros(2, 2), target, active, active)
    assert p.delta.eq(0).all()
    p.advance(torch.full((2, 2), .1), target, ~active, active)
    p.advance(torch.zeros(2, 2), target, active, active)
    assert p.delta.eq(0).all()  # first sample after initial wait


def test_no_update_preserves_cached_reward_and_frame_rate_total():
    def total(steps):
        p = ReachProgress(1, "cpu")
        flag = torch.ones(1, dtype=torch.bool)
        box = torch.zeros(1, dtype=torch.long)
        result = 0.
        for distance in torch.linspace(.2, .01, steps + 1):
            p.advance(torch.full((1, 2), float(distance)), box, flag, flag)
            cached = p.delta.clone()
            p.advance(torch.zeros(1, 2), box, flag, ~flag)
            torch.testing.assert_close(p.delta, cached)
            result += 4 * p.delta[0, 1].item()
        return result
    assert abs(total(30) - total(60)) < 1e-6


def test_per_hand_grasp_acquisition_hold_loss_rebase_and_reapproach():
    p = ReachProgress(1, "cpu")
    active = torch.ones(1, dtype=torch.bool)
    target = torch.zeros(1, dtype=torch.long)
    held = torch.zeros(1, 2, dtype=torch.bool)

    def step(distance):
        p.advance(torch.full((1, 2), distance), target, active, active, held=held)

    step(.10)
    step(.05)
    assert p.delta.gt(0).all()
    held[0, 1] = True
    step(.03)  # acquiring right grasp must not generate reaching reward
    assert p.delta[0, 0] > 0 and p.delta[0, 1] == 0
    step(.15)  # lifting/hand motion while held must not penalize reaching
    assert p.delta[0, 0] < 0 and p.delta[0, 1] == 0
    assert p.initialized.tolist() == [[True, False]]
    held[0, 1] = False
    step(.10)  # first lost-grasp step rebases, no jump penalty/bonus
    assert p.delta[0, 1] == 0
    step(.05)
    assert p.delta.gt(0).all()


def test_nonfinite_hand_does_not_poison_other_hand_or_next_valid_sample():
    p = ReachProgress(1, "cpu")
    active = torch.ones(1, dtype=torch.bool)
    target = torch.zeros(1, dtype=torch.long)
    p.advance(torch.full((1, 2), .1), target, active, active)
    p.advance(torch.tensor([[float("nan"), .05]]), target, active, active)
    assert p.delta[0, 0] == 0 and p.delta[0, 1] > 0
    p.advance(torch.full((1, 2), .03), target, active, active)
    assert p.delta[0, 0] == 0 and p.delta[0, 1] > 0
