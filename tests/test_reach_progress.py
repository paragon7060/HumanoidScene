"""CPU-only approach progress/reset/isolation tests."""

import torch

from kuavo_isaaclab_scene.rl.mdp.reach_progress import ReachProgress


def test_hold_and_round_trip_cannot_repeat_reward():
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
    assert step(.10).eq(0).all()  # retreat
    assert step(.05).eq(0).all()  # revisiting the paid position
    second = step(.04)
    torch.testing.assert_close(first + second, torch.full((1, 2), torch.exp(torch.tensor(-.48)) - torch.exp(torch.tensor(-1.2))))


def test_partial_reset_wait_and_box_switch_rebase_without_windfall():
    p = ReachProgress(2, "cpu")
    active = torch.ones(2, dtype=torch.bool)
    target = torch.zeros(2, dtype=torch.long)
    p.advance(torch.full((2, 2), .2), target, active, active)
    p.advance(torch.full((2, 2), .1), target, active, active)
    other_best = p.best[1].clone()
    p.reset(torch.tensor([0]))
    p.advance(torch.full((2, 2), .01), target, active, torch.tensor([True, False]))
    assert p.delta[0].eq(0).all()
    torch.testing.assert_close(p.best[1], other_best)
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
