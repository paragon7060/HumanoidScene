"""Reward regressions for a low-friction box resting on the rack."""

import math
import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.grasp_stability import grasp_stability_scores
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


def scores(*, x=0., vx=0., vz=0., angular=0., angle=0., grasp=False, height=0.):
    return grasp_stability_scores(torch.tensor([[x, 0., 0.]]),
        torch.tensor([[math.cos(angle/2), 0., 0., math.sin(angle/2)]]),
        torch.tensor([[1., 0., 0., 0.]]),
        torch.tensor([[vx, 0., vz, 0., 0., angular]]),
        torch.tensor([grasp]), torch.tensor([height]), task_spec("pick"))


def test_idle_box_has_no_disturbance_and_no_unearned_grasp_bonus():
    penalty, bonus = scores()
    assert penalty == 0 and bonus == 0
    penalty, bonus = scores(grasp=True)
    assert penalty == 0 and bonus == 1


@pytest.mark.parametrize("kwargs", [dict(x=.02), dict(vx=.05), dict(vz=.05),
                                    dict(angular=.5), dict(angle=math.radians(10))])
def test_sliding_stopped_displacement_and_shaking_are_penalized(kwargs):
    penalty, bonus = scores(**kwargs)
    assert penalty > 0 and bonus == 0


def test_airborne_motion_without_grasp_is_penalized_but_not_old_rest_pose():
    penalty, bonus = scores(x=.02, vz=.2, height=.10)
    assert penalty > 0 and bonus == 0  # tossing does not count
    penalty, bonus = scores(x=.20, height=.10, angle=math.radians(20))
    assert penalty == 0 and bonus == 0  # no stale shelf-position cost in air
    penalty, bonus = scores(x=.02, grasp=True, height=.005)
    assert penalty > 0 and bonus < 1  # still on the rack
    penalty, bonus = scores(x=.02, grasp=True, height=.02)
    assert penalty == 0 and bonus == 1


def test_rest_pose_penalty_fades_continuously_with_height():
    low, _ = scores(x=.02, grasp=True, height=0.)
    middle, _ = scores(x=.02, grasp=True, height=.005)
    lifted, _ = scores(x=.02, grasp=True, height=.01)
    assert middle == pytest.approx(low.item() * .5)
    assert lifted == 0


def test_genuine_vertical_lift_is_allowed_before_full_clearance():
    penalty, bonus = scores(vz=.2, grasp=True, height=.005)
    assert penalty == 0 and bonus == 1


def test_shaking_reduces_the_grasp_bonus_even_after_liftoff():
    _, stable = scores(grasp=True, height=.02)
    penalty, shaking = scores(grasp=True, height=.02, vx=.1, angular=.5)
    assert penalty == 0 and 0 < shaking < stable


def test_small_contact_adjustments_are_free_and_total_penalty_is_capped():
    penalty, _ = scores(x=.004, vx=.015, vz=.015, angular=.09, angle=math.radians(4))
    assert penalty == 0
    penalty, _ = scores(x=10., vx=10., vz=10., angular=10., angle=math.radians(90))
    assert penalty == 1
    free, _ = scores(x=.02)
    held, _ = scores(x=.02, grasp=True)
    assert held == pytest.approx(free.item() * .25)
