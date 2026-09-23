"""CPU checks for the independent v2 reward ratios and composition."""

from dataclasses import replace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.rewards import (
    CommonRewardInput,
    GraspRewardInput,
    HighLevelRewardInput,
    MultiBoxRewardModel,
    MultiBoxRewardWeights,
    potential_progress,
)


def common(n=2, *, rack=False, drop=False, workspace=False):
    no = torch.zeros(n, dtype=torch.bool)
    zero = torch.zeros(n)
    return CommonRewardInput(torch.full((n,), rack, dtype=torch.bool), no,
                             torch.full((n,), drop, dtype=torch.bool), no,
                             torch.full((n,), workspace, dtype=torch.bool),
                             zero, zero, zero)


def grasp(n=2, *, drop=False, events=True):
    zero, one = torch.zeros(n), torch.ones(n)
    event = torch.full((n,), events, dtype=torch.bool)
    return GraspRewardInput(
        zero, one, zero, one, zero, one, zero, one,
        event, event, event, common(n, drop=drop),
    )


def test_default_weight_hierarchy_keeps_success_and_failures_dominant():
    weights = MultiBoxRewardWeights()
    weights.validate()
    assert weights.discount == pytest.approx(0.999)
    assert weights.grasp.success_event > sum((weights.grasp.approach_progress,
        weights.grasp.alignment_progress, weights.grasp.capture_progress,
        weights.grasp.proof_lift_progress))
    assert weights.carry.success_event > 2.5 - 1e-6
    assert weights.place.success_event > 3.0 - 1e-6
    assert weights.common.box_drop > weights.place.success_event
    assert weights.common.workspace_limit > weights.place.success_event
    assert weights.high_level.full_success_event > 2 * weights.high_level.first_placement


def test_grasp_success_is_positive_and_drop_outweighs_it():
    model = MultiBoxRewardModel()
    successful = model.grasp(grasp())
    dropped = model.grasp(grasp(drop=True))
    assert (successful.total > 5).all()
    assert (dropped.total < 0).all()
    torch.testing.assert_close(
        successful.terms["box_drop"] - dropped.terms["box_drop"],
        torch.full((2,), 8.0),
    )


def test_one_hand_pinch_gives_progress_without_declaring_success():
    model = MultiBoxRewardModel()
    value = replace(grasp(events=False), one_hand_pinch_event=torch.ones(2, dtype=torch.bool))
    result = model.grasp(value)
    assert result.terms["one_hand_pinch_event"].eq(0.5).all()
    assert result.terms["bilateral_pinch_event"].eq(0).all()
    assert result.terms["success_event"].eq(0).all()


def test_grasp_motion_costs_are_lower_than_other_skills():
    model = MultiBoxRewardModel()
    moving = replace(common(), normalized_base_motion=torch.ones(2),
                     normalized_action_rate=torch.ones(2))
    reward = model.grasp(replace(grasp(events=False), common=moving))
    torch.testing.assert_close(reward.terms["base_motion"], torch.full((2,), -0.0002))
    torch.testing.assert_close(reward.terms["action_rate"], torch.full((2,), -0.0001))
    assert model.weights.common.base_motion == 0.002
    assert model.weights.common.action_rate == 0.001


def test_workspace_hard_limit_has_a_terminal_scale_penalty():
    model = MultiBoxRewardModel()
    safe = model.grasp(replace(grasp(events=False), common=common()))
    outside = model.grasp(replace(
        grasp(events=False), common=common(workspace=True)))
    torch.testing.assert_close(
        safe.terms["workspace_limit"] - outside.terms["workspace_limit"],
        torch.full((2,), model.weights.common.workspace_limit),
    )


def test_robot_rack_collision_uses_its_dedicated_penalty():
    model = MultiBoxRewardModel()
    safe = model.grasp(replace(grasp(events=False), common=common()))
    rack = model.grasp(replace(
        grasp(events=False), common=common(rack=True)))
    torch.testing.assert_close(
        safe.terms["robot_rack_collision"] - rack.terms["robot_rack_collision"],
        torch.full((2,), model.weights.common.robot_rack_collision),
    )
    assert rack.terms["obstacle_collision"].eq(0).all()


def test_static_potential_cannot_produce_repeated_positive_reward():
    model = MultiBoxRewardModel()
    value = grasp(events=False)
    one = torch.ones(2)
    value = replace(value,
        previous_approach=one, approach=one,
        previous_alignment=one, alignment=one,
        previous_capture=one, capture=one,
        previous_proof_lift=one, proof_lift=one)
    result = model.grasp(value)
    assert (result.total < 0).all()


def test_potential_shaping_telescopes_on_discounted_closed_loop():
    gamma = 0.99
    first = potential_progress(torch.tensor([0.]), torch.tensor([1.]), gamma)
    second = potential_progress(torch.tensor([1.]), torch.tensor([0.]), gamma)
    torch.testing.assert_close(first + gamma * second, torch.zeros(1), atol=1e-7, rtol=0)


def test_high_level_rewards_count_first_placement_once_and_penalize_displacement():
    model = MultiBoxRewardModel()
    result = model.high_level(HighLevelRewardInput(
        first_placement_count=torch.tensor([1., 0.]),
        displacement_count=torch.tensor([0., 1.]),
        full_success_event=torch.tensor([True, False]),
        failure_event=torch.tensor([False, False]),
        elapsed_seconds=torch.tensor([2., 2.]),
    ))
    torch.testing.assert_close(result.total, torch.tensor([16.98, -3.02]))


def test_reward_events_must_be_boolean_pulses():
    value = grasp()
    bad = replace(value, success_event=torch.ones(2))
    with pytest.raises(TypeError, match="boolean"):
        MultiBoxRewardModel().grasp(bad)
