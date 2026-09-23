"""CPU checks for VR-calibrated raw metric normalization and sidecar logs."""

import json
import math

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.debug import (
    PoseShadowRewardEvaluator,
    ShadowRewardLogger,
    format_shadow_reward,
)
from kuavo_isaaclab_scene.rl.multi_box.metrics import (
    CarryRawMetrics,
    GraspRawMetrics,
    PlaceRawMetrics,
    carry_potentials,
    grasp_gated_lift_inputs,
    grasp_potentials,
    grasp_reward_potentials,
    opposing_flap_reach_assignment,
    place_potentials,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards import RewardBreakdown


def test_grasp_potentials_are_monotonic_and_proof_lift_is_bounded():
    raw = GraspRawMetrics(
        matched_flap_distance_m=torch.tensor([0.30, 0.05]),
        jaw_alignment_error_rad=torch.tensor([1.0, 0.1]),
        capture_error_m=torch.tensor([0.04, 0.002]),
        proof_lift_m=torch.tensor([-0.01, 0.02]),
    )
    values = grasp_potentials(raw)
    assert values["approach"][1] > values["approach"][0]
    assert values["alignment"][1] > values["alignment"][0]
    assert values["capture"][1] > values["capture"][0]
    assert values["proof_lift"].tolist() == [0.0, 1.0]


def test_grasp_reward_pays_each_hand_but_prefers_opposing_flaps():
    distances = torch.tensor([
        [[0.30, 0.30], [0.30, 0.30]],
        [[0.02, 0.30], [0.30, 0.30]],
        [[0.02, 0.30], [0.03, 0.30]],
        [[0.02, 0.30], [0.30, 0.03]],
    ])
    matched = torch.tensor([[0.30, 0.30], [0.02, 0.30],
                            [0.02, 0.30], [0.02, 0.03]])
    values = grasp_reward_potentials(
        distances, matched, torch.ones(4, 2), torch.zeros(4, 2),
        torch.full((4, 2), 0.02), torch.full((4, 2), 0.01), torch.zeros(4))
    assert values["approach"][1] > values["approach"][0]
    assert values["approach"][3] > values["approach"][2]


def test_opposing_flap_reach_rewards_weaker_hand_without_same_flap_shortcut():
    distances = torch.tensor([
        [[0.0, 10.0], [10.0, 10.0]],  # Only the left hand reaches.
        [[0.0, 10.0], [0.0, 10.0]],   # Both hands reach the same flap.
        [[0.0, 10.0], [10.0, 0.0]],   # Both hands reach distinct flaps.
        [[10.0, 0.0], [0.0, 10.0]],   # Swapped assignment also succeeds.
    ])
    reach, assignment = opposing_flap_reach_assignment(distances, 1.0 / 12.0)
    torch.testing.assert_close(reach, torch.tensor([0.25, 0.25, 1.0, 1.0]),
                               rtol=0, atol=1e-6)
    assert assignment.tolist() == [[0, 1], [0, 1], [0, 1], [1, 0]]
    matched = distances[torch.arange(4)[:, None], torch.arange(2)[None], assignment]
    potentials = grasp_reward_potentials(
        distances, matched, torch.ones(4, 2), torch.zeros(4, 2),
        torch.full((4, 2), 0.02), torch.full((4, 2), 0.01), torch.zeros(4))
    torch.testing.assert_close(potentials["approach"], reach)


def test_grasp_reward_gap_preparation_and_premature_close():
    distance = torch.tensor([[[0.01, 0.30], [0.30, 0.01]]]).expand(4, -1, -1)
    matched = torch.tensor([[0.01, 0.01], [0.01, 0.01],
                            [0.30, 0.30], [0.01, 0.01]])
    gap = torch.tensor([[0.02, 0.02], [0.08, 0.08],
                        [0.005, 0.005], [0.005, 0.005]])
    values = grasp_reward_potentials(
        distance, matched, torch.ones(4, 2), torch.zeros(4, 2),
        gap, torch.full((4, 2), 0.01), torch.zeros(4))
    assert values["jaw_gap"][0] > values["jaw_gap"][1]
    assert values["premature_close"][2] == 0
    assert values["premature_close"][3] > values["premature_close"][0]


def test_box_bounce_cannot_earn_lift_without_continuous_opposing_pinch():
    gamma = 0.999
    false = torch.tensor([False])
    true = torch.tensor([True])
    prior, current = grasp_gated_lift_inputs(
        torch.tensor([1.0]), torch.tensor([0.0]), false, false, gamma)
    torch.testing.assert_close(gamma * current - prior, torch.zeros(1))
    prior, current = grasp_gated_lift_inputs(
        torch.tensor([0.5]), torch.tensor([0.0]), true, false, gamma)
    torch.testing.assert_close(gamma * current - prior, torch.zeros(1))
    prior, current = grasp_gated_lift_inputs(
        torch.tensor([1.0]), current, true, true, gamma)
    assert (gamma * current - prior > 0).all()
    prior, current = grasp_gated_lift_inputs(current, current, false, true, gamma)
    torch.testing.assert_close(gamma * current - prior, torch.zeros(1))


def test_carry_height_scores_interval_and_free_space_penalizes_overlap():
    raw = CarryRawMetrics(
        extraction_remaining_m=torch.tensor([0.20, 0.0, 0.0]),
        footprint_distance_to_belt_m=torch.tensor([0.5, 0.0, 0.0]),
        free_space_clearance_m=torch.tensor([-0.02, 0.05, 0.20]),
        box_bottom_height_m=torch.tensor([0.0, 0.10, 0.20]),
    )
    values = carry_potentials(raw)
    assert values["extraction"][1] == 1
    assert values["belt"][1] == 1
    assert values["free_space"].tolist() == [0.0, 0.5, 1.0]
    assert values["pre_place_height"][1] == 1
    torch.testing.assert_close(values["pre_place_height"][[0, 2]],
                               torch.full((2,), math.exp(-1)), rtol=1e-5, atol=1e-6)


def test_place_descent_is_gated_by_footprint_alignment_and_free_space():
    raw = PlaceRawMetrics(
        footprint_outside_m=torch.tensor([0.0, 0.20]),
        long_axis_error_rad=torch.tensor([0.0, math.pi / 2]),
        free_space_clearance_m=torch.tensor([0.10, -0.01]),
        box_bottom_height_m=torch.tensor([0.01, 0.01]),
        linear_speed_mps=torch.tensor([0.0, 0.5]),
        angular_speed_radps=torch.tensor([0.0, 2.0]),
    )
    values = place_potentials(raw)
    assert values["descent"][0] > 0.8
    assert values["descent"][1] == 0
    assert values["stability"][0] == 1
    assert values["stability"][1] < 1e-6


def test_shadow_logger_keeps_dataset_independent_jsonl_and_term_summary(tmp_path):
    path = tmp_path / "shadow.jsonl"
    breakdown = RewardBreakdown(
        {"progress": torch.tensor([0.25]), "base_motion": torch.tensor([-0.01])},
        torch.tensor([0.24]),
    )
    with ShadowRewardLogger(path) as logger:
        logger.record(step=7, sim_time_s=0.2, phase="grasp",
                      raw={"distance_m": 0.12}, potentials={"approach": 0.45},
                      breakdown=breakdown, events={"success": False})
        text = format_shadow_reward("grasp", {"distance_m": 0.12}, breakdown, logger.stats)
        assert "step_reward=+0.240" in text and "distance_m=0.120" in text
        assert logger.stats.total_return == pytest.approx(0.24)
    rows = path.read_text().splitlines()
    assert len(rows) == 1
    row = json.loads(rows[0])
    assert row["schema"] == "multi_box_v2_shadow_reward_v1"
    assert row["weighted_terms"] == pytest.approx({"progress": 0.25, "base_motion": -0.01})
    assert row["raw"] == {"distance_m": 0.12}


def test_shadow_logger_records_all_skill_breakdowns_in_one_v2_row(tmp_path):
    path = tmp_path / "all_phases.jsonl"
    breakdowns = {
        phase: RewardBreakdown(
            {f"{phase}_term": torch.tensor([float(index)])},
            torch.tensor([float(index)]),
        )
        for index, phase in enumerate(("grasp", "carry", "place"), start=1)
    }
    with ShadowRewardLogger(path) as logger:
        logger.record(
            step=1, sim_time_s=1 / 30, phase="grasp",
            raw={"distance_m": 0.1}, potentials={"approach": 0.2},
            breakdown=breakdowns["grasp"], events={"success": False},
            breakdowns_by_phase=breakdowns,
        )
    row = json.loads(path.read_text())
    assert row["schema"] == "multi_box_v2_shadow_reward_v2"
    assert row["step_reward_by_phase"] == {
        "grasp": 1.0, "carry": 2.0, "place": 3.0}
    assert row["weighted_terms_by_phase"]["place"] == {"place_term": 3.0}


def test_pose_shadow_evaluator_uses_selected_phase_and_leaves_events_disabled():
    evaluator = PoseShadowRewardEvaluator("grasp")
    first = evaluator.evaluate({
        "approach": torch.tensor([0.2]),
        "alignment": torch.tensor([0.3]),
        "capture": torch.tensor([0.4]),
        "proof_lift": torch.tensor([0.0]),
    })
    assert first.terms["bilateral_pinch_event"].item() == 0
    assert first.terms["success_event"].item() == 0
    assert first.terms["robot_rack_collision"].item() == 0

    improved = evaluator.evaluate({
        "approach": torch.tensor([0.4]),
        "alignment": torch.tensor([0.5]),
        "capture": torch.tensor([0.6]),
        "proof_lift": torch.tensor([1.0]),
    })
    assert improved.terms["approach_progress"].item() > 0
    assert improved.terms["proof_lift_progress"].item() > 0

    evaluator.set_phase("place")
    place = evaluator.evaluate({
        "footprint": torch.tensor([1.0]),
        "alignment": torch.tensor([1.0]),
        "free_space": torch.tensor([1.0]),
        "descent": torch.tensor([1.0]),
        "stability": torch.tensor([1.0]),
    })
    assert "footprint_progress" in place.terms
    assert place.terms["support_event"].item() == 0


def test_pose_shadow_evaluator_calibrates_all_phases_and_real_event_pulses():
    evaluator = PoseShadowRewardEvaluator("grasp")
    potentials = {
        "grasp": {
            "approach": torch.tensor([0.2]),
            "alignment": torch.tensor([0.3]),
            "capture": torch.tensor([0.4]),
            "proof_lift": torch.tensor([0.0]),
        },
        "carry": {
            "extraction": torch.tensor([0.2]),
            "belt": torch.tensor([0.3]),
            "free_space": torch.tensor([0.4]),
            "pre_place_height": torch.tensor([0.5]),
        },
        "place": {
            "footprint": torch.tensor([0.2]),
            "alignment": torch.tensor([0.3]),
            "free_space": torch.tensor([0.4]),
            "descent": torch.tensor([0.5]),
            "stability": torch.tensor([0.6]),
        },
    }
    event = torch.tensor([True])
    results = evaluator.evaluate_all(potentials, events_by_phase={
        "grasp": {"bilateral_pinch_event": event, "success_event": event},
        "carry": {"success_event": event},
        "place": {
            "support_event": event,
            "correct_release_event": event,
            "success_event": event,
        },
    })
    assert results["grasp"].terms["success_event"].item() == 5.0
    assert results["carry"].terms["success_event"].item() == 4.0
    assert results["place"].terms["success_event"].item() == 5.0
    # Initialization uses gamma*Phi as the previous value, matching the live
    # training manager and avoiding a synthetic first-step dense reward.
    assert results["grasp"].terms["approach_progress"].item() == pytest.approx(0.0)
