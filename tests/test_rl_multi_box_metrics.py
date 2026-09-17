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
    grasp_potentials,
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
