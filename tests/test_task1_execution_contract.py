from data_collection.task1.execution_contract import (
    active_hand_indices,
    box_extraction_metrics,
    closed_motor_targets,
    evenly_spaced_capture_indices,
    resolved_kinematic_capture_frame_count,
    motor_obstruction_gates,
    paired_box_acceptance,
    paired_retention_metrics,
    physical_acceptance,
    retention_reference,
    executor_target_keys,
)


def test_left_only_closure_keeps_right_hand_open():
    names = ["l_f_bar_1_joint", "l_b_bar_1_joint", "r_f_bar_1_joint", "r_b_bar_1_joint"]
    opened = [-0.2, 0.2, -0.3, 0.3]

    assert closed_motor_targets(opened, names, "left") == [0.0, 0.0, -0.3, 0.3]


def test_right_only_closure_keeps_left_hand_open():
    names = ["l_f_bar_1_joint", "l_b_bar_1_joint", "r_f_bar_1_joint", "r_b_bar_1_joint"]
    opened = [-0.2, 0.2, -0.3, 0.3]

    assert closed_motor_targets(opened, names, "right") == [-0.2, 0.2, 0.0, 0.0]


def test_both_mode_preserves_existing_bimanual_contract():
    names = ["l_f_bar_1_joint", "l_b_bar_1_joint", "r_f_bar_1_joint", "r_b_bar_1_joint"]

    assert closed_motor_targets([-0.2, 0.2, -0.3, 0.3], names, "both") == [0.0] * 4
    assert active_hand_indices("both") == (0, 1)
    assert retention_reference([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]], "both") == [2.0, 3.0, 4.0]


def test_single_arm_retention_uses_only_the_active_tcp():
    eef_positions = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]]

    assert active_hand_indices("left") == (0,)
    assert active_hand_indices("right") == (1,)
    assert retention_reference(eef_positions, "left") == [1.0, 2.0, 3.0]
    assert retention_reference(eef_positions, "right") == [3.0, 4.0, 5.0]


def test_single_arm_report_does_not_claim_bilateral_obstruction():
    assert motor_obstruction_gates([0.006, 0.5], "left", 0.005) == (True, None)
    assert motor_obstruction_gates([0.006, 0.004], "both", 0.005) == (False, False)


def test_box_extraction_metrics_use_oriented_front_edge_for_single_arm_pull():
    closed_pose = [
        0.6351961493492126,
        0.0017780144698917866,
        1.04098641872406,
        0.7068216800689697,
        -0.031525492668151855,
        -0.03156113624572754,
        0.705983579158783,
    ]
    final_pose = [
        0.6280081272125244,
        0.0546792671084404,
        1.1528406143188477,
        0.575096607208252,
        -0.23920804262161255,
        -0.41389575600624084,
        0.6638776659965515,
    ]

    metrics = box_extraction_metrics(
        closed_pose,
        final_pose,
        box_size_m=(0.32, 0.22, 0.185),
        rack_front_x_b_m=0.415,
        front_progress_min_m=0.05,
        front_inside_max_m=0.05,
        final_hold_box_motion_m=0.000052477262215688825,
        final_hold_box_motion_max_m=0.01,
    )

    assert abs(metrics["box_robotward_front_progress_m"] - 0.06067375) < 1e-7
    assert abs(metrics["final_box_front_inside_rack_m"] - 0.04152671) < 1e-7
    assert metrics["partial_extraction_success"] is True
    assert metrics["full_extraction_success"] is False


def test_partial_extraction_requires_final_settling():
    metrics = box_extraction_metrics(
        [0.60, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        [0.54, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        box_size_m=(0.10, 0.10, 0.10),
        rack_front_x_b_m=0.50,
        front_progress_min_m=0.05,
        front_inside_max_m=0.05,
        final_hold_box_motion_m=0.02,
        final_hold_box_motion_max_m=0.01,
    )

    assert metrics["partial_extraction_success"] is False


def test_single_arm_accepts_partial_extraction_without_rigid_retention():
    assert physical_acceptance(
        "left",
        approach_contact_free=True,
        partial_extraction_success=True,
        stable_retention=False,
    ) == (True, "partial_extraction")


def test_bimanual_mode_keeps_strict_retention_gate():
    assert physical_acceptance(
        "both",
        approach_contact_free=True,
        partial_extraction_success=True,
        stable_retention=False,
    ) == (False, "stable_bimanual_retention")


def test_evenly_spaced_capture_indices_preserve_endpoints_and_frame_count():
    indices = evenly_spaced_capture_indices(waypoint_count=176, frame_count=73)

    assert len(indices) == 73
    assert indices[0] == 0
    assert indices[-1] == 175
    assert tuple(sorted(set(indices))) == indices


def test_evenly_spaced_capture_indices_reject_more_frames_than_waypoints():
    import pytest

    with pytest.raises(ValueError, match="cannot exceed waypoint_count"):
        evenly_spaced_capture_indices(waypoint_count=4, frame_count=5)


def test_kinematic_capture_defaults_to_normal_speed_without_skipping_short_paths():
    assert resolved_kinematic_capture_frame_count(176, None) == 73
    assert resolved_kinematic_capture_frame_count(40, None) == 40
    assert resolved_kinematic_capture_frame_count(176, 90) == 90


def test_pair_executor_keeps_exactly_two_targets_without_parking():
    assert executor_target_keys(["medium_box_0", "medium_box_1"], False) == (
        "medium_box_0",
        "medium_box_1",
    )


def test_pair_executor_rejects_clearing_its_shared_shelf():
    import pytest

    with pytest.raises(ValueError, match="must remain in the scene"):
        executor_target_keys(["medium_box_0", "medium_box_1"], True)


def test_paired_retention_detects_one_box_slipping_from_left_hand():
    closed = {"a": [0.60, -0.05, 1.0], "b": [0.60, 0.05, 1.0]}
    samples = [
        {
            "phase": "retreat",
            "box_body_positions_b_m": closed,
            "eef_positions_b_m": [[0.60, 0.0, 1.0], [0.0, 0.0, 0.0]],
        },
        {
            "phase": "retreat",
            "box_body_positions_b_m": {
                "a": [0.53, -0.05, 1.0],
                "b": [0.59, 0.05, 1.0],
            },
            "eef_positions_b_m": [[0.53, 0.0, 1.0], [0.0, 0.0, 0.0]],
        },
        {
            "phase": "final_hold",
            "box_body_positions_b_m": {
                "a": [0.53, -0.05, 1.0],
                "b": [0.59, 0.05, 1.0],
            },
            "eef_positions_b_m": [[0.53, 0.0, 1.0], [0.0, 0.0, 0.0]],
        },
        {
            "phase": "final_hold",
            "box_body_positions_b_m": {
                "a": [0.53, -0.05, 1.0],
                "b": [0.59, 0.05, 1.0],
            },
            "eef_positions_b_m": [[0.53, 0.0, 1.0], [0.0, 0.0, 0.0]],
        },
    ]

    metrics = paired_retention_metrics(samples, ("a", "b"), "left", closed)

    assert metrics["box_robotward_progress_m"] == {"a": 0.07, "b": 0.01}
    assert metrics["pair_separation_drift_max_m"] > 0.01
    passed, _ = paired_box_acceptance(
        metrics,
        approach_box_motion_m={"a": 0.0, "b": 0.0},
        approach_box_motion_max_m=0.001,
        progress_min_m=0.05,
        pair_separation_drift_max_m=0.01,
        hand_pair_center_drift_max_m=0.05,
        final_hold_box_motion_max_m=0.01,
        active_motor_obstruction=True,
        tracking_passed=True,
    )
    assert passed is False


def test_paired_retention_accepts_two_boxes_moving_together():
    closed = {"a": [0.60, -0.05, 1.0], "b": [0.60, 0.05, 1.0]}
    moved = {"a": [0.53, -0.05, 1.0], "b": [0.53, 0.05, 1.0]}
    samples = [
        {"phase": "retreat", "box_body_positions_b_m": closed, "eef_positions_b_m": [[0.60, 0.0, 1.0], [0, 0, 0]]},
        {"phase": "retreat", "box_body_positions_b_m": moved, "eef_positions_b_m": [[0.53, 0.0, 1.0], [0, 0, 0]]},
        {"phase": "final_hold", "box_body_positions_b_m": moved, "eef_positions_b_m": [[0.53, 0.0, 1.0], [0, 0, 0]]},
        {"phase": "final_hold", "box_body_positions_b_m": moved, "eef_positions_b_m": [[0.53, 0.0, 1.0], [0, 0, 0]]},
    ]

    metrics = paired_retention_metrics(samples, ("a", "b"), "left", closed)
    passed, mode = paired_box_acceptance(
        metrics,
        approach_box_motion_m={"a": 0.0, "b": 0.0},
        approach_box_motion_max_m=0.001,
        progress_min_m=0.05,
        pair_separation_drift_max_m=0.01,
        hand_pair_center_drift_max_m=0.05,
        final_hold_box_motion_max_m=0.01,
        active_motor_obstruction=True,
        tracking_passed=True,
    )
    assert passed is True
    assert mode == "paired_single_hand_partial_extraction"
