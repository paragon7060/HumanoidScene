import json
from pathlib import Path

import pytest

from data_collection.task1.contract import ARM_JOINT_NAMES, WAIST_ARM_JOINT_NAMES
from data_collection.task1.scenario import (
    DEFAULT_SCENARIO_DIR,
    build_physical_executor_argv,
    load_scenario_config,
    validate_plan_files,
)


WAIST16_CONFIG = DEFAULT_SCENARIO_DIR / "single_medium_box_waist16_verified_v1.json"
ARM14_CONFIG = DEFAULT_SCENARIO_DIR / "single_medium_box_arm14_baseline_v1.json"
PAIR_CONFIG = (
    DEFAULT_SCENARIO_DIR / "paired_medium_boxes_mms_left_arm14_experimental_v1.json"
)
VERIFIED_SINGLE_ARM_DIR = DEFAULT_SCENARIO_DIR / "verified" / "single_arm"
VERIFIED_SINGLE_ARM_CONFIGS = {
    "left_arm14": (
        VERIFIED_SINGLE_ARM_DIR / "single_medium_box_left_arm14_verified_v1.json",
        "left",
        14,
        "verified_partial_extraction",
        True,
        "06448da0e9ee17d69680e31f5c00d60ec6593dad92803348ea92804df209b2ea",
    ),
    "left_arm14_waist16": (
        VERIFIED_SINGLE_ARM_DIR
        / "single_medium_box_left_arm14_waist16_verified_v1.json",
        "left",
        16,
        "verified_partial_extraction",
        True,
        "a0a484966746e528ab860bbaffc18789d0733abd1003f2ab25d3626f8adf53e9",
    ),
    "right_arm14": (
        VERIFIED_SINGLE_ARM_DIR
        / "single_medium_box_right_arm14_verified_failure_v1.json",
        "right",
        14,
        "verified_failure",
        False,
        "c58404b9d062851c7f4467ce83e56080d63dcf407f3068706b05d15b6b619b08",
    ),
    "right_arm14_waist16": (
        VERIFIED_SINGLE_ARM_DIR
        / "single_medium_box_right_arm14_waist16_verified_v1.json",
        "right",
        16,
        "verified_partial_extraction",
        True,
        "f22a3fdf64e26cd3b3642ae65f2bebb7b8b26f67676292ec0b48eb2ee4233b85",
    ),
}


def test_packaged_single_box_configs_keep_common_physical_contract():
    waist16 = load_scenario_config(WAIST16_CONFIG)
    arm14 = load_scenario_config(ARM14_CONFIG)

    for scenario in (waist16, arm14):
        assert scenario["robot"] == {
            "model": "s200062",
            "gripper_preset": "s200062_integrated",
            "tcp_frame": "endeffector_center",
        }
        assert scenario["scene"] == {
            "rack_box_poses": "configs/rack_box_poses_task1_centered.json",
            "target_box": "medium_box_0",
            "clear_same_shelf_boxes": True,
        }
        assert scenario["planning"]["torso_height_m"] == 0.25
        assert scenario["gripper_contract"]["actuator"] == {
            "effort_limit_sim": 100.0,
            "stiffness": 4000.0,
            "damping": 400.0,
        }
        assert scenario["gripper_contract"]["finger_contact"] == {
            "static_friction": 20.0,
            "dynamic_friction": 16.0,
        }


def test_mms_left_pair_scenario_matches_pose_contract():
    scenario = load_scenario_config(PAIR_CONFIG)

    assert scenario["schema_version"] == 2
    assert scenario["scene"]["paired_boxes"] == ["medium_box_0", "medium_box_1"]
    assert scenario["scene"]["pair_grasp"] == "adjacent_inner_flaps"
    assert scenario["scene"]["clear_same_shelf_boxes"] is False
    assert scenario["planning"]["active_arm"] == "left"
    assert scenario["planning"]["cspace_joint_names"] == ARM_JOINT_NAMES
    assert scenario["execution"]["active_gripper"] == "left"
    assert scenario["execution"]["pair_separation_drift_max_m"] == 0.01
    assert scenario["execution"]["retention_drift_max_m"] == 0.05
    assert scenario["verification"]["state"] == "experimental_unverified"
    assert scenario["verification"]["expected_passed"] is None


def test_pair_scenario_forwards_pair_arguments(tmp_path):
    scenario = load_scenario_config(PAIR_CONFIG)
    argv = build_physical_executor_argv(
        scenario,
        approach_plan=tmp_path / "approach.json",
        retreat_plan=tmp_path / "retreat.json",
        output=tmp_path / "report.json",
        video_out=tmp_path / "video.mp4",
    )

    pair_index = argv.index("--paired-boxes")
    assert argv[pair_index + 1 : pair_index + 3] == [
        "medium_box_0",
        "medium_box_1",
    ]
    assert argv[argv.index("--scenario-path") + 1] == str(PAIR_CONFIG.resolve())
    assert "--no-clear-same-shelf-boxes" in argv
    assert argv[argv.index("--pair-separation-drift-max-m") + 1] == "0.01"


def test_packaged_single_box_configs_separate_waist16_and_arm14_planning():
    waist16 = load_scenario_config(WAIST16_CONFIG)
    arm14 = load_scenario_config(ARM14_CONFIG)

    assert waist16["planning"]["include_waist"] is True
    assert waist16["planning"]["arm_only_baseline"] is False
    assert waist16["planning"]["cspace_dof"] == 16
    assert waist16["planning"]["cspace_joint_names"] == WAIST_ARM_JOINT_NAMES
    assert waist16["verification"]["state"] == "verified"

    assert arm14["planning"]["include_waist"] is False
    assert arm14["planning"]["arm_only_baseline"] is True
    assert arm14["planning"]["cspace_dof"] == 14
    assert arm14["planning"]["cspace_joint_names"] == ARM_JOINT_NAMES
    assert arm14["verification"]["state"] == "verified"
    assert arm14["verification"]["expected_passed"] is True
    assert arm14["verification"]["acceptance_mode"] == "stable_bimanual_retention"
    assert arm14["verification"]["reference_video_sha256"] == (
        "91a86d7c06088254bd8edaa7c1ab4e16c89fdeecd655bf2345f1004c0584622b"
    )


@pytest.mark.parametrize(
    ("config_path", "active_gripper", "planning_dof", "state", "passed", "video_sha256"),
    VERIFIED_SINGLE_ARM_CONFIGS.values(),
    ids=VERIFIED_SINGLE_ARM_CONFIGS.keys(),
)
def test_verified_single_arm_configs_preserve_placement_execution_and_evidence(
    config_path, active_gripper, planning_dof, state, passed, video_sha256
):
    scenario = load_scenario_config(config_path)

    assert scenario["scene"]["rack_box_poses"] == (
        "configs/rack_box_poses_task1_centered.json"
    )
    assert scenario["scene"]["target_box"] == "medium_box_0"
    assert scenario["execution"]["active_gripper"] == active_gripper
    assert scenario["execution"]["waypoint_tracking_tolerance_rad"] == 0.06
    assert scenario["execution"]["partial_extraction_front_progress_min_m"] == 0.05
    assert scenario["execution"]["partial_extraction_front_inside_max_m"] == 0.05
    assert scenario["planning"]["cspace_dof"] == planning_dof
    assert scenario["planning"]["include_waist"] is (planning_dof == 16)
    assert scenario["verification"]["state"] == state
    assert scenario["verification"]["expected_passed"] is passed
    assert scenario["verification"]["reference_video_sha256"] == video_sha256


@pytest.mark.parametrize(
    ("config_path", "active_gripper", "_planning_dof", "_state", "_passed", "_sha"),
    VERIFIED_SINGLE_ARM_CONFIGS.values(),
    ids=VERIFIED_SINGLE_ARM_CONFIGS.keys(),
)
def test_verified_single_arm_configs_forward_active_hand_and_partial_gate(
    tmp_path,
    config_path,
    active_gripper,
    _planning_dof,
    _state,
    _passed,
    _sha,
):
    scenario = load_scenario_config(config_path)
    argv = build_physical_executor_argv(
        scenario,
        approach_plan=tmp_path / "approach.json",
        retreat_plan=tmp_path / "retreat.json",
        output=tmp_path / "report.json",
        video_out=tmp_path / "video.mp4",
    )

    assert argv[argv.index("--active-gripper") + 1] == active_gripper
    assert argv[argv.index("--partial-extraction-front-progress-min-m") + 1] == "0.05"
    assert argv[argv.index("--partial-extraction-front-inside-max-m") + 1] == "0.05"


def test_single_arm_config_rejects_success_claim_for_verified_failure(tmp_path):
    source = VERIFIED_SINGLE_ARM_CONFIGS["right_arm14"][0]
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["verification"]["expected_passed"] = True
    path = tmp_path / "bad_verified_failure.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="verified_failure"):
        load_scenario_config(path)


def test_physical_executor_argv_reproduces_verified_single_box_settings(tmp_path):
    scenario = load_scenario_config(WAIST16_CONFIG)
    argv = build_physical_executor_argv(
        scenario,
        approach_plan=tmp_path / "approach.json",
        retreat_plan=tmp_path / "retreat.json",
        output=tmp_path / "report.json",
        video_out=tmp_path / "video.mp4",
    )

    assert argv == [
        "--approach-plan", str(tmp_path / "approach.json"),
        "--retreat-plan", str(tmp_path / "retreat.json"),
        "--output", str(tmp_path / "report.json"),
        "--video-out", str(tmp_path / "video.mp4"),
        "--rack-box-poses", str(
            WAIST16_CONFIG.parents[3] / "configs/rack_box_poses_task1_centered.json"
        ),
        "--initial-state", "meta_default",
        "--seed", "0",
        "--torso-height-m", "0.25",
        "--settle-steps", "120",
        "--approach-steps-per-waypoint", "4",
        "--close-steps", "180",
        "--closed-hold-steps", "60",
        "--retreat-steps-per-waypoint", "8",
        "--final-hold-steps", "120",
        "--capture-stride", "4",
        "--waypoint-tracking-tolerance-rad", "0.01",
        "--approach-box-motion-max-m", "0.003",
        "--retreat-box-motion-min-m", "0.03",
        "--retention-drift-max-m", "0.05",
        "--motor-obstruction-min-rad", "0.005",
        "--direct-approach-replay",
        "--kinematic-direct-approach-render",
        "--overwrite-video",
    ]


def test_plan_validation_rejects_mixing_arm14_plan_with_waist16_config(tmp_path):
    scenario = load_scenario_config(WAIST16_CONFIG)
    approach = tmp_path / "approach.json"
    retreat = tmp_path / "retreat.json"
    payload = {
        "status": "SUCCESS",
        "joint_names": ARM_JOINT_NAMES,
        "waypoint_q_rad": [[0.0] * 14],
    }
    approach.write_text(json.dumps(payload), encoding="utf-8")
    retreat.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario requires 16DoF"):
        validate_plan_files(scenario, approach, retreat)


def test_scenario_rejects_inconsistent_dof_contract(tmp_path):
    payload = json.loads(WAIST16_CONFIG.read_text(encoding="utf-8"))
    payload["planning"]["cspace_dof"] = 14
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="cspace_dof"):
        load_scenario_config(path)
