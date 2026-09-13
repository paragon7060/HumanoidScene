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
    assert arm14["verification"]["state"] == "configured_baseline"


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
