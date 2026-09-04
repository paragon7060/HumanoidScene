import argparse

import pytest

from kuavo_isaaclab_scene.robots.robot_model import (
    DEFAULT_ROBOT_MODEL,
    S56_ACTUATOR_LIMITS,
    S56_MUJOCO_ARMATURE,
    S56_MUJOCO_FRICTIONLOSS,
    add_robot_model_cli_args,
    resolve_robot_model,
    validate_robot_gripper,
)


def test_default_robot_is_full_s200062() -> None:
    model = resolve_robot_model(DEFAULT_ROBOT_MODEL)
    assert model.name == "s200062"
    assert model.has_integrated_grippers
    assert model.usd_path.endswith("kuavo_s200062/usd/kuavo_s200062_fixed.usd")
    assert model.head_camera_body == "camera"
    assert model.wrist_camera_bodies == {
        "left": "l_d405_camera",
        "right": "r_d405_camera",
    }


def test_s63_remains_selectable_for_comparison() -> None:
    model = resolve_robot_model("s63")
    assert not model.has_integrated_grippers
    assert model.usd_path.endswith("kuavo_s63/usd/kuavo_s63_fixed.usd")
    assert model.wrist_camera_bodies["left"] == "zarm_l7_end_effector"


def test_s56_biped_uses_source_home_height_and_qiangnao_hands() -> None:
    model = resolve_robot_model("s56")
    assert model.has_integrated_grippers
    assert not model.has_wheel_base
    assert model.default_gripper_preset == "s56_qiangnao"
    assert model.usd_path.endswith("kuavo_s56/usd/kuavo_s56_fixed.usd")
    assert model.urdf_path.endswith("kuavo_s56/urdf/kuavo_s56.urdf")
    assert model.spawn_position((1.0, 2.0, 0.1)) == (1.0, 2.0, 1.08)
    assert model.teleop_body_joint_names[:2] == ("leg_l1_joint", "leg_l2_joint")


def test_s56_twofinger_selection_switches_complete_asset_and_camera_rig() -> None:
    model = resolve_robot_model("s56", "s56_twofinger")
    assert model.name == "s56"
    assert model.integrated_gripper_preset == "s56_twofinger"
    assert model.usd_path.endswith(
        "kuavo_s56_twofinger/usd/kuavo_s56_twofinger_fixed.usd"
    )
    assert model.urdf_path.endswith("kuavo_s56/urdf/kuavo_s56_twofinger.urdf")
    assert model.wrist_camera_bodies == {
        "left": "l_d405_camera",
        "right": "r_d405_camera",
    }


def test_s56_twofinger_cli_environment_selects_variant(monkeypatch) -> None:
    monkeypatch.setenv("KUAVO_ROBOT_MODEL", "s56")
    monkeypatch.setenv("KUAVO_GRIPPER", "s56_twofinger")
    assert resolve_robot_model().integrated_gripper_preset == "s56_twofinger"


def test_s56_none_selection_switches_to_bare_wrist_asset() -> None:
    model = resolve_robot_model("s56", "none")
    assert not model.has_integrated_grippers
    assert model.default_gripper_preset == "none"
    assert model.usd_path.endswith("kuavo_s56_bare/usd/kuavo_s56_bare_fixed.usd")
    assert model.urdf_path.endswith("kuavo_s56/urdf/kuavo_s56_bare.urdf")


def test_s56_actuator_physics_matches_upstream_mujoco_and_urdf() -> None:
    assert S56_MUJOCO_ARMATURE == 0.05
    assert S56_MUJOCO_FRICTIONLOSS == 0.02
    assert S56_ACTUATOR_LIMITS["lower_body"]["effort_limit_sim"]["leg_[lr]4_joint"] == 280.0
    assert S56_ACTUATOR_LIMITS["arms"]["effort_limit_sim"]["zarm_[lr][567]_joint"] == 14.1
    assert S56_ACTUATOR_LIMITS["upper_body"]["effort_limit_sim"] == {
        "waist_yaw_joint": 102.0,
        "zhead_1_joint": 1.5,
        "zhead_2_joint": 12.0,
    }


def test_robot_cli_exposes_all_versions() -> None:
    parser = argparse.ArgumentParser()
    add_robot_model_cli_args(parser)
    assert parser.parse_args(["--robot-model", "s200062"]).robot_model == "s200062"
    assert parser.parse_args(["--robot-model", "s63"]).robot_model == "s63"
    assert parser.parse_args(["--robot-model", "s56"]).robot_model == "s56"


def test_integrated_and_external_grippers_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="already contains"):
        validate_robot_gripper(resolve_robot_model("s200062"), "robotiq_2f85")
