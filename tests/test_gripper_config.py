import json
import math

import pytest

from kuavo_isaaclab_scene.gripper_config import (
    GRIPPER_ENV,
    gripper_teleop_action,
    load_gripper_settings,
    resolve_gripper_settings,
    teleop_action_names,
)
from kuavo_isaaclab_scene.robot_model import ROBOT_MODEL_ENV


def test_default_robotiq_2f85_preset_has_two_binary_actions() -> None:
    settings = load_gripper_settings()
    assert settings.name == "robotiq_2f85"
    assert settings.active_sides == ("left", "right")
    assert settings.usd_path.endswith("robotiq_2f85/usd/robotiq_2f85.usd")
    assert settings.usd_path_for("right") == settings.usd_path
    assert settings.attachment_mount_body == "base_mount"
    assert settings.close_command[".*_driver_joint$"] == 0.8
    assert settings.close_command[".*_coupler_joint$"] == -0.8
    assert settings.sides["left"].robot_mount_rot == (1.0, 0.0, 0.0, 0.0)
    assert settings.sides["left"].robot_mount_body == "zarm_l7_end_effector"
    assert settings.sides["right"].robot_mount_body == "zarm_r7_end_effector"
    assert len(teleop_action_names(settings)) == 16
    assert gripper_teleop_action(settings, 0.02, 0.08) == (-1.0, 1.0)
    assert gripper_teleop_action(settings, math.nan, 0.01) == (1.0, -1.0)
def test_none_preset_preserves_legacy_teleop_dimension() -> None:
    settings = load_gripper_settings("none")
    assert not settings.enabled
    assert settings.active_sides == ()
    assert len(teleop_action_names(settings)) == 14
    assert gripper_teleop_action(settings, 0.0, 0.0) == ()


def test_s200062_integrated_preset_targets_robot_side_joints() -> None:
    settings = load_gripper_settings("s200062_integrated")
    assert settings.integrated
    assert settings.active_sides == ("left", "right")
    assert settings.asset_name_for("left") == "robot"
    assert settings.joint_names_for("left") == (
        "l_f_bar_1_joint",
        "l_f_bar_3_joint",
        "l_b_bar_1_joint",
        "l_b_bar_3_joint",
    )
    assert settings.command_for("right", settings.close_command) == {
        "r_f_bar_1_joint": 0.55,
        "r_f_bar_3_joint": -0.55,
        "r_b_bar_1_joint": -0.55,
        "r_b_bar_3_joint": 0.55,
    }
    assert len(teleop_action_names(settings)) == 16


def test_runtime_default_gripper_follows_robot_model(monkeypatch) -> None:
    monkeypatch.delenv(GRIPPER_ENV, raising=False)
    monkeypatch.setenv(ROBOT_MODEL_ENV, "s200062")
    assert resolve_gripper_settings().name == "s200062_integrated"

    monkeypatch.setenv(ROBOT_MODEL_ENV, "s63")
    assert resolve_gripper_settings().name == "robotiq_2f85"


def test_custom_preset_resolves_relative_side_usd(tmp_path) -> None:
    payload = {
        "default": "custom",
        "presets": {
            "custom": {
                "enabled": True,
                "usd_path": "base.usd",
                "attachment_mount_body": "mount",
                "joint_names": ["finger_joint"],
                "default_joint_pos": {"finger_joint": 0.0},
                "open_command": {"finger_joint": 0.0},
                "close_command": {"finger_joint": 1.0},
                "actuator": {
                    "effort_limit_sim": 2.0,
                    "stiffness": 10.0,
                    "damping": 1.0,
                    "friction": 0.1,
                },
                "sides": {
                    "left": {
                        "robot_mount_body": "left_wrist",
                        "usd_path": "left.usd",
                    },
                    "right": {"enabled": False},
                },
            }
        },
    }
    path = tmp_path / "grippers.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    settings = load_gripper_settings(config_path=path)
    assert settings.active_sides == ("left",)
    assert settings.usd_path == str((tmp_path / "base.usd").resolve())
    assert settings.usd_path_for("left") == str((tmp_path / "left.usd").resolve())


def test_invalid_quaternion_is_rejected(tmp_path) -> None:
    source = load_gripper_settings().config_path
    payload = json.loads(source.read_text(encoding="utf-8"))
    selected = payload["default"]
    payload["presets"][selected]["sides"]["left"]["robot_mount_rot"] = [0, 0, 0, 0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="quaternion"):
        load_gripper_settings(config_path=path)
