import json
import math
import re

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
        "l_b_bar_1_joint",
    )
    assert settings.command_for("right", settings.close_command) == {
        "r_f_bar_1_joint": 0.0,
        "r_b_bar_1_joint": 0.0,
    }
    assert len(teleop_action_names(settings)) == 16


def test_runtime_default_gripper_follows_robot_model(monkeypatch) -> None:
    monkeypatch.delenv(GRIPPER_ENV, raising=False)
    monkeypatch.setenv(ROBOT_MODEL_ENV, "s200062")
    assert resolve_gripper_settings().name == "s200062_integrated"

    monkeypatch.setenv(ROBOT_MODEL_ENV, "s63")
    assert resolve_gripper_settings().name == "robotiq_2f85"

    monkeypatch.setenv(ROBOT_MODEL_ENV, "s56")
    assert resolve_gripper_settings().name == "s56_qiangnao"


def test_s56_qiangnao_integrated_preset_uses_ten_joints_per_hand() -> None:
    settings = load_gripper_settings("s56_qiangnao")
    assert settings.integrated
    assert len(settings.joint_names_for("left")) == 10
    assert settings.joint_names_for("left")[:2] == ("l_thumbCMC", "l_thumbMCP")
    assert settings.close_command["{side}_(index|middle|ring|little)MCP"] == 1.309
    assert settings.body_names_for("right").startswith("r_(palm|")
    assert len(teleop_action_names(settings)) == 16


def test_s56_twofinger_reuses_s200062_joint_commands() -> None:
    donor = load_gripper_settings("s200062_integrated")
    settings = load_gripper_settings("s56_twofinger")
    assert settings.integrated
    assert settings.joint_names == donor.joint_names
    assert settings.open_command == donor.open_command
    assert settings.close_command == donor.close_command
    assert settings.actuator == donor.actuator


@pytest.mark.parametrize(
    ("preset", "urdf"),
    (
        ("s200062_integrated", "kuavo_s200062/urdf/biped_s200062.urdf"),
        ("s56_qiangnao", "kuavo_s56/urdf/kuavo_s56.urdf"),
        ("s56_twofinger", "kuavo_s56/urdf/kuavo_s56_twofinger.urdf"),
    ),
)
def test_integrated_commands_stay_inside_actual_urdf_limits(preset, urdf):
    import xml.etree.ElementTree as ET
    from kuavo_isaaclab_scene.paths import ASSET_DIR

    settings = load_gripper_settings(preset)
    joints = {j.attrib["name"]: j for j in ET.parse(
        ASSET_DIR / urdf
    ).findall("joint")}
    for side in settings.active_sides:
        for command in (settings.open_command, settings.close_command):
            for pattern, position in settings.command_for(side, command).items():
                matched = [joint for name, joint in joints.items() if re.fullmatch(pattern, name)]
                assert matched, pattern
                for joint in matched:
                    limit = joint.find("limit").attrib
                    assert float(limit["lower"]) <= position <= float(limit["upper"])


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
