import json
import math

import pytest

from kuavo_isaaclab_scene.gripper_config import (
    gripper_teleop_action,
    load_gripper_settings,
    teleop_action_names,
)


def test_default_allegro_preset_has_two_binary_actions() -> None:
    settings = load_gripper_settings()
    assert settings.name == "allegro"
    assert settings.active_sides == ("left", "right")
    assert settings.usd_path.endswith("allegro_hand_instanceable.usd")
    assert len(teleop_action_names(settings)) == 16
    assert gripper_teleop_action(settings, 0.02, 0.08) == (-1.0, 1.0)
    assert gripper_teleop_action(settings, math.nan, 0.01) == (1.0, -1.0)


def test_none_preset_preserves_legacy_teleop_dimension() -> None:
    settings = load_gripper_settings("none")
    assert not settings.enabled
    assert settings.active_sides == ()
    assert len(teleop_action_names(settings)) == 14
    assert gripper_teleop_action(settings, 0.0, 0.0) == ()


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
    payload["presets"]["allegro"]["sides"]["left"]["robot_mount_rot"] = [0, 0, 0, 0]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="quaternion"):
        load_gripper_settings(config_path=path)
