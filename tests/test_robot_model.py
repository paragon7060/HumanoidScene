import argparse

import pytest

from kuavo_isaaclab_scene.robot_model import (
    DEFAULT_ROBOT_MODEL,
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


def test_robot_cli_exposes_both_versions() -> None:
    parser = argparse.ArgumentParser()
    add_robot_model_cli_args(parser)
    assert parser.parse_args(["--robot-model", "s200062"]).robot_model == "s200062"
    assert parser.parse_args(["--robot-model", "s63"]).robot_model == "s63"


def test_integrated_and_external_grippers_cannot_overlap() -> None:
    with pytest.raises(ValueError, match="already contains"):
        validate_robot_gripper(resolve_robot_model("s200062"), "robotiq_2f85")
