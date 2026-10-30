"""Offline calibrated TCP geometry and Jacobian tests; no Isaac Sim startup."""

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from kuavo_isaaclab_scene.robots.end_effector import (
    CENTER_TOOL_FRAMES,
    center_position_from_original_pose,
    closed_offsets,
    original_position_for_center_target,
    urdf_with_center_frames,
)
from kuavo_isaaclab_scene.robots.gripper_config import load_gripper_settings
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.teleop.urdf_arm_ik import UrdfArm

ROOT = Path(__file__).resolve().parents[1]


def definition():
    return json.loads((ROOT / "configs/grasp_reference_points.json").read_text())


def test_mirrored_points_and_packaged_default():
    data = definition()
    assert data == json.loads((ROOT / "src/kuavo_isaaclab_scene/configs/grasp_reference_points.json").read_text())
    for jaw in "fb":
        np.testing.assert_allclose(np.array(data["offsets"][f"l_{jaw}_finger"]) * [1, -1, 1],
                                   data["offsets"][f"r_{jaw}_finger"], atol=1e-12)


def test_closed_center_is_mirrored_and_not_average_of_unrelated_local_frames():
    model = resolve_robot_model("s200062")
    result = closed_offsets(model.urdf_path, definition()["offsets"], load_gripper_settings("s200062_integrated"))
    np.testing.assert_allclose(np.array(result["left"]) * [1, -1, 1], result["right"], atol=1e-12)
    np.testing.assert_allclose(result["left"], [.00018399152322091938, -.0011112558194255602, -.055497895148762716], atol=1e-10)


def test_center_and_original_target_positions_are_inverse_transforms():
    angle = np.deg2rad(90.0)
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    original_position = np.array([0.45, -0.2, 1.1])
    offset = np.array([0.01, -0.02, -0.05])

    center = center_position_from_original_pose(
        original_position, rotation, offset
    )
    recovered = original_position_for_center_target(center, rotation, offset)

    np.testing.assert_allclose(center, original_position + rotation @ offset)
    np.testing.assert_allclose(recovered, original_position)


def test_planner_urdf_adds_fixed_center_frames_without_moving_original_eefs():
    model = resolve_robot_model("s200062")
    original = ET.parse(model.urdf_path).getroot()
    augmented = ET.fromstring(urdf_with_center_frames(Path(model.urdf_path).read_text()))

    for side, letter in (("left", "l"), ("right", "r")):
        original_joint = original.find(
            f"./joint[@name='zarm_{letter}7_end_effector_joint']"
        )
        assert original_joint.find("origin").get("xyz") == "0 0.0 -0.17"
        joint = augmented.find(
            f"./joint[@name='{CENTER_TOOL_FRAMES[side]}_joint']"
        )
        assert joint.get("type") == "fixed"
        assert joint.find("parent").get("link") == f"zarm_{letter}7_end_effector"
        assert joint.find("child").get("link") == CENTER_TOOL_FRAMES[side]
        np.testing.assert_allclose(
            np.fromstring(joint.find("origin").get("xyz"), sep=" "),
            closed_offsets(
                model.urdf_path,
                definition()["offsets"],
                load_gripper_settings("s200062_integrated"),
            )[side],
            atol=1e-12,
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_center_jacobian_finite_difference(side):
    model = resolve_robot_model("s200062")
    arm = UrdfArm(model.urdf_path, side)
    arm.tool_offset = np.array(closed_offsets(model.urdf_path, definition()["offsets"],
                                             load_gripper_settings("s200062_integrated"))[side])
    q = np.array([-.2, .5, .3, -1.5, .2, -.4, .1])
    p, _, jac, points = arm.fk(q)
    np.testing.assert_allclose(points["endeffector_center"], p)
    for i in range(7):
        dq = np.eye(7)[i] * 1e-6
        numerical = (arm.fk(q + dq)[0] - arm.fk(q - dq)[0]) / 2e-6
        np.testing.assert_allclose(jac[:3, i], numerical, atol=1e-8)
