"""Offline calibrated TCP geometry and Jacobian tests; no Isaac Sim startup."""

import json
from pathlib import Path

import numpy as np
import pytest

from kuavo_isaaclab_scene.robots.end_effector import closed_offsets
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
