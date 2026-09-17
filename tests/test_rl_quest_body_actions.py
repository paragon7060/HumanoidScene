"""Exercise the Quest body adapter without starting Isaac Sim or an XR session."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from kuavo_isaaclab_scene.core.paths import ASSET_DIR
from kuavo_isaaclab_scene.teleop.teleop_body import TeleopBodyMapper


def control_class():
    # Isaac action classes require an active Kit runtime. Compile the actual
    # adapter methods with fake action terms to test their tensor contracts.
    path = Path(__file__).parents[1] / "src/kuavo_isaaclab_scene/rl/debug/quest_control.py"
    tree = ast.parse(path.read_text())
    delta = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "normalized_delta")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "QuestRLControl")
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in {"action", "reset"}]
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.Module(body=[delta, cls], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["QuestRLControl"]


@pytest.mark.parametrize("per_joint_scale", [False, True])
def test_whole_body_routes_height_and_waist_to_separate_terms(per_joint_scale):
    cls = control_class()
    control = cls.__new__(cls)
    control.env = NS(device="cpu", step_dt=1 / 30, action_manager=NS(total_action_dim=25))
    control.sides = ()
    control.term_slices = {"base": slice(0, 3), "upper_body": slice(3, 18), "height": slice(18, 21)}
    control.base = NS(_scale=torch.tensor([.25, .25, .70]))
    control.height = NS(processed_actions=torch.tensor([[.20, -.40, .30]]), _scale=.015)
    control.upper = NS(processed_actions=torch.zeros(1, 15),
                       _scale=torch.full((1, 15), .02) if per_joint_scale else .02)
    control.upper.processed_actions[0, 0] = .40
    control.waist_column = 0
    control.body_mapper = NS(advance=lambda *a, **kw: np.array(
        [.125, -.125, .35, .2075, -.4075, .33, .41], dtype=np.float32))

    action = control.action({"left": None, "right": None})

    assert action.shape == (1, 25)
    torch.testing.assert_close(action[:, :3], torch.tensor([[.5, -.5, .5]]))
    torch.testing.assert_close(action[:, 18:21], torch.tensor([[.5, -.5, 1.]]))
    torch.testing.assert_close(action[:, 3:4], torch.tensor([[.5]]))
    assert not torch.count_nonzero(action[:, 4:18])


def test_pause_reset_preserves_commanded_torso_instead_of_adopting_sag():
    cls = control_class()
    control = cls.__new__(cls)
    control.mapper = NS(reset=lambda: None)
    control.solvers = {}
    control.robot = NS(data=NS(joint_pos=torch.tensor([[.1, -.1, .35, .4]])))
    control.height = NS(processed_actions=torch.tensor([[.2, -.4, .3]]), _scale=.015)
    control.upper = NS(processed_actions=torch.tensor([[.6]]), _scale=.035)
    control.waist_column = 0
    control.body_joint_ids = [0, 1, 2, 3]
    captured = []
    control.body_mapper = NS(reset=lambda joints: captured.append(joints))

    control.reset()

    np.testing.assert_allclose(captured[0], [.2, -.4, .3, .6])

    control.body_mapper = TeleopBodyMapper(ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf")
    control.env = NS(device="cpu", step_dt=1 / 30, action_manager=NS(total_action_dim=7))
    control.sides = ()
    control.term_slices = {"base": slice(0, 3), "upper_body": slice(3, 4), "height": slice(4, 7)}
    control.base = NS(_scale=torch.tensor([.25, .25, .70]))
    neutral = np.zeros((2, 7))
    neutral[0, 3] = 1
    # Settling and resume reset the adapter repeatedly while actual joints sag.
    # They must not issue a command toward that unrequested measured posture.
    for _ in range(20):
        control.robot.data.joint_pos += .001
        control.reset()
        action = control.action({"left": neutral, "right": neutral})
        torch.testing.assert_close(action, torch.zeros_like(action))


def test_quest_trigger_drives_the_rl_gripper_as_zero_open_one_close():
    cls = control_class()
    control = cls.__new__(cls)
    gripper = NS(_close_requested=torch.zeros(1, 1, dtype=torch.bool))
    control.env = NS(device="cpu", step_dt=1 / 30,
                     action_manager=NS(total_action_dim=1, get_term=lambda _name: gripper))
    control.sides = ("left",)
    control.term_slices = {"upper_body": slice(0, 0), "left_gripper": slice(0, 1)}
    control.upper = NS(processed_actions=torch.zeros(1, 0), _scale=1.0)
    control.columns = {"left": []}
    control.frames = NS(center_pose_w=torch.zeros(1, 2, 7))
    control.mapper = NS(target=lambda *args, **kwargs: np.zeros(7))
    control.xr = NS(controller_aim_pose=lambda _side: None)
    control.pose = lambda _body=None: np.zeros(7)
    control.torso = 0
    control.body_mapper = None
    control.solvers = {"left": NS(
        _joint_command=torch.zeros(1, 0),
        process_actions=lambda _goal: None,
    )}
    packet = np.zeros((2, 7))

    packet[1, 2] = 0.0
    torch.testing.assert_close(control.action({"left": packet}), torch.tensor([[0.0]]))
    packet[1, 2] = 0.5
    torch.testing.assert_close(control.action({"left": packet}), torch.tensor([[1.0]]))


@pytest.mark.parametrize("mode,compensated,expected_height", [
    ("arms-only", False, (400., 40.)), ("whole-body", False, (8000., 200.)),
    ("arms-only", True, (400., 40.)), ("whole-body", True, (400., 40.)),
])
def test_experiment_supports_released_torso_without_changing_locked_mode(mode, compensated, expected_height):
    path = Path(__file__).parents[1] / "configs/rl_pick_arms_only.py"
    configure = runpy.run_path(str(path))["configure"]
    height = NS(stiffness=400., damping=40.)
    yaw = NS(stiffness=120., damping=15.)
    cfg = NS(task=NS(control_mode=mode),
             scene=NS(robot=NS(class_type=NS(gravity_compensation_enabled=compensated),
                              actuators={"height_axis": height, "upper_body": yaw})),
             rewards=NS(prelift_disturbance=NS(), orientation=NS(params={})),
             actions=NS(upper_body=NS(), left_gripper=NS(), right_gripper=NS()))
    configure(cfg, NS(policy=NS(), algorithm=NS(), _skip_training_setup=True))
    assert (height.stiffness, height.damping) == expected_height
    assert (yaw.stiffness, yaw.damping) == ((800., 50.) if mode == "whole-body" and not compensated else (120., 15.))
