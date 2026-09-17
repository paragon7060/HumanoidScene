"""Simulator-independent checks for binary commands and force feedforward."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

torch = pytest.importorskip("torch")


ROOT = Path(__file__).resolve().parents[1]


def _production_method(class_name, method_name):
    path = ROOT / "src/kuavo_isaaclab_scene/rl/mdp/actions.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method_name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


def _force_module():
    from kuavo_isaaclab_scene.robots.claw_assets import force
    return force


def test_policy_scalar_is_executed_as_zero_open_or_one_close():
    process = _production_method("BinaryGripper", "process_actions")
    term = NS(
        _command_enabled=lambda: torch.ones(3, 1, dtype=torch.bool),
        _close_requested=torch.zeros(3, 1, dtype=torch.bool),
        _raw_actions=torch.zeros(3, 1),
        _targets_from_signed=lambda value: value.repeat(1, 2),
        _desired_actions=torch.zeros(3, 2),
        _processed_actions=torch.zeros(3, 2),
        _target_filter=None,
        _position_mapping=None,
    )
    process(term, torch.tensor([[-0.2], [0.3], [0.0]]))
    torch.testing.assert_close(term._raw_actions[:, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(term._processed_actions[:, 0], torch.tensor([1.0, -1.0, 1.0]))


def test_shared_rl_config_selects_binary_grippers_with_50_newtons_total():
    path = ROOT / "src/kuavo_isaaclab_scene/rl/managers/actions.py"
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "hand_action")
    returned = next(node.value for node in ast.walk(function) if isinstance(node, ast.Return))
    assert isinstance(returned, ast.Call) and returned.func.id == "BinaryGripperCfg"
    assert "default_close_force_n()" in ast.unparse(function)
    assert "hand.name in TWO_FINGER_PRESETS" in ast.unparse(function)

    from kuavo_isaaclab_scene.robots.claw_assets.package import default_close_force_n
    assert default_close_force_n() == 50.0


def test_reward_debug_keeps_rl_binary_pd_force_path():
    reward_path = ROOT / "src/kuavo_isaaclab_scene/rl/debug/quest_reward.py"
    reward_source = reward_path.read_text()
    assert "configure_rl_gripper_force" in reward_source
    assert "configure_vr_gripper_force" not in reward_source

    vr_path = ROOT / "src/kuavo_isaaclab_scene/robots/claw_assets/vr.py"
    tree = ast.parse(vr_path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "configure_rl_gripper_force")
    source = ast.unparse(function)
    assert "action.force_sensor_names = None" in source
    assert "action.close_force_n = force_n or None" in source


def test_binary_command_holds_the_reset_pose_until_environment_is_ready():
    process = _production_method("BinaryGripper", "process_actions")
    term = NS(
        _command_enabled=lambda: torch.tensor([[False], [True]]),
        _close_requested=torch.zeros(2, 1, dtype=torch.bool),
        _raw_actions=torch.zeros(2, 1),
        _targets_from_signed=lambda value: value.repeat(1, 2),
        _desired_actions=torch.tensor([[0.25, -0.25], [0.25, -0.25]]),
        _processed_actions=torch.zeros(2, 2),
        _target_filter=None,
        _position_mapping=None,
    )
    process(term, torch.ones(2, 1))
    torch.testing.assert_close(term._desired_actions[0], torch.tensor([0.25, -0.25]))
    torch.testing.assert_close(term._desired_actions[1], torch.tensor([-1.0, -1.0]))


def test_force_feedforward_adds_25_newtons_per_jaw_only_while_closing(monkeypatch):
    module = _force_module()
    monkeypatch.setattr(module, "jaw_leverage_table", lambda _side: (
        np.array([-1.0, 0.0]), np.array([0.10, 0.20])))
    from kuavo_isaaclab_scene.robots.claw_assets.package import default_close_force_n
    drive = module.JawForceFeedforward(
        "left", 2, "cpu", default_close_force_n(), torch.tensor([1.0, -1.0]))
    q = torch.tensor([[-0.5, 0.5], [-0.5, 0.5]])
    torque = drive.advance(q, torch.tensor([[True], [False]]), torch.full((2, 2), 100.0))
    assert drive.per_jaw_n == 25.0
    torch.testing.assert_close(torque[0], torch.tensor([3.75, -3.75]))
    torch.testing.assert_close(torque[1], torch.zeros(2))


def test_force_feedforward_does_not_load_an_empty_closed_mechanical_stop(monkeypatch):
    module = _force_module()
    monkeypatch.setattr(module, "jaw_leverage_table", lambda _side: (
        np.array([-1.0, 0.0]), np.array([0.10, 0.20])))
    drive = module.JawForceFeedforward(
        "right", 1, "cpu", 50.0, torch.tensor([1.0, -1.0]))
    torque = drive.advance(torch.zeros(1, 2), torch.ones(1, 1, dtype=torch.bool),
                           torch.full((1, 2), 100.0))
    torch.testing.assert_close(torque, torch.zeros(1, 2))
