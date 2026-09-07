"""Exercise the last-action drive override with a fake Isaac API, no simulator."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch


@pytest.fixture
def action(monkeypatch):
    for name in ("isaaclab", "isaaclab.managers", "isaaclab.utils"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    sys.modules["isaaclab.managers"].ActionTerm = object
    sys.modules["isaaclab.managers"].ActionTermCfg = object
    sys.modules["isaaclab.utils"].configclass = lambda cls: cls
    path = Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/teleop/self_collision_action.py"
    spec = importlib.util.spec_from_file_location("kuavo_isaaclab_scene.teleop._guard_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    term = module.SelfCollisionAction.__new__(module.SelfCollisionAction)
    term.validated = True
    term._joint_ids = list(range(14))
    # These are properties on real ActionTerm; this fake base intentionally
    # exposes attributes so the production apply_actions can be tested alone.
    term.device = "cpu"
    names = [f"{side}{i}" for side in ("left", "right") for i in range(7)]
    term.model = SimpleNamespace(names=names, edges=[], pair_name=lambda index: "left / right")
    term._physics_steps = 0
    term.step_modified = False
    term.step_minimum_distance = float("inf")
    outputs = {}
    term._asset = SimpleNamespace(
        data=SimpleNamespace(joint_pos=torch.zeros(1, 14), joint_pos_target=torch.ones(1, 14),
                             joint_vel=torch.zeros(1, 14), joint_vel_target=torch.zeros(1, 14)),
        set_joint_position_target=lambda value, ids: outputs.update(position=value.clone()),
        set_joint_velocity_target=lambda value, ids: outputs.update(velocity=value.clone()),
    )
    arms = {side: SimpleNamespace(_joint_names=names[i*7:(i+1)*7], _joint_command=torch.zeros(1, 7),
                                  _joint_velocity=torch.zeros(1, 7), _gravity_bias=torch.full((1, 7), .01))
            for i, side in enumerate(("left", "right"))}
    term._env = SimpleNamespace(physics_dt=.01, step_dt=.01,
                               action_manager=SimpleNamespace(get_term=lambda name: arms[name.removesuffix("_arm")]))
    term._cached_target = None
    term._cached_velocity = None
    term._recording = True
    term._collision_event = None
    term.step_collision = False
    term._checked_this_tick = False
    term._test_clearance_violation = module.ClearanceViolation
    term.guard = SimpleNamespace(filter_light=lambda *args: np.full(14, .02),
                                 check_state=lambda *args: np.array([.01]),
                                 cached_path_safe=lambda *args: True,
                                 status={"modified": True, "minimum_distance_m": .01,
                                         "pair": "left / right", "scale": 1.})
    return term, outputs, arms


def test_guard_overrides_both_arm_drives_and_removes_bias_from_feedback(action):
    term, outputs, arms = action
    term.apply_actions()
    torch.testing.assert_close(outputs["position"], torch.full((1, 14), .02))
    torch.testing.assert_close(outputs["velocity"], torch.full((1, 14), 2.))
    for arm in arms.values():
        torch.testing.assert_close(arm._joint_command, torch.full((1, 7), .01))
    assert term.step_modified and term.step_minimum_distance == .01


def test_collision_failure_overrides_pending_drives_and_raises_before_physics(action):
    term, outputs, _ = action
    def fail(*args):
        raise RuntimeError("collision")
    term.guard.filter_light = fail
    with pytest.raises(RuntimeError, match="collision"):
        term.apply_actions()
    assert outputs["position"].count_nonzero() == 0
    assert outputs["velocity"].count_nonzero() == 0


def test_recording_clearance_violation_holds_without_terminating_process(action):
    term, outputs, _ = action
    term.guard.filter_light = lambda *args: (_ for _ in ()).throw(
        term._test_clearance_violation("current pose collision")
    )
    term.apply_actions()
    assert outputs["position"].count_nonzero() == 0
    assert outputs["velocity"].count_nonzero() == 0
    assert term.consume_collision_event() == {
        "message": "current pose collision",
        "recording": True,
    }


def test_nonrecording_clearance_violation_is_monitor_only(action):
    term, outputs, _ = action
    term.set_recording(False)
    term.guard.filter_light = lambda *args: (_ for _ in ()).throw(
        term._test_clearance_violation("current pose collision")
    )
    term.apply_actions()
    assert outputs == {}
    assert term.step_collision
    assert term.consume_collision_event()["recording"] is False


def test_nonrecording_monitor_queries_only_once_per_control_tick(action):
    term, outputs, _ = action
    term.set_recording(False)
    calls = []
    term.guard.filter_light = lambda *args: calls.append("query") or np.full(14, .02)
    term.apply_actions()
    term.apply_actions()
    assert calls == ["query"]
    assert outputs == {}


def test_blocked_recording_target_reports_collision_event(action):
    term, outputs, _ = action
    term.guard.status["scale"] = 0.0
    term.guard.filter_light = lambda *args: np.zeros(14)
    term.apply_actions()
    assert outputs["position"].count_nonzero() == 0
    assert term.consume_collision_event() == {
        "message": "Command would violate self-collision clearance",
        "recording": True,
    }


def test_reset_requires_revalidation(action):
    term, _, _ = action
    term.reset()
    with pytest.raises(RuntimeError, match="not validated"):
        term.apply_actions()


def test_intervening_physics_steps_reuse_command_without_collision_queries(action):
    term, _, _ = action
    calls = []
    term.apply_actions()
    term.guard.filter_light = lambda *args: (_ for _ in ()).throw(AssertionError("Unnecessary query"))
    term.guard.check_state = lambda *args: calls.append("state") or np.array([.01])
    term.guard.cached_path_safe = lambda *args: calls.append("path") or True
    term.apply_actions()
    assert calls == []
    term.process_actions(None)
    assert term._cached_target is None
    term.guard.filter_light = lambda *args: calls.append("new control tick") or np.full(14, .02)
    term.apply_actions()
    assert calls == ["new control tick"]
