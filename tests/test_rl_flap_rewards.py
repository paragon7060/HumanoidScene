"""Exercise real reward/phase code with CPU state, without starting Isaac Sim."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


@pytest.fixture
def modules(monkeypatch):
    # Only the simulator imports are stubbed. The reward functions and refresh
    # implementation below are loaded unmodified from the production modules.
    managers = ModuleType("isaaclab.managers")
    managers.CommandTerm = object
    maths = ModuleType("isaaclab.utils.math")
    maths.quat_mul = maths.quat_conjugate = lambda *args: None
    geometry = ModuleType("kuavo_isaaclab_scene.rl.mdp.geometry")
    for name in ("rotate", "unrotate", "yaw", "wrap_angle", "projected_half_size", "slot_offsets"):
        setattr(geometry, name, lambda *args: None)
    for name, module in ((managers.__name__, managers), (maths.__name__, maths), (geometry.__name__, geometry)):
        monkeypatch.setitem(sys.modules, name, module)
    root = Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/rl/mdp"
    loaded = []
    for name in ("commands", "rewards"):
        key = f"kuavo_isaaclab_scene.rl.mdp.{name}"
        spec = importlib.util.spec_from_file_location(key, root / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, key, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


def test_flap_shaping_uses_only_the_selected_hand(modules):
    _, rewards = modules
    t = SimpleNamespace(spec=task_spec("pick", grasp_hand="right"), settling=None,
        hand_target_distance=torch.tensor([[0., .1]]), grasp_alignment=torch.ones(1, 2),
        finger_grasp_contacts=torch.tensor([[[True, True], [True, False]]]),
        hand_grasp_flags=torch.tensor([[True, False]]), refresh=lambda: None)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t))
    reach = rewards.flap_reaching(env)
    contact = rewards.flap_contact(env)
    t.hand_target_distance[:, 0] = 100
    t.grasp_alignment[:, 0] = 0
    t.finger_grasp_contacts[:, 0] = False
    t.hand_grasp_flags[:, 0] = False
    torch.testing.assert_close(rewards.flap_reaching(env), reach)
    torch.testing.assert_close(rewards.flap_contact(env), contact)
    assert contact.item() == pytest.approx(.125)
    t.finger_grasp_contacts[:, 1] = True
    t.hand_grasp_flags[:, 1] = True
    assert rewards.flap_contact(env).item() == pytest.approx(1.25)
    t.settling = SimpleNamespace(ready=torch.tensor([False]))
    assert rewards.flap_reaching(env).item() == 0
    assert rewards.flap_contact(env).item() == 0


def test_lift_requires_grasp_and_discrete_bonuses_are_dt_independent(modules):
    _, rewards = modules
    t = SimpleNamespace(spec=task_spec("pick", lift_height=.06), settling=None, ids=torch.arange(2),
        reward_box=torch.zeros(2, dtype=torch.long), reward_phase=torch.ones(2, dtype=torch.long),
        centers=torch.tensor([[[0., 0., .3]], [[0., 0., .3]]]), initial_z=torch.full((2, 1), .2),
        grasped=torch.tensor([True, False]), transition=torch.tensor([True, False]),
        success=torch.tensor([True, False]), failure=torch.tensor([False, True]), refresh=lambda: None)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t),
        scene=SimpleNamespace(env_origins=torch.zeros(2, 3)))
    assert rewards.lift(env).tolist() == [1., 0.]
    for dt in (1/30, 1/60):
        env.step_dt = dt
        for name in ("stage_completed", "success", "failure"):
            expected = t.transition if name == "stage_completed" else getattr(t, name)
            torch.testing.assert_close(getattr(rewards, name)(env) * dt, expected.float())


def test_pick_dwell_resets_on_lost_grasp_and_refresh_is_once_per_step(modules):
    commands, _ = modules
    n = 2
    zeros = lambda: torch.zeros(n)
    flags = lambda: torch.zeros(n, dtype=torch.bool)
    t = SimpleNamespace(spec=task_spec("pick", hold_seconds=.3, lift_height=.06), settling=None,
        ids=torch.arange(n), last_step=torch.full((n,), -1), phase=torch.ones(n, dtype=torch.long),
        reward_phase=torch.ones(n, dtype=torch.long), active_box=torch.zeros(n, dtype=torch.long),
        reward_box=torch.zeros(n, dtype=torch.long), transition=flags(),
        centers=torch.tensor([[[0., 0., .3]], [[0., 0., .3]]]), initial_z=torch.full((n, 1), .2),
        upright=torch.ones(n, 1), nav_distance=zeros(), heading_error=zeros(),
        grasped=torch.ones(n, dtype=torch.bool), velocities=torch.zeros(n, 1, 6),
        flap_grasp=object(), unexpected_finger_force=torch.zeros(n, 4), obstacle_forces=torch.zeros(n, 2),
        supported=torch.zeros(n, 1, dtype=torch.bool), released=flags(), free_slots=torch.ones(n, 1, dtype=torch.bool),
        button_pressed=flags(), tools=torch.zeros(n, 2, 3), button_point=torch.zeros(n, 3),
        cargo_ok=torch.ones(n, 1, dtype=torch.bool), dwell=zeros(), success=flags(), failure=flags(),
        belt_running=flags(), belt_time=zeros(), cfg=SimpleNamespace(collision_force=200),
        robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros(n, 3))),
        hand_grasp_flags=torch.ones(n, 2, dtype=torch.bool), hand_target_distance=torch.zeros(n, 2),
        _measure=lambda: None, _goals=lambda: None)
    scene = type("Scene", (dict,), {})(robot_contact=SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.zeros(n, 1, 3))))
    scene.env_origins = torch.zeros(n, 3)
    t._env = SimpleNamespace(common_step_counter=1, scene=scene, step_dt=.1, episode_length_buf=torch.full((n,), 10))
    t.metrics = {name: zeros() for name in ("success", "boxes_placed", "phase", "cargo_retained", "grasp_left",
        "grasp_right", "lift_height", "hold_fraction", "left_target_distance", "right_target_distance")}
    commands.WorkcellCommand.refresh(t)
    commands.WorkcellCommand.refresh(t)
    torch.testing.assert_close(t.dwell, torch.full((n,), .1))
    t._env.common_step_counter += 1
    assert t.spec.obstacle_contact_force == .1
    t.obstacle_forces[0, 0] = .1  # the threshold itself is allowed; above it fails
    t.grasped[1] = False
    commands.WorkcellCommand.refresh(t)
    assert t.dwell[1] == 0
    t._env.common_step_counter += 1
    t.grasped[1] = True
    commands.WorkcellCommand.refresh(t)
    assert t.success.tolist() == [True, False]
    assert t.transition.tolist() == [True, False]
    # Even the first step has no collision grace. A contact below the old 180 N
    # threshold invalidates success and the stage bonus immediately.
    t._env.common_step_counter += 1
    t._env.episode_length_buf[:] = 1
    t.obstacle_forces[0, 0] = .11
    commands.WorkcellCommand.refresh(t)
    assert t.failure.tolist() == [True, False]
    assert not t.success.any() and not t.transition.any()
