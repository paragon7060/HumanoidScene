"""Exercise real reward/phase code with CPU state, without starting Isaac Sim."""

import importlib.util
from dataclasses import replace
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.tasks.specs import task_spec
from kuavo_isaaclab_scene.rl.mdp.reach_progress import ReachProgress
from kuavo_isaaclab_scene.rl.mdp.flap_progress import FlapProgress


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
    t.reach_progress = SimpleNamespace(delta=torch.tensor([[.3, .1]]))
    t.flap_progress = SimpleNamespace(grasp_bonus=torch.zeros(1))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t), step_dt=1/30)
    reach = rewards.flap_reaching(env)
    contact = rewards.flap_contact(env)
    t.hand_target_distance[:, 0] = 100
    t.grasp_alignment[:, 0] = 0
    t.finger_grasp_contacts[:, 0] = False
    t.hand_grasp_flags[:, 0] = False
    torch.testing.assert_close(rewards.flap_reaching(env), reach)
    torch.testing.assert_close(rewards.flap_contact(env), contact)
    # Alignment no longer reverses the monotonic distance-only reward.
    t.grasp_alignment[:, 1] = 0
    torch.testing.assert_close(rewards.flap_reaching(env), reach)
    t.hand_target_distance[:, 1] = .05
    # Rewards read cached progress; another distance read must not accrue it twice.
    torch.testing.assert_close(rewards.flap_reaching(env), reach)
    t.reach_progress.delta[:, 1] = .2
    assert rewards.flap_reaching(env) > reach
    assert contact.item() == 0
    t.finger_grasp_contacts[:, 1] = True
    t.hand_grasp_flags[:, 1] = True
    t.flap_progress.grasp_bonus[:] = 1
    assert (rewards.flap_contact(env) * env.step_dt).item() == 1
    t.settling = SimpleNamespace(ready=torch.tensor([False]))
    assert rewards.flap_reaching(env).item() == 0
    assert rewards.flap_contact(env).item() == 0
    t.settling = None
    t.reach_progress.delta[:, 1] = -.2
    assert (rewards.flap_reaching(env) * env.step_dt * 4).item() == pytest.approx(-.8)


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
    t.flap_progress = SimpleNamespace(lift_delta=torch.tensor([.5, -.2]),
                                     grasp_bonus=torch.tensor([1., 0.]))
    for dt in (1/30, 1/60):
        env.step_dt = dt
        for name in ("stage_completed", "success", "failure"):
            expected = t.transition if name == "stage_completed" else getattr(t, name)
            torch.testing.assert_close(getattr(rewards, name)(env) * dt, expected.float())
        torch.testing.assert_close(rewards.flap_lift_progress(env) * dt, t.flap_progress.lift_delta)
        torch.testing.assert_close(rewards.flap_contact(env) * dt, t.flap_progress.grasp_bonus)


def test_orientation_is_nearby_pregrasp_right_only_and_additive(modules):
    _, rewards = modules
    t = SimpleNamespace(spec=task_spec("pick", grasp_hand="right"), settling=None,
        hand_target_distance=torch.tensor([[0., .05]]), grasp_alignment=torch.tensor([[0., 1.]]),
        hand_grasp_flags=torch.zeros(1, 2, dtype=torch.bool), reward_phase=torch.ones(1, dtype=torch.long),
        refresh=lambda: None)
    t.reach_progress = SimpleNamespace(delta=torch.zeros(1, 2))
    t.flap_progress = SimpleNamespace(orientation_delta=torch.tensor([[0., 1.]]),
                                     orientation_distance=torch.tensor([[0., .05]]))
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t), step_dt=1/30)
    assert (rewards.flap_orientation(env) * env.step_dt).item() == pytest.approx(.5)
    reaching = rewards.flap_reaching(env).clone()
    t.hand_target_distance[:, 0] = 100
    t.grasp_alignment[:, 0] = 1
    assert (rewards.flap_orientation(env) * env.step_dt).item() == pytest.approx(.5)
    t.flap_progress.orientation_delta[:, 1] = -.75
    assert (rewards.flap_orientation(env) * env.step_dt).item() == pytest.approx(-.375)
    torch.testing.assert_close(rewards.flap_reaching(env), reaching)
    t.flap_progress.orientation_distance[:, 1] = .10
    assert rewards.flap_orientation(env).item() == 0
    t.flap_progress.orientation_distance[:, 1] = 0
    t.flap_progress.orientation_delta[:, 1] = 0  # tracker masks grasp transitions
    assert rewards.flap_orientation(env).item() == 0
    t.hand_grasp_flags[:, 1] = False
    t.settling = SimpleNamespace(ready=torch.tensor([False]))
    assert rewards.flap_orientation(env).item() == 0
    with pytest.raises(ValueError, match="distance_threshold"):
        rewards.flap_orientation(env, distance_threshold=0)


@pytest.mark.parametrize("collision_enabled", [True, False])
def test_pick_dwell_resets_on_lost_grasp_and_refresh_is_once_per_step(modules, collision_enabled):
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
        flap_grasp=object(), reach_progress=ReachProgress(n, "cpu"), flap_progress=FlapProgress(n, "cpu"),
        grasp_alignment=torch.ones(n, 2), nearest_flap_index=torch.zeros(n, 2, dtype=torch.long),
        unexpected_finger_force=torch.zeros(n, 4), obstacle_forces=torch.zeros(n, 2),
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
    t.spec = replace(t.spec, collision_constraints_enabled=collision_enabled)
    # Pick success no longer depends on box speed or residual finger force.
    t.velocities[:] = 10.
    t.unexpected_finger_force[:] = 15.
    commands.WorkcellCommand.refresh(t)
    commands.WorkcellCommand.refresh(t)
    torch.testing.assert_close(t.dwell, torch.full((n,), .1))
    t._env.common_step_counter += 1
    assert t.spec.obstacle_contact_force == 20.
    t.obstacle_forces[0, 0] = 20.  # the threshold itself is allowed; above it fails
    assert "speed" not in t.pick_checks and "contact" not in t.pick_checks
    assert "hold" in t.pick_checks
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
    t.obstacle_forces[0, 0] = 20.01
    commands.WorkcellCommand.refresh(t)
    assert t.failure.tolist() == [collision_enabled, False]
    assert ("obstacle_collision" in t.failure_checks) == collision_enabled
    if collision_enabled:
        assert not t.success.any() and not t.transition.any()
    else:
        assert t.success[0]  # collision cannot veto a valid pick
        t._env.common_step_counter += 1
        t.centers[1, 0, 2] = .1
        commands.WorkcellCommand.refresh(t)
        assert t.failure.tolist() == [False, True]  # real box-drop failure remains


def test_disabled_collision_cost_ignores_obstacle_and_residual_finger_forces(modules):
    _, rewards = modules
    t = SimpleNamespace(spec=task_spec("pick", collision_constraints_enabled=False), settling=None,
        obstacle_forces=torch.full((2, 2), 500.), unexpected_finger_force=torch.full((2, 4), 100.),
        refresh=lambda: None)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t))
    assert rewards.unwanted_contact(env).eq(0).all()
    t.spec = replace(t.spec, collision_constraints_enabled=True)
    assert rewards.unwanted_contact(env).gt(0).all()


def test_static_grasp_below_goal_has_no_positive_shaping_after_acquisition(modules):
    _, rewards = modules
    p = FlapProgress(1, "cpu")
    flags = torch.ones(1, dtype=torch.bool)
    target = torch.zeros(1, dtype=torch.long)
    hands = torch.ones(1, 2, dtype=torch.bool)
    t = SimpleNamespace(spec=task_spec("pick"), settling=None, flap_progress=p,
                        reach_progress=SimpleNamespace(delta=torch.zeros(1, 2)),
                        dwell=torch.zeros(1), refresh=lambda: None)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t), step_dt=1/30)
    def advance(held, height):
        p.advance(torch.ones(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long),
                  hands & held, flags & held, torch.full((1,), height), target, flags, flags)
    advance(False, 0.)
    advance(True, 0.)
    assert (rewards.flap_contact(env) * env.step_dt * 3).item() == 3.
    for height in (0., .99):
        advance(True, height)
        for _ in range(3):
            advance(True, height)
            for function in (rewards.flap_reaching, rewards.flap_orientation,
                             rewards.flap_contact, rewards.flap_lift_progress, rewards.flap_hold):
                assert function(env).item() == 0


def test_flap_observation_fixed_wait_is_finite_with_timeout_disabled(modules):
    path = Path(__file__).resolve().parents[1] / "src/kuavo_isaaclab_scene/rl/mdp/observations.py"
    spec = importlib.util.spec_from_file_location("kuavo_isaaclab_scene.rl.mdp.observations", path)
    obs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obs)
    t = SimpleNamespace(spec=task_spec("pick", reset_settle_seconds=.5, reset_settle_timeout=0.),
        obstacle_forces=torch.zeros(1, 2), centers=torch.tensor([[[0., 0., .5]]]), ids=torch.tensor([0]),
        active_box=torch.tensor([0]), initial_z=torch.tensor([[.5]]), contact_force=torch.zeros(1, 2),
        hand_grasp_flags=torch.zeros(1, 2, dtype=torch.bool), raw_hand_grasp_flags=torch.zeros(1, 2, dtype=torch.bool),
        grasp_contact_missing_s=torch.zeros(1, 2), contact_flap_index=torch.zeros(1, 2, dtype=torch.long),
        unexpected_finger_force=torch.zeros(1, 4), half_size=torch.ones(1, 3), dwell=torch.zeros(1),
        settling=SimpleNamespace(ready=torch.tensor([False]), elapsed=torch.tensor([0.])), refresh=lambda: None)
    env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: t),
                          scene=SimpleNamespace(env_origins=torch.zeros(1, 3)))
    assert torch.isfinite(obs.flap_pick_state(env)).all()
    t.settling.ready[:] = True
    t.settling.elapsed[:] = .5
    result = obs.flap_pick_state(env)
    assert torch.isfinite(result).all()
    assert result[0, -2:].tolist() == [1., 1.]


@pytest.mark.parametrize("relative", ["configs", "src/kuavo_isaaclab_scene/configs"])
def test_pick_presets_disable_collision_constraints(relative):
    path = Path(__file__).resolve().parents[1] / relative / "rl_pick_arms_only.py"
    spec = importlib.util.spec_from_file_location("pick_preset", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configured = module.configure_task(task_spec("pick"))
    configured.validate()
    assert not configured.collision_constraints_enabled
