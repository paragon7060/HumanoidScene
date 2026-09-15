from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.rl.debug.contact_rate import (
    configure_obstacle_contact_rate,
    configure_realtime_reward_debug,
)


def test_only_obstacle_contact_reports_are_decimated():
    obstacle_0 = SimpleNamespace(update_period=0.0, history_length=4)
    obstacle_1 = SimpleNamespace(update_period=0.0, history_length=4)
    grasp = SimpleNamespace(update_period=0.0, history_length=1)
    scene = SimpleNamespace(obstacle_contact_0=obstacle_0,
                            obstacle_contact_1=obstacle_1,
                            grasp_contact_0=grasp,
                            obstacle_contact_disabled=None)

    assert configure_obstacle_contact_rate(scene, 30) == 2
    assert obstacle_0.update_period == pytest.approx(1 / 30)
    assert obstacle_1.update_period == pytest.approx(1 / 30)
    assert obstacle_0.history_length == obstacle_1.history_length == 4
    assert grasp.update_period == 0.0


def test_obstacle_contact_rate_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        configure_obstacle_contact_rate(SimpleNamespace(), 0)


def test_realtime_profile_removes_policy_and_filtered_contacts():
    robot_props = SimpleNamespace(solver_position_iteration_count=32,
                                  solver_velocity_iteration_count=8)
    box_props = SimpleNamespace(solver_position_iteration_count=32,
                                solver_velocity_iteration_count=8)
    scene = SimpleNamespace(
        robot=SimpleNamespace(spawn=SimpleNamespace(articulation_props=robot_props)),
        box=SimpleNamespace(spawn=SimpleNamespace(articulation_props=box_props)),
        obstacle_contact_0=SimpleNamespace(),
        grasp_contact_0=SimpleNamespace(),
    )
    cfg = SimpleNamespace(
        sim=SimpleNamespace(dt=1 / 120, render_interval=4),
        decimation=4,
        observations=SimpleNamespace(policy=object()),
        recorders=SimpleNamespace(success_states=object()),
        commands=SimpleNamespace(workcell=SimpleNamespace(
            collision_reporting="filtered", post_step_measurement=True)),
        scene=scene,
        task=SimpleNamespace(box_names=("box",)),
    )

    assert configure_realtime_reward_debug(cfg) == 1
    assert cfg.sim.dt == pytest.approx(1 / 30)
    assert cfg.decimation == cfg.sim.render_interval == 1
    assert cfg.observations.policy is None
    assert cfg.recorders.success_states is None
    assert cfg.commands.workcell.collision_reporting == "aggregate"
    assert not cfg.commands.workcell.post_step_measurement
    assert scene.obstacle_contact_0 is None
    assert scene.grasp_contact_0 is not None
    assert (robot_props.solver_position_iteration_count,
            robot_props.solver_velocity_iteration_count) == (8, 2)
    assert (box_props.solver_position_iteration_count,
            box_props.solver_velocity_iteration_count) == (8, 2)
