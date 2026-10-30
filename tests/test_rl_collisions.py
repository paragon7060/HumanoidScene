"""Obstacle contact histories preserve substep impacts and ignore box support."""

from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.collisions import obstacle_forces


def test_filtered_history_detects_brief_impacts_without_force_cancellation():
    history = torch.zeros(2, 4, 1, 2, 3)
    # The most recent frame is clear; earlier opposing collider contacts remain.
    history[0, 2, 0, 0, 0] = 2.
    history[0, 2, 0, 1, 0] = -2.
    obstacle = SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=history,
        net_forces_w=torch.full((2, 1, 3), 100.)))
    # A robot/box support contact sensor is deliberately outside the obstacle set.
    support = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.full((2, 1, 3), 100.)))
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=4),
        scene=SimpleNamespace(sensors={"obstacle_contact_0": obstacle, "grasp_contact_0": support}))
    assert obstacle_forces(env).tolist() == [[2.], [0.]]
    history[0] = 0  # simulates a reset of only env 0
    history[1, 1, 0, 1, 2] = 3.
    assert obstacle_forces(env).tolist() == [[0.], [3.]]


@pytest.mark.parametrize("history", [None, torch.zeros(1, 4, 1, 0, 3), torch.zeros(1, 1, 1, 2, 3)])
def test_invalid_collision_sensing_fails_loudly(history):
    env = SimpleNamespace(cfg=SimpleNamespace(decimation=4), scene=SimpleNamespace(sensors={
        "obstacle_contact_0": SimpleNamespace(data=SimpleNamespace(force_matrix_w_history=history))}))
    with pytest.raises(RuntimeError, match="Missing obstacle"):
        obstacle_forces(env)
    env.scene.sensors.clear()
    with pytest.raises(RuntimeError, match="requires obstacle"):
        obstacle_forces(env)
