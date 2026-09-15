from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.rl.debug.contact_rate import configure_obstacle_contact_rate


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
