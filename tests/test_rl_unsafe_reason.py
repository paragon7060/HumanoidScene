"""The Quest v2 demonstration console must name the exact unsafe predicate."""

from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.debug.unsafe_reason import (
    BoxSafetyValues,
    safety_measurements,
    unsafe_causes,
)


def _safety(**overrides):
    values = {
        "robot_rack_collision": False,
        "self_collision": False,
        "obstacle_collision": False,
        "workspace_limit": False,
        "box_drop": False,
        "box_lift_limit": False,
        "box_speed_limit": False,
        "rack_force_n": 0.0,
        "obstacle_force_n": 0.0,
        "base_distance_m": 0.0,
        "self_collision_distance_m": 0.012,
    }
    values.update(overrides)
    # Matches the pre-reset `transition_safety` mapping the environment publishes.
    return {name: torch.tensor([value]) for name, value in values.items()}


def _cfg(self_collision_enabled=False):
    return SimpleNamespace(
        task=SimpleNamespace(obstacle_contact_force=5.0),
        multi_box=SimpleNamespace(
            rack_contact_force=10.0,
            workspace_radius=1.5,
            self_collision_clearance=0.003,
            self_collision_enabled=self_collision_enabled,
            max_box_lift_height=0.5,
            max_box_linear_speed=10.0,
            max_box_angular_speed=100.0,
        ),
    )


def test_every_triggered_predicate_is_reported_in_one_stable_order():
    safety = _safety(robot_rack_collision=True, obstacle_collision=True)
    assert unsafe_causes(safety) == ("robot_rack_collision", "obstacle_collision")
    assert unsafe_causes(_safety()) == ()


def test_an_incomplete_snapshot_is_rejected_instead_of_reporting_no_cause():
    incomplete = _safety()
    del incomplete["box_drop"]
    with pytest.raises(KeyError):
        unsafe_causes(incomplete)


def test_measurements_pair_each_value_with_its_configured_limit():
    safety = _safety(robot_rack_collision=True, rack_force_n=24.3, base_distance_m=0.22)
    text = safety_measurements(safety, _cfg())
    assert "rack 24.3/10.0 N" in text
    assert "obstacle 0.0/5.0 N" in text
    assert "base 0.22/1.50 m" in text
    # The demonstration run may disable self-collision; do not print a clearance
    # that the termination manager never evaluated.
    assert "self-collision off" in text
    assert "box" not in text


def test_enabled_self_collision_and_box_values_are_reported():
    safety = _safety(self_collision=True, self_collision_distance_m=0.0004)
    text = safety_measurements(
        safety, _cfg(self_collision_enabled=True),
        BoxSafetyValues(height_m=1.05, lift_m=0.03, linear_speed=0.2, angular_speed=1.5))
    assert "self clearance 0.04/0.30 cm" in text
    assert "box z 1.05 m" in text
    assert "lift 0.03/0.50 m" in text
    assert "box speed 0.20/10.0 m/s, 1.50/100.0 rad/s" in text
