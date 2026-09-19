"""Base-model selection shared by teleop, RL and evaluation entry points."""

import argparse
from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.robots.base_drive import (
    DYNAMIC_BASE_ENV,
    add_base_drive_cli_args,
    apply_base_drive,
    export_base_drive_cli,
    resolve_base_drive_settings,
)


def robot_cfg():
    return SimpleNamespace(spawn=SimpleNamespace(
        articulation_props=SimpleNamespace(fix_root_link=True)))


def test_default_is_the_existing_kinematic_base(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    settings = resolve_base_drive_settings()
    assert settings.dynamic is False
    assert settings.fix_root_link is True


def test_environment_selects_the_dynamic_base(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    assert resolve_base_drive_settings().dynamic is True
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "0")
    assert resolve_base_drive_settings().dynamic is False
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "yes")
    with pytest.raises(ValueError):
        resolve_base_drive_settings()


def test_cli_round_trip(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    parser = argparse.ArgumentParser()
    add_base_drive_cli_args(parser)
    export_base_drive_cli(parser.parse_args([]))
    assert resolve_base_drive_settings().dynamic is False
    export_base_drive_cli(parser.parse_args(["--dynamic-base"]))
    assert resolve_base_drive_settings().dynamic is True
    export_base_drive_cli(parser.parse_args(["--no-dynamic-base"]))
    assert resolve_base_drive_settings().dynamic is False


def test_apply_leaves_the_kinematic_base_untouched(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    robot, actions = robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False))
    assert apply_base_drive(robot, actions, "base") is False
    assert robot.spawn.articulation_props.fix_root_link is True
    assert actions.base.dynamic is False


def test_apply_frees_the_root_and_enables_the_action(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    robot, actions = robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False))
    assert apply_base_drive(robot, actions, "base") is True
    assert robot.spawn.articulation_props.fix_root_link is False
    assert actions.base.dynamic is True


def test_apply_reports_configurations_without_a_base_action(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    actions = SimpleNamespace(base=None)
    with pytest.raises(ValueError, match="locks the base"):
        apply_base_drive(robot_cfg(), actions, "base")


def test_apply_requires_articulation_properties(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    robot = SimpleNamespace(spawn=SimpleNamespace(articulation_props=None))
    with pytest.raises(ValueError, match="articulation properties"):
        apply_base_drive(robot, SimpleNamespace(base=SimpleNamespace(dynamic=False)), "base")
