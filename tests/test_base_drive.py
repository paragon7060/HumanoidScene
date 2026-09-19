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


def test_dynamic_base_is_the_default(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    settings = resolve_base_drive_settings()
    assert settings.dynamic is True
    assert settings.fix_root_link is False
    assert settings.explicit is False


def test_environment_selects_the_base_model(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    assert resolve_base_drive_settings() == type(resolve_base_drive_settings())(True, True)
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "0")
    assert resolve_base_drive_settings().dynamic is False
    assert resolve_base_drive_settings().explicit is True
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "yes")
    with pytest.raises(ValueError):
        resolve_base_drive_settings()


def test_cli_round_trip(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    parser = argparse.ArgumentParser()
    add_base_drive_cli_args(parser)
    export_base_drive_cli(parser.parse_args([]))
    assert resolve_base_drive_settings().dynamic is True
    export_base_drive_cli(parser.parse_args(["--dynamic-base"]))
    assert resolve_base_drive_settings().dynamic is True
    export_base_drive_cli(parser.parse_args(["--no-dynamic-base"]))
    assert resolve_base_drive_settings().dynamic is False


def test_apply_leaves_the_kinematic_base_untouched(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "0")
    robot, actions = robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False))
    assert apply_base_drive(robot, actions, "base") is False
    assert robot.spawn.articulation_props.fix_root_link is True
    assert actions.base.dynamic is False


def test_apply_frees_the_root_and_enables_the_action(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    robot, actions = robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False))
    assert apply_base_drive(robot, actions, "base") is True
    assert robot.spawn.articulation_props.fix_root_link is False
    assert actions.base.dynamic is True


def test_a_locked_base_keeps_its_fixed_root_by_default(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    robot = robot_cfg()
    assert apply_base_drive(robot, SimpleNamespace(base=None), "base") is False
    assert robot.spawn.articulation_props.fix_root_link is True


def test_apply_reports_an_explicit_request_a_locked_base_cannot_serve(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    with pytest.raises(ValueError, match="locks the base"):
        apply_base_drive(robot_cfg(), SimpleNamespace(base=None), "base")


def test_a_model_without_wheels_keeps_its_fixed_root(monkeypatch):
    monkeypatch.delenv(DYNAMIC_BASE_ENV, raising=False)
    monkeypatch.setenv("KUAVO_ROBOT_MODEL", "s56")
    robot, actions = robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False))
    assert apply_base_drive(robot, actions, "base") is False
    assert robot.spawn.articulation_props.fix_root_link is True


def test_apply_reports_an_explicit_request_on_a_model_without_wheels(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    monkeypatch.setenv("KUAVO_ROBOT_MODEL", "s56")
    with pytest.raises(ValueError, match="wheeled chassis"):
        apply_base_drive(robot_cfg(), SimpleNamespace(base=SimpleNamespace(dynamic=False)), "base")


def test_apply_requires_articulation_properties(monkeypatch):
    monkeypatch.setenv(DYNAMIC_BASE_ENV, "1")
    robot = SimpleNamespace(spawn=SimpleNamespace(articulation_props=None))
    with pytest.raises(ValueError, match="articulation properties"):
        apply_base_drive(robot, SimpleNamespace(base=SimpleNamespace(dynamic=False)), "base")
