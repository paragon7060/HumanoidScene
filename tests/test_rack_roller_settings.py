"""Lightweight defaults for the shared rack model; no Isaac app required."""

from kuavo_isaaclab_scene.workcell.rack_rollers import resolve_rack_roller_settings


def test_rack_rollers_are_enabled_by_default_and_can_be_disabled(monkeypatch):
    monkeypatch.delenv("KUAVO_RACK_ROLLERS", raising=False)
    assert resolve_rack_roller_settings().enabled
    assert not resolve_rack_roller_settings(enabled=False).enabled
