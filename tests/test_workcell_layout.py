"""Pure-Python checks for 6-DoF workcell layout transforms."""

from __future__ import annotations

import math

import pytest

from kuavo_isaaclab_scene.workcell import workcell_layout as layout


def test_anchor_origin_maps_to_captured_position() -> None:
    for name, default_pose in layout.DEFAULT_ANCHORS.items():
        assert layout.remap_point(name, default_pose.pos) == pytest.approx(
            layout.position(name), abs=1.0e-7
        )


def test_default_orientation_maps_to_captured_orientation() -> None:
    for name, default_pose in layout.DEFAULT_ANCHORS.items():
        assert layout.remap_quat(name, default_pose.rot) == pytest.approx(
            layout.rotation(name), abs=1.0e-7
        )


def test_quaternion_rotates_vector_about_z() -> None:
    half_angle = math.radians(45.0)
    yaw_90 = (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    assert layout.quat_rotate(yaw_90, (1.0, 0.0, 0.0)) == pytest.approx(
        (0.0, 1.0, 0.0), abs=1.0e-7
    )


def test_conveyor_remap_preserves_anchor_relative_distance() -> None:
    default_slot = (0.65, -0.52, 0.775)
    mapped_slot = layout.remap_point("conveyor", default_slot)
    default_anchor = layout.DEFAULT_ANCHORS["conveyor"].pos
    captured_anchor = layout.position("conveyor")
    default_distance = math.dist(default_slot, default_anchor)
    mapped_distance = math.dist(mapped_slot, captured_anchor)
    assert mapped_distance == pytest.approx(default_distance, abs=1.0e-7)
