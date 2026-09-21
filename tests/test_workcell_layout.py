"""Pure-Python checks for 6-DoF workcell layout transforms."""

from __future__ import annotations

import json
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


def test_current_rack_base_matches_webpage_frame_definition() -> None:
    assert layout.position("rack_base") == pytest.approx(
        (0.4245231106, 0.0212500007, 0.0), abs=1.0e-9
    )
    assert layout.rotation("rack_base") == pytest.approx(
        (1.0, 0.0, 0.0, 1.4142135685e-08), abs=1.0e-9
    )
    assert layout.position("rack") == pytest.approx((0.4, 0.44, 0.0), abs=1.0e-8)

    robot_to_rack = layout.robot_to_rack_base_pose()
    assert robot_to_rack.pos == pytest.approx(
        (0.5245231106, 0.0212500007, 0.0), abs=1.0e-9
    )


def test_rack_asset_and_base_pose_round_trip() -> None:
    half_angle = math.radians(17.0)
    base = layout.AnchorPose(
        (1.2, -0.4, 0.03),
        (math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)),
    )
    rack_scale = (0.97, 1.02, 1.0)
    asset = layout.rack_asset_pose_from_base(base, rack_scale)
    recovered = layout.rack_base_pose_from_asset(asset)
    assert recovered.pos == pytest.approx(base.pos, abs=1.0e-9)
    assert recovered.rot == pytest.approx(base.rot, abs=1.0e-9)
    assert recovered.scale == (1.0, 1.0, 1.0)


def test_rack_base_is_authoritative_when_present(tmp_path) -> None:
    source = layout.DEFAULT_LAYOUT_PATH
    raw = json.loads(source.read_text(encoding="utf-8"))
    raw["rack_base"]["pos"] = [1.0, 2.0, 0.0]
    raw["rack"]["pos"] = [99.0, 99.0, 99.0]
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = layout.load_layout(path)
    expected_asset = layout.rack_asset_pose_from_base(
        loaded["rack_base"],
        loaded["rack"].scale,
    )
    assert loaded["rack_base"].pos == (1.0, 2.0, 0.0)
    assert loaded["rack"].pos == pytest.approx(expected_asset.pos, abs=1.0e-9)


def test_legacy_layout_without_rack_base_is_still_supported(tmp_path) -> None:
    raw = json.loads(layout.DEFAULT_LAYOUT_PATH.read_text(encoding="utf-8"))
    del raw["rack_base"]
    path = tmp_path / "legacy_layout.json"
    path.write_text(json.dumps(raw), encoding="utf-8")

    loaded = layout.load_layout(path)
    expected_base = layout.rack_base_pose_from_asset(loaded["rack"])
    assert loaded["rack_base"].pos == pytest.approx(expected_base.pos, abs=1.0e-9)
    assert loaded["rack_base"].rot == pytest.approx(expected_base.rot, abs=1.0e-9)
