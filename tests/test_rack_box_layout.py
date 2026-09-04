"""Pure-Python checks for automatic and captured rack-box placement."""

from __future__ import annotations

import json

import pytest

from kuavo_isaaclab_scene.workcell import rack_box_layout as boxes
from kuavo_isaaclab_scene.workcell import workcell_layout as workcell


def test_physical_box_dimensions_and_spawn_scale() -> None:
    assert boxes.BOX_DIMENSIONS_M == {
        "small": (0.266, 0.185, 0.130),
        "medium": (0.320, 0.220, 0.185),
        "large": (0.380, 0.260, 0.230),
        "xlarge": (0.400, 0.320, 0.285),
    }
    assert boxes.BOX_FLAP_LENGTH_M == {
        "small": 0.100,
        "medium": 0.110,
        "large": 0.130,
        "xlarge": 0.155,
    }
    plan = boxes.build_box_spawn_plan({1: [], 2: [], 3: []}, 0.0)
    assert all(spec.scale == (1.0, 1.0, 1.0) for spec in plan.values())


def test_captured_pose_overrides_spawn_pose_in_rack_frame(tmp_path) -> None:
    capture_path = tmp_path / "rack_box_poses.json"
    capture_path.write_text(
        json.dumps(
            {
                "version": 1,
                "boxes": {
                    "SmallBox_0": {
                        "local_pos": [-0.42, -0.31, 1.12],
                        "local_rot": [0.92387953, 0.0, 0.0, 0.38268343],
                        "scale": [0.25, 0.22, 0.12],
                        "shelf": 2,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    empty_layout = {1: [], 2: [], 3: []}
    plan = boxes.build_box_spawn_plan(empty_layout, 0.0, capture_path)
    spec = plan["SmallBox_0"]

    assert spec.position == pytest.approx(
        workcell.local_point_to_world("rack", (-0.42, -0.31, 1.12))
    )
    assert spec.rotation == pytest.approx(
        workcell.local_quat_to_world(
            "rack", (0.92387953, 0.0, 0.0, 0.38268343)
        )
    )
    assert spec.scale == pytest.approx((0.25, 0.22, 0.12))
    assert spec.shelf == 2
    assert spec.row is None
    assert spec.column is None


def test_captured_pose_path_can_be_explicitly_ignored(tmp_path) -> None:
    capture_path = tmp_path / "poses.json"
    capture_path.write_text('{"boxes": {}}', encoding="utf-8")
    assert boxes.resolve_rack_box_pose_path(capture_path) == capture_path.resolve()
    assert boxes.resolve_rack_box_pose_path(capture_path, ignore=True) is None


def test_captured_pose_rejects_unknown_instance(tmp_path) -> None:
    capture_path = tmp_path / "poses.json"
    capture_path.write_text(
        json.dumps(
            {
                "boxes": {
                    "NotABox_0": {
                        "local_pos": [0.0, 0.0, 0.5],
                        "local_rot": [1.0, 0.0, 0.0, 0.0],
                        "scale": [1.0, 1.0, 1.0],
                        "shelf": 1,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown captured box instance"):
        boxes.load_captured_box_poses(capture_path)
