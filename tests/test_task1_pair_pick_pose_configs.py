"""Contract checks for shelf-2 adjacent-flap pair-pick pose configs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from kuavo_isaaclab_scene.workcell import rack_box_layout as rack_boxes


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
BOX_WIDTH_M = {
    "SmallBox": 0.266,
    "MediumBox": 0.320,
}
EXPECTED_CONFIGS = {
    "rack_box_poses_task1_ssm_left.json": {
        "box_order": ["SmallBox_0", "SmallBox_1", "MediumBox_0"],
        "paired_boxes": ["SmallBox_0", "SmallBox_1"],
        "active_arm": "left",
        "local_x": [-0.207, -0.473, -0.766],
    },
    "rack_box_poses_task1_mss_right.json": {
        "box_order": ["MediumBox_0", "SmallBox_0", "SmallBox_1"],
        "paired_boxes": ["SmallBox_0", "SmallBox_1"],
        "active_arm": "right",
        "local_x": [-0.234, -0.527, -0.793],
    },
    "rack_box_poses_task1_mms_left.json": {
        "box_order": ["MediumBox_0", "MediumBox_1", "SmallBox_0"],
        "paired_boxes": ["MediumBox_0", "MediumBox_1"],
        "active_arm": "left",
        "local_x": [-0.207, -0.527, -0.820],
    },
    "rack_box_poses_task1_smm_right.json": {
        "box_order": ["SmallBox_0", "MediumBox_0", "MediumBox_1"],
        "paired_boxes": ["MediumBox_0", "MediumBox_1"],
        "active_arm": "right",
        "local_x": [-0.180, -0.473, -0.793],
    },
}


def _box_type(instance_name: str) -> str:
    return instance_name.rsplit("_", 1)[0]


@pytest.mark.parametrize(("file_name", "expected"), EXPECTED_CONFIGS.items())
def test_pair_pick_pose_config_is_directly_loadable_and_matches_approved_layout(
    file_name: str, expected: dict[str, object]
) -> None:
    path = CONFIG_DIR / file_name
    raw = json.loads(path.read_text(encoding="utf-8"))
    poses = rack_boxes.load_captured_box_poses(path)
    pair_pick = raw["pair_pick"]

    assert raw["version"] == 1
    assert pair_pick == {
        "order": "robot_view_left_to_right",
        "box_order": expected["box_order"],
        "paired_boxes": expected["paired_boxes"],
        "active_arm": expected["active_arm"],
        "grasp": "adjacent_inner_flaps",
        "pair_gap_m": 0.0,
    }
    assert set(poses) == set(expected["box_order"])
    assert all(pose.shelf == 2 for pose in poses.values())
    assert [poses[name].local_pos[0] for name in expected["box_order"]] == pytest.approx(
        expected["local_x"], abs=1.0e-9
    )


@pytest.mark.parametrize(("file_name", "expected"), EXPECTED_CONFIGS.items())
def test_pair_pick_pose_config_centers_three_touching_boxes_inside_shelf_width(
    file_name: str, expected: dict[str, object]
) -> None:
    path = CONFIG_DIR / file_name
    raw = json.loads(path.read_text(encoding="utf-8"))
    box_order = expected["box_order"]
    pair_pick = raw["pair_pick"]
    positions = {
        name: raw["boxes"][name]["local_pos"][0]
        for name in box_order
    }
    widths = {name: BOX_WIDTH_M[_box_type(name)] for name in box_order}

    gaps = [
        (positions[left] - widths[left] / 2.0)
        - (positions[right] + widths[right] / 2.0)
        for left, right in zip(box_order, box_order[1:])
    ]
    high_edge = positions[box_order[0]] + widths[box_order[0]] / 2.0
    low_edge = positions[box_order[-1]] - widths[box_order[-1]] / 2.0
    pair_indices = [box_order.index(name) for name in pair_pick["paired_boxes"]]

    assert gaps == pytest.approx([0.0, 0.0], abs=1.0e-9)
    assert high_edge - low_edge <= 0.92
    assert (high_edge + low_edge) / 2.0 == pytest.approx(-0.5, abs=1.0e-9)
    assert pair_indices in ([0, 1], [1, 2])
    assert pair_pick["active_arm"] == ("left" if pair_indices == [0, 1] else "right")
    assert len({_box_type(name) for name in pair_pick["paired_boxes"]}) == 1
    assert Counter(_box_type(name) for name in box_order).most_common()[0][1] == 2
