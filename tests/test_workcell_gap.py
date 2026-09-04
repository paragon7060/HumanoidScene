"""Exterior-clearance geometry and non-destructive layout editing checks."""

from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from kuavo_isaaclab_scene.workcell import workcell_layout as layout
from kuavo_isaaclab_scene.workcell.workcell_gap import adjusted_position, main, measure_gap


@pytest.fixture
def source(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    raw = json.loads((repository / "configs/workcell_layout.json").read_text())
    raw["conveyor"]["pos"] = [-1.611512, 1.158576, 0.0]
    raw["site_note"] = "preserve unrelated metadata"
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(raw, indent=2) + "\n")
    return path


def test_original_usd_exterior_gap_and_requested_adjustment(source):
    poses = layout.load_layout(source)
    gap, axis = measure_gap(poses)
    assert gap == pytest.approx(0.9091332674, abs=1e-6)
    assert axis == pytest.approx((1, 0, 0), abs=1e-6)
    target = adjusted_position(poses, 1.1)
    assert target == pytest.approx((-1.802378777, 1.158576, 0.0), abs=1e-8)
    adjusted = dict(poses, conveyor=replace(poses["conveyor"], pos=target))
    assert measure_gap(adjusted)[0] == pytest.approx(1.1, abs=1e-8)
    assert adjusted_position(adjusted, 1.1) == target
    assert poses["conveyor"].pos[0] == -1.611512


def test_defaults_are_110cm():
    repository = Path(__file__).resolve().parents[1]
    for path in (repository / "configs/workcell_layout.json",
                 repository / "src/kuavo_isaaclab_scene/configs/workcell_layout.json"):
        assert measure_gap(layout.load_layout(path))[0] == pytest.approx(1.1, abs=1e-6)


def test_conveyor_children_follow_anchor_translation(source, monkeypatch):
    poses = layout.load_layout(source)
    monkeypatch.setattr(layout, "LAYOUT", poses)
    old_surface = layout.offset("conveyor", (1.29, .43, .753))
    old_slot = layout.remap_point("conveyor", (.65, -.52, .775))
    target = adjusted_position(poses, 1.1)
    delta = tuple(a - b for a, b in zip(target, poses["conveyor"].pos))
    poses["conveyor"] = replace(poses["conveyor"], pos=target)
    for old, new in ((old_surface, layout.offset("conveyor", (1.29, .43, .753))),
                     (old_slot, layout.remap_point("conveyor", (.65, -.52, .775)))):
        assert tuple(a - b for a, b in zip(new, old)) == pytest.approx(delta)


def test_common_yaw_rotation_and_rack_scale(source):
    poses = layout.load_layout(source)
    q = (math.cos(.3), 0, 0, math.sin(.3))
    rotated = {name: replace(pose, pos=layout.quat_rotate(q, pose.pos),
                            rot=layout.quat_multiply(q, pose.rot)) for name, pose in poses.items()}
    assert measure_gap(rotated)[0] == pytest.approx(measure_gap(poses)[0])
    rotated["rack"] = replace(rotated["rack"], scale=(1.2, 1.1, 1))
    target = adjusted_position(rotated, 1.3)
    rotated["conveyor"] = replace(rotated["conveyor"], pos=target)
    assert measure_gap(rotated)[0] == pytest.approx(1.3, abs=1e-8)


@pytest.mark.parametrize("gap", [0, -.1, float("nan"), float("inf")])
def test_reject_invalid_gaps(source, gap):
    with pytest.raises(ValueError, match="finite positive"):
        adjusted_position(layout.load_layout(source), gap)


@pytest.mark.parametrize("changes, message", [
    ({"rot": (math.sqrt(.5), 0, 0, math.sqrt(.5))}, "parallel"),
    ({"rot": (math.sqrt(.5), math.sqrt(.5), 0, 0)}, "upright"),
    ({"pos": (-1.6, 10, 0)}, "overlap"),
    ({"pos": (10, 1.15, 0)}, "front side"),
    ({"pos": (float("nan"), 1, 0)}, "finite"),
    ({"scale": (2, 1, 1)}, "fixed asset scale"),
])
def test_reject_unsupported_layout(source, changes, message):
    poses = layout.load_layout(source)
    poses["conveyor"] = replace(poses["conveyor"], **changes)
    with pytest.raises(ValueError, match=message):
        measure_gap(poses)


def test_show_and_dry_run_do_not_write(source):
    original = source.read_bytes()
    assert main(["--layout", str(source)]) == 0
    assert main(["--layout", str(source), "--gap", "1.1", "--dry-run"]) == 0
    assert source.read_bytes() == original
    assert list(source.parent.iterdir()) == [source]


def test_edit_preserves_all_other_fields_with_backups_and_repeat(source):
    original_text = source.read_text()
    original = json.loads(original_text)
    assert main(["--layout", str(source), "--gap", "1.1"]) == 0
    updated = json.loads(source.read_text())
    target = updated["conveyor"]["pos"]
    assert measure_gap(layout.load_layout(source))[0] == pytest.approx(1.1, abs=1e-8)
    updated["conveyor"]["pos"] = original["conveyor"]["pos"]
    assert updated == original
    backups = list(source.parent.glob("layout.json.bak.*"))
    assert len(backups) == 1 and backups[0].read_text() == original_text
    assert main(["--layout", str(source), "--gap", "1.1"]) == 0
    assert len(list(source.parent.glob("layout.json.bak.*"))) == 1
    assert json.loads(source.read_text())["conveyor"]["pos"] == target
    assert main(["--layout", str(source), "--gap", "1.2"]) == 0
    assert len(list(source.parent.glob("layout.json.bak.*"))) == 2


def test_output_leaves_source_and_existing_destination_untouched(source):
    original = source.read_bytes()
    output = source.parent / "experiment" / "new.json"
    args = ["--layout", str(source), "--gap", "1.1", "--output", str(output)]
    assert main(args) == 0
    written = output.read_bytes()
    assert source.read_bytes() == original
    with pytest.raises(SystemExit):
        main(args)
    assert output.read_bytes() == written
    assert not list(source.parent.glob("*.bak.*"))
