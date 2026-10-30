from types import SimpleNamespace as NS
import json
from pathlib import Path

import pytest

from kuavo_isaaclab_scene.evaluation.box_filter import configure_rack_boxes_only


def make_cfg():
    return NS(scene=NS(small_box_0=object(), medium_box_0=object(), large_box_0=object(),
                       totes=object(), cargo=object(), robot=object()),
              observations=NS(policy=NS(joint_pos=object())),
              rewards=NS(progress=object(), success=object()),
              terminations=NS(success=object(), tote_drop=object()),
              events=NS(reset_flap_friction=NS(params={"asset_names": ("small_box_0",)})))


def test_only_selected_box_is_spawned_without_legacy_references():
    cfg = make_cfg()
    robot, box = cfg.scene.robot, cfg.scene.medium_box_0
    progress, success = cfg.rewards.progress, cfg.terminations.success
    meta = configure_rack_boxes_only(cfg, ("medium_box_0",),
                                    ("small_box_0", "medium_box_0", "large_box_0"))
    assert cfg.scene.robot is robot and cfg.scene.medium_box_0 is box
    assert cfg.scene.small_box_0 is cfg.scene.large_box_0 is cfg.scene.totes is cfg.scene.cargo is None
    assert cfg.observations.policy.tote_poses is cfg.observations.policy.cargo_state is None
    assert cfg.rewards.cargo_retention is cfg.rewards.tote_stability is None
    assert cfg.terminations.cargo_spill is cfg.events.reset_workcell is None
    assert cfg.rewards.progress is progress and cfg.terminations.success is success
    assert cfg.events.reset_flap_friction.params["asset_names"] == ("medium_box_0",)
    assert meta["spawned_box_scene_keys"] == ["medium_box_0"]


@pytest.mark.parametrize("active", [(), ("unknown",)])
def test_invalid_selection_fails_before_mutating_scene(active):
    cfg = make_cfg()
    box = cfg.scene.small_box_0
    with pytest.raises(ValueError):
        configure_rack_boxes_only(cfg, active, ("small_box_0", "medium_box_0"))
    assert cfg.scene.small_box_0 is box


def test_medium_only_pose_is_on_middle_shelf():
    path = Path(__file__).resolve().parents[1] / "configs/rack_box_poses_medium_only.json"
    boxes = json.loads(path.read_text())["boxes"]
    assert list(boxes) == ["MediumBox_0"]
    assert boxes["MediumBox_0"]["shelf"] == 2
