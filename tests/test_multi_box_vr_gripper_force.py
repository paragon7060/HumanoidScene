from types import SimpleNamespace

from kuavo_isaaclab_scene.robots.claw_assets.vr import (
    _incremental_delta_scale,
    _is_contact_box_asset,
)


def test_force_gripper_recognizes_legacy_and_multi_box_v2_assets():
    assert _is_contact_box_asset("small_box_0")
    assert _is_contact_box_asset("mb_s2_small_0")
    assert _is_contact_box_asset("mb_s2_medium_5")
    assert not _is_contact_box_asset("staging_boxes_group")
    assert not _is_contact_box_asset("conveyor_surface")


def test_reward_debug_restores_incremental_scale_from_binary_config():
    assert _incremental_delta_scale(SimpleNamespace(delta_scale=0.0)) == 0.12
    assert _incremental_delta_scale(SimpleNamespace(delta_scale=-1.0)) == 0.12
    assert _incremental_delta_scale(SimpleNamespace(delta_scale=float("nan"))) == 0.12
    assert _incremental_delta_scale(SimpleNamespace(delta_scale=0.07)) == 0.07
    assert _incremental_delta_scale(SimpleNamespace()) == 0.12
