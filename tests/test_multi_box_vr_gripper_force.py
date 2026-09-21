from kuavo_isaaclab_scene.robots.claw_assets.vr import (
    _is_contact_box_asset,
)


def test_force_gripper_recognizes_legacy_and_multi_box_v2_assets():
    assert _is_contact_box_asset("small_box_0")
    assert _is_contact_box_asset("mb_s2_small_0")
    assert _is_contact_box_asset("mb_s2_medium_5")
    assert not _is_contact_box_asset("staging_boxes_group")
    assert not _is_contact_box_asset("conveyor_surface")
