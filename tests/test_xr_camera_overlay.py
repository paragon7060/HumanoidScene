import numpy as np
import pytest

from kuavo_isaaclab_scene.xr_camera_overlay import QuestCameraOverlayCfg, as_rgba


def test_as_rgba_adds_opaque_alpha_and_preserves_pixels():
    rgb = np.zeros((4, 6, 3), dtype=np.uint8)
    rgb[..., 1] = 127
    rgba = as_rgba(rgb)
    assert rgba.shape == (4, 6, 4)
    assert rgba.flags.c_contiguous
    np.testing.assert_array_equal(rgba[..., :3], rgb)
    np.testing.assert_array_equal(rgba[..., 3], 255)


def test_overlay_config_rejects_plane_inside_near_clip():
    with pytest.raises(ValueError, match="near plane"):
        QuestCameraOverlayCfg(distance_m=0.05)


def test_overlay_defaults_are_compact_and_leave_room_for_three_panels():
    cfg = QuestCameraOverlayCfg()
    assert cfg.distance_m > 0.08
    assert cfg.plane_width_m <= 0.2
    assert cfg.plane_height_m <= 0.15
    assert cfg.horizontal_offset_m > cfg.plane_width_m / 2
