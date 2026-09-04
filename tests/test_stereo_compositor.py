import numpy as np

from kuavo_isaaclab_scene.display.stereo_compositor import compose_stereo_atlas


def test_stereo_atlas_keeps_left_and_right_scene_separate():
    left = np.zeros((240, 240, 3), dtype=np.uint8)
    right = np.zeros_like(left)
    left[..., 0] = 40
    right[..., 2] = 90
    head = np.full((90, 160, 3), 120, dtype=np.uint8)
    wrist = np.full((120, 160, 3), 180, dtype=np.uint8)

    atlas = compose_stereo_atlas(left, right, head, wrist, wrist)

    assert atlas.shape == (240, 480, 3)
    # Sample clear scene pixels away from all three panels.
    np.testing.assert_array_equal(atlas[120, 10], [40, 0, 0])
    np.testing.assert_array_equal(atlas[120, 250], [0, 0, 90])


def test_stereo_atlas_letterboxes_panels_without_changing_eye_size():
    scene = np.zeros((300, 400, 3), dtype=np.uint8)
    head = np.full((9, 16, 3), 255, dtype=np.uint8)
    wrist = np.full((3, 4, 3), 127, dtype=np.uint8)

    atlas = compose_stereo_atlas(scene, scene, head, wrist, wrist)

    assert atlas.shape == (300, 800, 3)
    assert atlas.dtype == np.uint8
    assert atlas.flags.c_contiguous
