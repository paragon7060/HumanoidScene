from types import SimpleNamespace

import numpy as np
import torch

from kuavo_isaaclab_scene.display.camera_frames import camera_rgb, camera_depth


class Camera:
    image_shape = (2, 3)

    def __init__(self, name, frames):
        self.frames = iter(frames)
        self._rep_registry = {name: [SimpleNamespace(get_data=lambda: next(self.frames))]}

    @property
    def data(self):
        raise AssertionError("Must validate raw frames before accessing the Lab shape-sensitive cache")

    def _process_annotator_output(self, name, output):
        if isinstance(output, dict):
            output = output["data"]
        return torch.as_tensor(output), None


def test_rgb_recovers_from_empty_frames_before_and_after_valid_frame():
    valid = np.full((2, 3, 4), 127, dtype=np.uint8)
    empty = np.array([], dtype=np.float32)
    camera = Camera("rgb", [empty, valid, empty, {"data": valid, "info": {}}])
    assert camera_rgb(camera) is None
    np.testing.assert_array_equal(camera_rgb(camera), valid[..., :3])
    assert camera_rgb(camera) is None
    np.testing.assert_array_equal(camera_rgb(camera), valid[..., :3])


def test_float_rgb_converts_and_wrong_resolution_is_not_recorded():
    camera = Camera("rgb", [np.full((2, 3, 4), .5), np.ones((1, 1, 3))])
    np.testing.assert_array_equal(camera_rgb(camera), np.full((2, 3, 3), 127, dtype=np.uint8))
    assert camera_rgb(camera) is None


def test_depth_waits_for_nonempty_data_and_preserves_infinite_distance():
    valid = np.full((2, 3, 1), np.inf, dtype=np.float32)
    camera = Camera("distance_to_image_plane", [np.array([]), valid])
    assert camera_depth(camera) is None
    result = camera_depth(camera)
    assert result.dtype == np.float16 and np.isinf(result).all()
