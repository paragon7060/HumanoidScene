"""CPU-only texture construction checks; no XR, CUDA, or visual inspection."""

import numpy as np
from kuavo_isaaclab_scene.display.xr_reward_panel import reward_image


def test_reward_texture_is_opaque_and_contains_drawn_pixels():
    rgba = reward_image("RL REWARD | READY\nlift: +0.16667\ncollision: -0.02000")
    assert rgba.shape == (864, 768, 4)
    assert rgba.dtype == np.uint8
    assert np.all(rgba[..., 3] == 255)
    # Interior pixels, not just the border, contain the text.
    assert (rgba[12:110, 12:-12, :3] != np.array([16, 22, 32])).any()


def test_custom_long_reward_list_fits_same_texture_size():
    rgba = reward_image("\n".join(f"term_{i}: +0.12345" for i in range(60)))
    assert rgba.shape == (864, 768, 4)
    assert np.all(rgba[..., 3] == 255)


def test_failure_header_never_shrinks_with_long_reward_body():
    kwargs = dict(headline="FAILED: OBSTACLE COLLISION", checks=["NEED: Hold 0.2 / 0.5 s"], failure=True)
    short = reward_image("one term", **kwargs)
    long = reward_image("\n".join(f"term_{i}: +0.1" for i in range(100)), **kwargs)
    np.testing.assert_array_equal(short[:100], long[:100])
    assert short.shape == long.shape == (864, 768, 4)
