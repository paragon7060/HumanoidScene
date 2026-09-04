"""Pure image composition helpers for the lightweight Quest browser preview."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class StereoPanelLayoutCfg:
    """Relative sizes for camera panels over each stereo scene eye."""

    head_width_fraction: float = 0.28
    wrist_width_fraction: float = 0.20
    margin_fraction: float = 0.025

    def __post_init__(self) -> None:
        for name, value in (
            ("head_width_fraction", self.head_width_fraction),
            ("wrist_width_fraction", self.wrist_width_fraction),
            ("margin_fraction", self.margin_fraction),
        ):
            if not 0.0 < value < 0.5:
                raise ValueError(f"{name} must be between 0 and 0.5.")


def _rgb(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError(f"Expected an HxWx3 RGB image, received {image.shape}.")
    return np.ascontiguousarray(image[..., :3], dtype=np.uint8)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize with letterboxing so camera aspect ratios are not stretched."""
    image = _rgb(image)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, int(round(image.shape[1] * scale)))
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    panel = np.full((height, width, 3), 8, dtype=np.uint8)
    x0 = (width - resized_width) // 2
    y0 = (height - resized_height) // 2
    panel[y0 : y0 + resized_height, x0 : x0 + resized_width] = resized
    return panel


def _overlay_panel(canvas: np.ndarray, image: np.ndarray, x0: int, y0: int, width: int, height: int) -> None:
    panel = _fit_panel(image, width, height)
    canvas[y0 : y0 + height, x0 : x0 + width] = panel
    cv2.rectangle(canvas, (x0, y0), (x0 + width - 1, y0 + height - 1), (118, 185, 0), 2)


def compose_stereo_atlas(
    left_scene: np.ndarray,
    right_scene: np.ndarray,
    head: np.ndarray,
    left_wrist: np.ndarray,
    right_wrist: np.ndarray,
    cfg: StereoPanelLayoutCfg | None = None,
) -> np.ndarray:
    """Return side-by-side eye images with small head and wrist panels.

    The left half is sampled only by the Quest left eye and the right half only
    by the right eye. Scene pixels therefore retain binocular disparity, while
    the three physical-camera views remain small status panels.
    """
    cfg = cfg or StereoPanelLayoutCfg()
    left_scene = _rgb(left_scene)
    right_scene = _rgb(right_scene)
    height, width = left_scene.shape[:2]
    if right_scene.shape[:2] != (height, width):
        right_scene = cv2.resize(right_scene, (width, height), interpolation=cv2.INTER_AREA)

    margin = max(8, int(round(min(width, height) * cfg.margin_fraction)))
    head_width = max(96, int(round(width * cfg.head_width_fraction)))
    wrist_width = max(80, int(round(width * cfg.wrist_width_fraction)))
    head_height = max(72, int(round(head_width * 0.5625)))
    wrist_height = max(60, int(round(wrist_width * 0.75)))
    if head_width + 2 * margin > width or wrist_width + 2 * margin > width:
        raise ValueError("Stereo eye resolution is too small for the configured camera panels.")

    eyes = []
    for scene in (left_scene, right_scene):
        eye = scene.copy()
        _overlay_panel(eye, head, (width - head_width) // 2, margin, head_width, head_height)
        y0 = height - wrist_height - margin
        _overlay_panel(eye, left_wrist, margin, y0, wrist_width, wrist_height)
        _overlay_panel(eye, right_wrist, width - wrist_width - margin, y0, wrist_width, wrist_height)
        eyes.append(eye)
    return np.ascontiguousarray(np.concatenate(eyes, axis=1))
