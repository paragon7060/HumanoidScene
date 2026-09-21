"""Pure helpers for staged real-robot trajectory replay."""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def quintic_approach(
    start_arm_rad: np.ndarray,
    target_arm_rad: np.ndarray,
    duration_s: float,
    publish_hz: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return a zero-velocity/acceleration endpoint approach trajectory."""
    start = np.asarray(start_arm_rad, dtype=np.float64)
    target = np.asarray(target_arm_rad, dtype=np.float64)
    if start.shape != (14,) or target.shape != (14,):
        raise ValueError("Approach start and target must contain 14 joints")
    if not np.isfinite(start).all() or not np.isfinite(target).all():
        raise ValueError("Approach endpoints must be finite")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Approach duration must be finite and positive")
    if not math.isfinite(publish_hz) or publish_hz <= 0.0:
        raise ValueError("Approach publish rate must be finite and positive")
    intervals = max(1, int(math.ceil(duration_s * publish_hz)))
    timestamps = np.linspace(0.0, duration_s, intervals + 1, dtype=np.float64)
    phase = timestamps / duration_s
    blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
    positions = start[None, :] + blend[:, None] * (target - start)[None, :]
    positions[0] = start
    positions[-1] = target
    return timestamps, positions
