"""Simulator-independent action assembly for the browser preview."""

import numpy as np

from .browser_teleop_bridge import BrowserTrackingSample
from .teleop_body import TeleopBodyMapper


def browser_body_action(
    sample: BrowserTrackingSample,
    mapper: TeleopBodyMapper,
    dt: float,
    *,
    control_allowed: bool,
) -> np.ndarray:
    """Share collector speed/deadzone/lift logic; stop on missing controllers."""
    left = sample.left_controller
    right = sample.right_controller
    return mapper.advance(
        None if left is None else left.native_packet(),
        None if right is None else right.native_packet(),
        dt,
        enabled=control_allowed and left is not None and right is not None,
    )


def compose_browser_action(mapped_action, gripper_action, body_action) -> np.ndarray:
    """Relative arms(12), head(2), active grippers(0..2), then body(6)."""
    mapped = np.asarray(mapped_action, dtype=np.float32)
    gripper = np.asarray(gripper_action, dtype=np.float32)
    body = np.asarray(body_action, dtype=np.float32)
    if mapped.shape != (14,) or gripper.ndim != 1 or gripper.size > 2 or body.shape != (6,):
        raise ValueError("Browser actions require 14 arm/head, 0..2 gripper and 6 body values.")
    return np.concatenate((mapped, gripper, body))
