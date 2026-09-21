"""Dataset action encoding, independent of simulator actuator commands."""

import numpy as np


GRIPPER_ACTION_ENCODING = "binary_close: 0=open, 1=close"
GRIPPER_CHANNELS = {"left_gripper", "right_gripper"}


def recorded_action_names(action_names):
    return tuple(name + "_close" if name in GRIPPER_CHANNELS else name
                 for name in action_names)


def encode_recorded_action(action, action_names):
    """Validate and copy the shared 0=open/1=close command for recording."""
    result = np.array(action, dtype=np.float32, copy=True)
    if result.shape != (len(action_names),):
        raise ValueError("Recorded action must match its named one-dimensional schema")
    indices = [i for i, name in enumerate(action_names) if name in GRIPPER_CHANNELS]
    if not np.isfinite(result[indices]).all():
        raise ValueError("Non-finite gripper command cannot be recorded as a binary action")
    if not np.isin(result[indices], (0.0, 1.0)).all():
        raise ValueError("Gripper commands must use binary 0=open/1=close encoding")
    return result
