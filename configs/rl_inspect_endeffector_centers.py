"""VR inspection only: control BOTH hands without editing right-arm training config."""

from dataclasses import replace
from pathlib import Path
import runpy

_base = runpy.run_path(str(Path(__file__).with_name("rl_pick_arms_only.py")))
INITIAL_STATE = _base["INITIAL_STATE"]
configure = _base["configure"]


def configure_task(spec):
    # Keep the right-hand task/rewards; only unlock both arm/gripper controls for inspection.
    return replace(_base["configure_task"](spec), active_arm="both")
