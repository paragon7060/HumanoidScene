"""VR inspection only: control both hands; keep the default right-hand task."""

from dataclasses import replace
from pathlib import Path
import runpy

_base = runpy.run_path(str(Path(__file__).with_name("rl_pick_arms_only.py")))
INITIAL_STATE = _base["INITIAL_STATE"]
configure = _base["configure"]


def configure_task(spec):
    return replace(_base["configure_task"](spec), active_arm="both")
