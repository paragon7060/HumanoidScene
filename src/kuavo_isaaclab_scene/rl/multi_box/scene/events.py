"""Isaac Lab event configuration for randomized v2 scenes."""

from isaaclab.managers import EventTermCfg as Term
from isaaclab.utils import configclass

from .randomization import reset_randomized_scene


@configclass
class SceneEventsCfg:
    reset = Term(func=reset_randomized_scene, mode="reset")
