"""Reset distribution schedule, independently editable from rewards/events."""

from isaaclab.managers import CurriculumTermCfg
from isaaclab.utils import configclass
from ..mdp.curriculum import reset_difficulty


@configclass
class CurriculumCfg:
    reset_difficulty = CurriculumTermCfg(func=reset_difficulty, params={"ramp_steps": 300_000})
