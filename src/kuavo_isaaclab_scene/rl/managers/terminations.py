"""True terminals vs time-limit truncations, handled separately by PPO."""

from isaaclab.envs import mdp
from isaaclab.managers import TerminationTermCfg as Done
from isaaclab.utils import configclass
from ..mdp import terminations


@configclass
class TerminationsCfg:
    success = Done(func=terminations.success)
    unsafe = Done(func=terminations.unsafe)
    time_out = Done(func=mdp.time_out, time_out=True)
