from isaaclab.managers import TerminationTermCfg as Term
from isaaclab.utils import configclass
from isaaclab.envs import mdp
from .._legacy_state import state


def success(env):
    return state(env).success


def unsafe(env):
    return state(env).failure


@configclass
class TerminationsCfg:
    success = Term(func=success)
    unsafe = Term(func=unsafe)
    time_out = Term(func=mdp.time_out, time_out=True)
