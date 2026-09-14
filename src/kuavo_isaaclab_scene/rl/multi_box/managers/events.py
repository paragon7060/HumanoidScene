from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as Term
from isaaclab.utils import configclass
from ..reset_states import restore


def reset(env, env_ids):
    mdp.reset_scene_to_default(env, env_ids)
    if env.cfg.multi_box.reset_bank:
        restore(env, env_ids)


@configclass
class EventsCfg:
    reset = Term(func=reset, mode="reset")
