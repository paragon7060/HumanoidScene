"""Reward weights and function parameters: primary reward tuning entrypoint."""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as Reward
from isaaclab.utils import configclass
from ..mdp import rewards


@configclass
class RewardsCfg:
    navigation = Reward(func=rewards.navigation, weight=2.0)
    reaching = Reward(func=rewards.reaching, weight=2.0)
    lift = Reward(func=rewards.lift, weight=3.0)
    carrying = Reward(func=rewards.carrying, weight=1.0)
    placement = Reward(func=rewards.placement, weight=3.0)
    button_reach = Reward(func=rewards.button_reach, weight=2.0)
    stability = Reward(func=rewards.stability, weight=-0.5)
    action_rate = Reward(func=mdp.action_rate_l2, weight=-0.01)
    joint_speed = Reward(func=mdp.joint_vel_l2, weight=-1e-4)
    time_cost = Reward(func=mdp.is_alive, weight=-0.10)
    stage_completed = Reward(func=rewards.stage_completed, weight=30.0)
    success = Reward(func=rewards.success, weight=150.0)
    failure = Reward(func=rewards.failure, weight=-60.0)
