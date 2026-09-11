"""Reward weights and function parameters: primary reward tuning entrypoint."""

from isaaclab.envs import mdp
from isaaclab.managers import RewardTermCfg as Reward
from isaaclab.utils import configclass
from ..mdp import rewards


@configclass
class RewardsCfg:
    navigation = Reward(func=rewards.navigation, weight=2.0)
    approach_reaching = Reward(func=rewards.approach_reaching, weight=1.0)
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


@configclass
class FlapPickRewardsCfg(RewardsCfg):
    navigation = None
    approach_reaching = None  # stationary pick: no base-navigation phase
    carrying = None
    placement = None
    button_reach = None
    reaching = Reward(func=rewards.flap_reaching, weight=4.0)
    orientation = Reward(func=rewards.flap_orientation, weight=0.5, params={"distance_threshold": 0.10})
    flap_contact = Reward(func=rewards.flap_contact, weight=3.0)
    lift = Reward(func=rewards.flap_lift_progress, weight=5.0)
    holding = Reward(func=rewards.flap_hold, weight=5.0)
    collision = Reward(func=rewards.unwanted_contact, weight=-2.0)
    action_rate = Reward(func=rewards.settled_action_rate, weight=-0.01)
    joint_speed = Reward(func=rewards.settled_joint_speed, weight=-1e-4)
    time_cost = Reward(func=rewards.settled_time, weight=-0.10)
    prelift_disturbance = Reward(func=rewards.prelift_disturbance, weight=-0.25)
    stable_grasp = None  # no renewable reward for grasping a box left on the shelf
