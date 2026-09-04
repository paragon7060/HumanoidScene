"""Startup domain randomization, ordered reset, and belt update events."""

from isaaclab.envs import mdp
from isaaclab.managers import EventTermCfg as Event, SceneEntityCfg
from isaaclab.utils import configclass
from ..mdp.events import reset_episode, conveyor_motion
from ...envs.manager_mdp import randomize_box_flap_joint_friction


@configclass
class EventsCfg:
    arm_mass = Event(func=mdp.randomize_rigid_body_mass, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", body_names="zarm_.*_link"),
        "mass_distribution_params": (0.95, 1.05), "operation": "scale"})
    arm_gains = Event(func=mdp.randomize_actuator_gains, mode="startup", params={
        "asset_cfg": SceneEntityCfg("robot", joint_names="zarm_.*_joint"),
        "stiffness_distribution_params": (0.95, 1.05), "damping_distribution_params": (0.95, 1.05),
        "operation": "scale"})
    reset = Event(func=reset_episode, mode="reset")
    flap_friction = Event(func=randomize_box_flap_joint_friction, mode="reset", params={
        "asset_names": (), "static_friction_range": (0.0, 0.04), "dynamic_friction_range": (0.0, 0.02)})
    belt = Event(func=conveyor_motion, mode="interval", interval_range_s=(0.0333333333, 0.0333333333))
