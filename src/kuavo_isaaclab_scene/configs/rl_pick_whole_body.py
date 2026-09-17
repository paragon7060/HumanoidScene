"""Single-box right-hand flap lift with base, torso and both arms available."""

from dataclasses import replace
from kuavo_isaaclab_scene.configs.rl_pick_arms_only import configure_task as flap_task
from kuavo_isaaclab_scene.configs.rl_pick_arms_only import INITIAL_STATE

def configure_task(spec):
    # Retain the currently approved flap/contact/reward task, but release the
    # second hand, torso and planar base. No curriculum or navigation stage.
    return replace(flap_task(spec), control_mode="whole-body", active_arm="both", action_space="all-joints")


def configure(env_cfg, agent_cfg):
    if env_cfg.task.control_mode == "whole-body":
        env_cfg.actions.base.velocity_limits = (.15, .15, .50)
        env_cfg.actions.base.acceleration_limits = (.30, .30, .80)
        env_cfg.actions.upper_body.scale = {"waist_yaw_joint": .01, "zarm_.*_joint": .02}
        env_cfg.actions.height.scale = .015
        env_cfg.actions.head.scale = .01
    else:
        env_cfg.actions.upper_body.scale = .02
    env_cfg.rewards.prelift_disturbance.weight = -.25
    env_cfg.rewards.orientation.weight = .5
    env_cfg.rewards.orientation.params["distance_threshold"] = .10
