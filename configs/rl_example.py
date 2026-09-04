"""Pass --config configs/rl_example.py. This trusted file runs after AppLauncher."""

from dataclasses import replace


def configure_task(spec):
    """Edit physical success thresholds/reset distributions BEFORE scene assembly."""
    return replace(spec,
        navigation_tolerance=0.08,
        heading_tolerance=0.15,
        hold_seconds=0.4,
        required_grasp_hands=1,
        reset_xy_jitter=0.15,
        # reset_bank/snapshot_dir/box_names are normally supplied through CLI.
    )


def configure(env_cfg, agent_cfg):
    """Edit individual manager terms and the PPO algorithm AFTER assembly."""
    env_cfg.rewards.reaching.weight = 3.0
    env_cfg.rewards.action_rate.weight = -0.015
    env_cfg.actions.base.velocity_limits = (0.20, 0.20, 0.60)
    env_cfg.actions.left_gripper.delta_scale = 0.08
    env_cfg.actions.right_gripper.delta_scale = 0.08
    agent_cfg.algorithm.learning_rate = 2e-4
    agent_cfg.num_steps_per_env = 32
    # Set a term to None to remove it, for example:
    # env_cfg.events.arm_mass = None
    # Do not change env_cfg.task here; use configure_task() instead.
