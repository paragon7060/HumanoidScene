"""Copy this trusted config for an experiment; pass --config PATH to BOTH strategies."""
from dataclasses import replace


def configure_spec(spec):
    return replace(spec,
        box_names=("small_box_0", "small_box_1", "medium_box_0", "large_box_0"),
        shelves=(1, 1, 2, 2),
        lift_height=.06,
        placement_hold=.5,
        rack_outward_local=(0., 1., 0.),
    )


def configure(env_cfg, agent_cfg):
    env_cfg.rewards.placed.weight = 25.
    env_cfg.rewards.success.weight = 100.
    env_cfg.rewards.time.weight = -.2
    agent_cfg.num_steps_per_env = 32
    agent_cfg.save_interval = 100
