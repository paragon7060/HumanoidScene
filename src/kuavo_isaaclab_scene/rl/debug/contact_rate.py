"""Quest reward-debug contact sampling controls; training configs stay unchanged."""


def configure_obstacle_contact_rate(scene, hz: int) -> int:
    """Decimate filtered obstacle reports without touching grasp contacts.

    PhysX collision simulation keeps its configured physics rate. This changes
    only how often per-body filtered force matrices are copied into contact
    sensor buffers.
    """
    if hz <= 0:
        raise ValueError("Obstacle contact rate must be positive.")
    period = 1.0 / hz
    count = 0
    for name, sensor in vars(scene).items():
        if name.startswith("obstacle_contact_") and sensor is not None:
            sensor.update_period = period
            count += 1
    return count


def configure_realtime_reward_debug(cfg) -> int:
    """Build the rollerless, policy-free 20 Hz inspection configuration."""
    cfg.sim.dt = 1.0 / 30.0
    cfg.decimation = 1
    cfg.sim.render_interval = 1
    cfg.observations.policy = None
    cfg.recorders.success_states = None
    cfg.commands.workcell.collision_reporting = "aggregate"
    cfg.commands.workcell.post_step_measurement = False

    removed = 0
    for name, sensor in vars(cfg.scene).items():
        if name.startswith("obstacle_contact_") and sensor is not None:
            setattr(cfg.scene, name, None)
            removed += 1

    for asset_name in ("robot", *cfg.task.box_names):
        props = getattr(cfg.scene, asset_name).spawn.articulation_props
        props.solver_position_iteration_count = 8
        props.solver_velocity_iteration_count = 2
    return removed
