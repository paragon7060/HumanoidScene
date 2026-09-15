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
