"""Obstacle-only contacts; box grasp/support forces do not count as collisions."""

import torch


def obstacle_forces(env):
    values = []
    for name, sensor in env.scene.sensors.items():
        if not name.startswith("obstacle_contact_"):
            continue
        history = sensor.data.force_matrix_w_history
        if history is None or history.shape[-2] == 0 or history.shape[1] < env.cfg.decimation:
            raise RuntimeError(f"Missing obstacle filters or physics-step history: {name}")
        # Reduce magnitudes per filtered obstacle before aggregating, so opposite forces
        # on different obstacles cannot cancel. Retain brief substep impacts.
        values.append(history[:, :env.cfg.decimation].norm(dim=-1).flatten(1).amax(-1))
    if not values:
        raise RuntimeError("flap_top requires obstacle contact sensors on the robot.")
    return torch.stack(values, dim=-1)
