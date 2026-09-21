"""Pure tensor helpers for privileged contact measurements."""

from __future__ import annotations

import torch


def maximum_filtered_force(env, sensor_names: tuple[str, ...]) -> torch.Tensor:
    """Return the maximum filtered normal force for each environment."""
    maxima = []
    for sensor_name in sensor_names:
        force = env.scene[sensor_name].data.force_matrix_w
        if force is None or force.shape[0] != env.num_envs:
            raise RuntimeError(f"{sensor_name} filtered contact forces are unavailable.")
        maxima.append(force.norm(dim=-1).flatten(1).amax(dim=-1))
    if not maxima:
        raise ValueError("At least one filtered contact sensor is required.")
    return torch.stack(maxima, dim=-1).amax(dim=-1)
