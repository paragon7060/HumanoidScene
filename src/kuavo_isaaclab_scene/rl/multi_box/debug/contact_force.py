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


def maximum_non_rack_force(
    net_forces_w: torch.Tensor,
    rack_force_matrices: tuple[torch.Tensor, ...],
    body_indices: tuple[int, ...],
) -> torch.Tensor:
    """Remove filtered rack contact vectors from each robot body's net force."""
    if net_forces_w.ndim != 3 or net_forces_w.shape[-1] != 3:
        raise ValueError("Net contact forces must have shape (envs, bodies, 3).")
    if len(rack_force_matrices) != len(body_indices) or len(set(body_indices)) != len(body_indices):
        raise ValueError("Rack sensors must map one-to-one to distinct robot bodies.")
    non_rack = net_forces_w.clone()
    for force, index in zip(rack_force_matrices, body_indices, strict=True):
        if not 0 <= index < net_forces_w.shape[1] or force.ndim != 4 \
                or force.shape[:2] != (net_forces_w.shape[0], 1) or force.shape[-1] != 3:
            raise ValueError("Invalid filtered rack force shape or robot body index.")
        non_rack[:, index] -= force[:, 0].sum(dim=1)
    return non_rack.norm(dim=-1).amax(dim=-1)
