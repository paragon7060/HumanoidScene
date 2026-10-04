"""Batched S63 torso X/Z kinematics shared by RL control and demo conversion."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import torch


TORSO_JOINT_NAMES = ("knee_joint", "leg_joint", "waist_pitch_joint")
TORSO_XZ_SPEED_M_S = 0.10
TORSO_FORWARD_LIMIT_M = 0.15
TORSO_HEIGHT_RANGE_M = (0.0, 0.40)
TORSO_JOINT_STEP_RAD = 0.015


def configure_upright_travel_profile(cfg, contract, extra_height_m):
    """Select explicit software travel without changing hard joint/safety limits.

    The identifier preserves the first measured higher-travel diagnostic. All
    controllers can use that same physical MDP; existing defaults stay intact.
    """
    import math
    if not math.isfinite(extra_height_m) or not 0 <= extra_height_m <= .08:
        raise ValueError('Upright extra height must be within0..8cm')
    if not extra_height_m:
        return contract
    effective = (TORSO_HEIGHT_RANGE_M[0], TORSO_HEIGHT_RANGE_M[1] + extra_height_m)
    name=f's63_upright_torso_xz_fixed_pitch_diagnostic_up_{extra_height_m:.4f}m'
    profile=dict(extra_height_m=extra_height_m,
        original_height_range_m=list(TORSO_HEIGHT_RANGE_M),
        effective_height_range_m=list(effective),global_default_changed=False,
        physical_joint_limits_changed=False,pitch_control='fixed_reset_pitch')
    # Prepared staged manifests already identify this reviewed extension.
    # Apply it to a fresh cfg without adding height again or weakening checks.
    if (contract.get('action_contract') not in ('s63_upright_torso_xz_fixed_pitch_v1',name)
            or tuple(cfg.actions.height.height_range_m) not in (TORSO_HEIGHT_RANGE_M,effective)
            or ('upright_torso_diagnostic' in contract and contract['upright_torso_diagnostic']!=profile)):
        raise ValueError('Higher torso travel requires the original reviewed upright profile')
    cfg.actions.height.height_range_m = effective
    return contract | dict(action_contract=name,upright_torso_diagnostic=profile)


@lru_cache(maxsize=4)
def torso_links_from_urdf(path: str | Path) -> tuple[tuple[float, float], ...]:
    """Use the same leg/waist link origins as the Quest upright-body mapper."""
    joints = {joint.attrib["name"]: joint for joint in ET.parse(path).findall("joint")}
    return tuple(
        tuple(float(value) for value in joints[name].find("origin").attrib["xyz"].split()[::2])
        for name in ("leg_joint", "waist_pitch_joint")
    )


def planar_position(q: torch.Tensor, links: torch.Tensor) -> torch.Tensor:
    """Torso X/Z above the chassis from the knee and leg joint angles."""
    a, b = q[..., 0], q[..., 0] + q[..., 1]
    x = a.cos() * links[0, 0] + a.sin() * links[0, 1] \
        + b.cos() * links[1, 0] + b.sin() * links[1, 1]
    z = -a.sin() * links[0, 0] + a.cos() * links[0, 1] \
        - b.sin() * links[1, 0] + b.cos() * links[1, 1]
    return torch.stack((x, z), dim=-1)


def planar_jacobian(q: torch.Tensor, links: torch.Tensor) -> torch.Tensor:
    a, b = q[..., 0], q[..., 0] + q[..., 1]
    second = torch.stack((
        -b.sin() * links[1, 0] + b.cos() * links[1, 1],
        -b.cos() * links[1, 0] - b.sin() * links[1, 1],
    ), dim=-1)
    first = second + torch.stack((
        -a.sin() * links[0, 0] + a.cos() * links[0, 1],
        -a.cos() * links[0, 0] - a.sin() * links[0, 1],
    ), dim=-1)
    return torch.stack((first, second), dim=-1)


def upright_joint_step(
    current: torch.Tensor,
    target_xz: torch.Tensor,
    pitch_reference: torch.Tensor,
    links: torch.Tensor,
    joint_limits: torch.Tensor,
    *,
    max_joint_step: float = TORSO_JOINT_STEP_RAD,
    iterations: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve reachable X/Z motion and preserve the reset torso pitch exactly.

    Targets that exceed speed or physical joint limits are projected to the
    achieved X/Z position, so repeated commands cannot build up IK windup.
    """
    q = current[:, :2].clone()
    for _ in range(iterations):
        error = target_xz - planar_position(q, links)
        jac = planar_jacobian(q, links)
        # Solve the damped 2x2 normal equations explicitly. Unlike CUDA
        # torch.linalg.solve this does not synchronize thousands of envs with
        # the CPU on every control tick.
        j00, j01 = jac[:, 0, 0], jac[:, 0, 1]
        j10, j11 = jac[:, 1, 0], jac[:, 1, 1]
        a = j00.square() + j10.square() + 1e-5
        b = j00 * j01 + j10 * j11
        d = j01.square() + j11.square() + 1e-5
        r0 = j00 * error[:, 0] + j10 * error[:, 1]
        r1 = j01 * error[:, 0] + j11 * error[:, 1]
        determinant = a * d - b.square()
        update = torch.stack(((d * r0 - b * r1) / determinant,
                              (a * r1 - b * r0) / determinant), dim=-1)
        q = (q + update.clamp(-0.05, 0.05)).clamp(
            joint_limits[:, :2, 0], joint_limits[:, :2, 1])
    delta = (q - current[:, :2]).clamp(-max_joint_step, max_joint_step)
    pitch_change = -delta.sum(-1)
    old_pitch = current[:, 2]
    pitch_bound = torch.where(pitch_change >= 0, joint_limits[:, 2, 1], joint_limits[:, 2, 0])
    fraction = torch.where(
        pitch_change.abs() > 1e-12,
        (pitch_bound - old_pitch) / torch.where(
            pitch_change.abs() > 1e-12, pitch_change, torch.ones_like(pitch_change)),
        torch.ones_like(pitch_change),
    ).clamp(0, 1)
    delta *= fraction[:, None]
    q = current[:, :2] + delta
    pitch = pitch_reference - q.sum(-1)
    result = torch.cat((q, pitch[:, None]), dim=-1)
    return result, planar_position(q, links)
