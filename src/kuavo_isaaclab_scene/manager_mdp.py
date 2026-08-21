"""Manager terms for the Kuavo robustness environment."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaacsim.core.utils.stage as stage_utils

from isaaclab.assets import Articulation, RigidObjectCollection
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils
from .workcell_layout import (
    position as layout_position,
    quat_conjugate,
    remap_point,
    remap_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


TASK_TOTE_COUNT = 6
CARGO_PER_TOTE = 2
CONVEYOR_VISUAL_POS = layout_position("conveyor")
CONVEYOR_SLOT_PITCH = 0.26
CONVEYOR_SLOTS = tuple(
    remap_point(
        "conveyor",
        (0.65 + CONVEYOR_SLOT_PITCH * slot_id, -0.52, 0.775),
    )
    for slot_id in range(9)
)
CONVEYOR_SURFACE_CENTER = remap_point("conveyor", (1.69, -0.52, 0.753))
CONVEYOR_GEOMETRY_ROT = remap_quat("conveyor", (1.0, 0.0, 0.0, 0.0))
CONVEYOR_INV_GEOMETRY_ROT = quat_conjugate(CONVEYOR_GEOMETRY_ROT)
CONVEYOR_OBJECT_ROT = CONVEYOR_GEOMETRY_ROT
BUTTON_STATION_POS = layout_position("button_station")
BUTTON_PRESS_THRESHOLD = 0.006


def _env_ids(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, device: str) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=device, dtype=torch.long)


def task_totes_on_conveyor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
) -> torch.Tensor:
    """Whether each of the six task totes occupies the stopped conveyor."""
    totes: RigidObjectCollection = env.scene[asset_cfg.name]
    positions = totes.data.object_link_pos_w[:, :TASK_TOTE_COUNT] - env.scene.env_origins[:, None, :]
    center = torch.tensor(CONVEYOR_SURFACE_CENTER, device=totes.device)
    relative = positions - center
    inverse_rotation = torch.tensor(CONVEYOR_INV_GEOMETRY_ROT, device=totes.device)
    local = math_utils.quat_apply(
        inverse_rotation.expand(relative.shape[0] * relative.shape[1], -1),
        relative.reshape(-1, 3),
    ).reshape_as(relative)
    return (
        (torch.abs(local[..., 0]) <= 1.285)
        & (torch.abs(local[..., 1]) <= 0.18)
        & (local[..., 2] >= -0.133)
        & (local[..., 2] <= 0.297)
    )


def custom_boxes_on_conveyor(
    env: ManagerBasedRLEnv,
    box_asset_names: tuple[str, ...],
) -> torch.Tensor:
    """Whether each configured standalone box occupies the stopped conveyor."""
    if not box_asset_names:
        return torch.empty((env.num_envs, 0), device=env.device, dtype=torch.bool)
    positions = torch.stack(
        [env.scene[name].data.root_pos_w for name in box_asset_names],
        dim=1,
    )
    positions = positions - env.scene.env_origins[:, None, :]
    center = torch.tensor(CONVEYOR_SURFACE_CENTER, device=positions.device)
    relative = positions - center
    inverse_rotation = torch.tensor(CONVEYOR_INV_GEOMETRY_ROT, device=positions.device)
    local = math_utils.quat_apply(
        inverse_rotation.expand(relative.shape[0] * relative.shape[1], -1),
        relative.reshape(-1, 3),
    ).reshape_as(relative)
    return (
        (torch.abs(local[..., 0]) <= 1.285)
        & (torch.abs(local[..., 1]) <= 0.34)
        & (local[..., 2] >= -0.20)
        & (local[..., 2] <= 0.45)
    )


def task_objects_on_conveyor(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Return occupancy for the selected task representation.

    Captured/local Rack USD box instances are authoritative when supplied;
    otherwise the original six-object tote collection remains supported.
    """
    if box_asset_names:
        return custom_boxes_on_conveyor(env, box_asset_names)
    return task_totes_on_conveyor(env, asset_cfg)


def task_progress(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return task_objects_on_conveyor(env, asset_cfg, box_asset_names).float().mean(dim=1)


def task_progress_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return task_progress(env, asset_cfg, box_asset_names).unsqueeze(-1)


def button_armed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return torch.all(task_objects_on_conveyor(env, asset_cfg, box_asset_names), dim=1)


def button_armed_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return button_armed(env, asset_cfg, box_asset_names).float().unsqueeze(-1)


def button_pressed(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("button_station"),
    threshold: float = BUTTON_PRESS_THRESHOLD,
) -> torch.Tensor:
    """Detect a real press from the spring-loaded plunger joint travel."""
    button_station = env.scene[asset_cfg.name]
    return button_station.data.joint_pos[:, 0] >= threshold


def button_travel_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("button_station"),
) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos[:, :1]


def task_success(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    button_cfg: SceneEntityCfg = SceneEntityCfg("button_station"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return button_armed(env, tote_cfg, box_asset_names) & button_pressed(env, button_cfg)


def task_success_reward(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    button_cfg: SceneEntityCfg = SceneEntityCfg("button_station"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    return task_success(env, tote_cfg, button_cfg, box_asset_names).float()


def tote_poses(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
) -> torch.Tensor:
    """Task-tote poses relative to each environment origin."""
    totes: RigidObjectCollection = env.scene[asset_cfg.name]
    positions = totes.data.object_link_pos_w[:, :TASK_TOTE_COUNT] - env.scene.env_origins[:, None, :]
    quaternions = totes.data.object_link_quat_w[:, :TASK_TOTE_COUNT]
    return torch.cat((positions, quaternions), dim=-1).flatten(start_dim=1)


def tote_motion_obs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
) -> torch.Tensor:
    totes: RigidObjectCollection = env.scene[asset_cfg.name]
    return totes.data.object_com_vel_w[:, :TASK_TOTE_COUNT].flatten(start_dim=1)


def tote_stability_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
) -> torch.Tensor:
    """Reward upright, low-angular-rate transport to protect loose cargo."""
    totes: RigidObjectCollection = env.scene[asset_cfg.name]
    quaternions = totes.data.object_link_quat_w[:, :TASK_TOTE_COUNT]
    local_up = torch.zeros(
        (env.num_envs, TASK_TOTE_COUNT, 3),
        device=totes.device,
    )
    local_up[..., 2] = 1.0
    world_up = math_utils.quat_apply(
        quaternions.reshape(-1, 4),
        local_up.reshape(-1, 3),
    ).reshape(env.num_envs, TASK_TOTE_COUNT, 3)
    upright = torch.clamp(world_up[..., 2], min=0.0, max=1.0)
    angular_speed = torch.linalg.norm(
        totes.data.object_com_vel_w[:, :TASK_TOTE_COUNT, 3:],
        dim=-1,
    )
    smooth = torch.exp(-0.35 * angular_speed)
    return torch.mean(upright * smooth, dim=1)


def cargo_local_positions(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    """Cargo centers expressed in the coordinate frame of their assigned tote."""
    totes: RigidObjectCollection = env.scene[tote_cfg.name]
    cargo: RigidObjectCollection = env.scene[cargo_cfg.name]
    cargo_count = cargo.num_objects
    tote_ids = torch.arange(cargo_count, device=cargo.device) // CARGO_PER_TOTE
    tote_pos = totes.data.object_link_pos_w[:, tote_ids]
    tote_quat = totes.data.object_link_quat_w[:, tote_ids]
    delta = cargo.data.object_link_pos_w - tote_pos
    return math_utils.quat_apply_inverse(
        tote_quat.reshape(-1, 4),
        delta.reshape(-1, 3),
    ).reshape(env.num_envs, cargo_count, 3)


def cargo_state_obs(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    cargo: RigidObjectCollection = env.scene[cargo_cfg.name]
    local_pos = cargo_local_positions(env, tote_cfg, cargo_cfg)
    local_vel = cargo.data.object_com_vel_w[..., :3]
    return torch.cat((local_pos, local_vel), dim=-1).flatten(start_dim=1)


def cargo_retained_mask(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    local = cargo_local_positions(env, tote_cfg, cargo_cfg)
    return (
        (torch.abs(local[..., 0]) <= 0.108)
        & (torch.abs(local[..., 1]) <= 0.128)
        & (local[..., 2] >= -0.005)
        & (local[..., 2] <= 0.205)
    )


def cargo_retained_fraction(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    return cargo_retained_mask(env, tote_cfg, cargo_cfg).float().mean(dim=1)


def cargo_retained_obs(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    return cargo_retained_fraction(env, tote_cfg, cargo_cfg).unsqueeze(-1)


def cargo_spilled(
    env: ManagerBasedRLEnv,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
) -> torch.Tensor:
    return ~torch.all(cargo_retained_mask(env, tote_cfg, cargo_cfg), dim=1)


def tote_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.28,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
) -> torch.Tensor:
    totes: RigidObjectCollection = env.scene[asset_cfg.name]
    local_z = totes.data.object_link_pos_w[:, :TASK_TOTE_COUNT, 2] - env.scene.env_origins[:, None, 2]
    return torch.any(local_z < minimum_height, dim=1)


def task_object_dropped(
    env: ManagerBasedRLEnv,
    minimum_height: float = 0.10,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    box_asset_names: tuple[str, ...] = (),
) -> torch.Tensor:
    """Detect a task object falling to the floor for legacy or local boxes."""
    if not box_asset_names:
        return tote_dropped(env, minimum_height=max(minimum_height, 0.28), asset_cfg=asset_cfg)
    positions = torch.stack(
        [env.scene[name].data.root_pos_w for name in box_asset_names],
        dim=1,
    )
    local_z = positions[..., 2] - env.scene.env_origins[:, None, 2]
    return torch.any(local_z < minimum_height, dim=1)


def obstacle_state_obs(
    env: ManagerBasedRLEnv,
    human_cfg: SceneEntityCfg = SceneEntityCfg("moving_human"),
    mobile_cfg: SceneEntityCfg = SceneEntityCfg("moving_robot"),
) -> torch.Tensor:
    human = env.scene[human_cfg.name]
    mobile = env.scene[mobile_cfg.name]
    human_state = human.data.root_state_w.clone()
    mobile_state = mobile.data.root_state_w.clone()
    human_state[:, :3] -= env.scene.env_origins
    mobile_state[:, :3] -= env.scene.env_origins
    return torch.cat((human_state[:, :3], human_state[:, 7:10], mobile_state[:, :3], mobile_state[:, 7:10]), dim=1)


def obstacle_clearance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    human_cfg: SceneEntityCfg = SceneEntityCfg("moving_human"),
    mobile_cfg: SceneEntityCfg = SceneEntityCfg("moving_robot"),
) -> torch.Tensor:
    """Minimum planar clearance between selected Kuavo links and movers."""
    robot = env.scene[robot_cfg.name]
    human = env.scene[human_cfg.name]
    mobile = env.scene[mobile_cfg.name]
    body_xy = robot.data.body_link_pos_w[:, robot_cfg.body_ids, :2]
    obstacle_xy = torch.stack((human.data.root_pos_w[:, :2], mobile.data.root_pos_w[:, :2]), dim=1)
    center_dist = torch.linalg.norm(body_xy[:, :, None, :] - obstacle_xy[:, None, :, :], dim=-1)
    radii = torch.tensor((0.22, 0.31), device=robot.device)
    return torch.amin(center_dist - radii[None, None, :], dim=(1, 2))


def obstacle_proximity_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    safety_margin: float = 0.55,
) -> torch.Tensor:
    clearance = obstacle_clearance(env, robot_cfg)
    return torch.clamp((safety_margin - clearance) / safety_margin, min=0.0, max=1.0)


def obstacle_collision(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    collision_margin: float = 0.025,
) -> torch.Tensor:
    return obstacle_clearance(env, robot_cfg) <= collision_margin


def prefill_count_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    counts = getattr(env, "_workcell_prefill_count", None)
    if counts is None:
        counts = torch.zeros(env.num_envs, device=env.device)
    return (counts.float() / 3.0).unsqueeze(-1)


def robustness_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ramp_steps: int = 1_500_000,
) -> torch.Tensor:
    """Ramp pose noise, mover speed, and cargo disturbances during training."""
    del env_ids
    difficulty = min(float(env.common_step_counter) / float(ramp_steps), 1.0)
    env._robustness_difficulty = difficulty
    return torch.tensor(difficulty, device=env.device)


def robustness_difficulty_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    value = float(getattr(env, "_robustness_difficulty", 0.0))
    return torch.full((env.num_envs, 1), value, device=env.device)


def reset_totes_and_cargo(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    tote_cfg: SceneEntityCfg = SceneEntityCfg("totes"),
    cargo_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
    position_jitter: float = 0.012,
    yaw_jitter: float = math.radians(3.0),
    cargo_jitter: float = 0.008,
) -> None:
    """Randomize rack poses, prefill count, and loose cargo placement."""
    totes: RigidObjectCollection = env.scene[tote_cfg.name]
    cargo: RigidObjectCollection = env.scene[cargo_cfg.name]
    env_ids = _env_ids(env, env_ids, totes.device)
    difficulty = 0.25 + 0.75 * float(getattr(env, "_robustness_difficulty", 0.0))
    position_jitter *= difficulty
    yaw_jitter *= difficulty
    cargo_jitter *= difficulty

    tote_state = totes.data.default_object_state[env_ids].clone()
    tote_state[..., :3] += env.scene.env_origins[env_ids, None, :]

    pos_noise = torch.empty((len(env_ids), TASK_TOTE_COUNT, 2), device=totes.device).uniform_(
        -position_jitter,
        position_jitter,
    )
    tote_state[:, :TASK_TOTE_COUNT, :2] += pos_noise
    yaw = torch.empty((len(env_ids), TASK_TOTE_COUNT), device=totes.device).uniform_(-yaw_jitter, yaw_jitter)
    zeros = torch.zeros_like(yaw)
    yaw_quat = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    tote_state[:, :TASK_TOTE_COUNT, 3:7] = math_utils.quat_mul(
        tote_state[:, :TASK_TOTE_COUNT, 3:7],
        yaw_quat,
    )

    if not hasattr(env, "_workcell_prefill_count"):
        env._workcell_prefill_count = torch.zeros(env.num_envs, device=totes.device, dtype=torch.long)
    prefill_count = torch.randint(0, 4, (len(env_ids),), device=totes.device)
    env._workcell_prefill_count[env_ids] = prefill_count
    for local_env_id in range(len(env_ids)):
        origin = env.scene.env_origins[env_ids[local_env_id]]
        for foreign_id in range(3):
            object_id = TASK_TOTE_COUNT + foreign_id
            if foreign_id < int(prefill_count[local_env_id]):
                center = torch.tensor(CONVEYOR_SLOTS[foreign_id], device=totes.device)
                center[:2] += torch.empty(2, device=totes.device).uniform_(-0.008, 0.008)
            else:
                center = torch.tensor(
                    (2.45 + 0.27 * foreign_id, 1.55, 0.015),
                    device=totes.device,
                )
            tote_state[local_env_id, object_id, :3] = origin + center
            tote_state[local_env_id, object_id, 3:7] = torch.tensor(
                CONVEYOR_OBJECT_ROT if foreign_id < int(prefill_count[local_env_id]) else (1.0, 0.0, 0.0, 0.0),
                device=totes.device,
            )

    totes.write_object_link_pose_to_sim(tote_state[..., :7], env_ids=env_ids)
    totes.write_object_com_velocity_to_sim(
        torch.zeros((len(env_ids), totes.num_objects, 6), device=totes.device),
        env_ids=env_ids,
    )

    cargo_state = cargo.data.default_object_state[env_ids].clone()
    cargo_tote_ids = torch.arange(cargo.num_objects, device=cargo.device) // CARGO_PER_TOTE
    parent_pos = tote_state[:, cargo_tote_ids, :3]
    parent_quat = tote_state[:, cargo_tote_ids, 3:7]
    base_local = torch.tensor(
        [(-0.045, -0.052, 0.042), (0.045, 0.052, 0.042)] * TASK_TOTE_COUNT,
        device=cargo.device,
    )
    local = base_local[None, :, :].repeat(len(env_ids), 1, 1)
    local[..., :2] += torch.empty_like(local[..., :2]).uniform_(-cargo_jitter, cargo_jitter)
    world_offset = math_utils.quat_apply(
        parent_quat.reshape(-1, 4),
        local.reshape(-1, 3),
    ).reshape(len(env_ids), cargo.num_objects, 3)
    cargo_state[..., :3] = parent_pos + world_offset
    cargo_state[..., 3:7] = parent_quat
    cargo.write_object_link_pose_to_sim(cargo_state[..., :7], env_ids=env_ids)
    cargo.write_object_com_velocity_to_sim(
        torch.zeros((len(env_ids), cargo.num_objects, 6), device=cargo.device),
        env_ids=env_ids,
    )


def randomize_collection_physics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_scale_range: tuple[float, float],
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
) -> None:
    """Mass/material randomization for RigidObjectCollection assets."""
    collection: RigidObjectCollection = env.scene[asset_cfg.name]
    env_ids_gpu = _env_ids(env, env_ids, collection.device)
    view_ids = collection._env_obj_ids_to_view_ids(env_ids_gpu, slice(None)).cpu()

    masses = collection.root_physx_view.get_masses()
    default_masses = collection.reshape_data_to_view(collection.data.default_mass).cpu()
    scales = torch.empty_like(masses[view_ids]).uniform_(*mass_scale_range)
    masses[view_ids] = default_masses[view_ids] * scales
    collection.root_physx_view.set_masses(masses, view_ids)

    inertias = collection.root_physx_view.get_inertias()
    default_inertias = collection.reshape_data_to_view(collection.data.default_inertia).cpu()
    inertias[view_ids] = default_inertias[view_ids] * scales
    collection.root_physx_view.set_inertias(inertias, view_ids)

    materials = collection.root_physx_view.get_material_properties()
    count = len(view_ids)
    static = torch.empty((count, materials.shape[1]), device="cpu").uniform_(*static_friction_range)
    dynamic = torch.empty_like(static).uniform_(*dynamic_friction_range)
    dynamic = torch.minimum(dynamic, static)
    restitution = torch.empty_like(static).uniform_(*restitution_range)
    materials[view_ids] = torch.stack((static, dynamic, restitution), dim=-1)
    collection.root_physx_view.set_material_properties(materials, view_ids)


def randomize_box_flap_joint_friction(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    asset_names: tuple[str, ...],
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
) -> None:
    """Randomize PhysX static/dynamic friction for every box flap joint."""
    for asset_name in asset_names:
        box: Articulation = env.scene[asset_name]
        resolved_env_ids = _env_ids(env, env_ids, box.device)
        joint_ids, _ = box.find_joints("joint_(front|back|left|right)")
        shape = (len(resolved_env_ids), len(joint_ids))
        static = torch.empty(shape, device=box.device).uniform_(*static_friction_range)
        dynamic = torch.empty(shape, device=box.device).uniform_(*dynamic_friction_range)
        # PhysX requires static friction >= dynamic friction for every DOF.
        dynamic = torch.minimum(dynamic, static)
        box.write_joint_friction_coefficient_to_sim(
            static,
            joint_ids=joint_ids,
            env_ids=resolved_env_ids,
        )
        box.write_joint_dynamic_friction_coefficient_to_sim(
            dynamic,
            joint_ids=joint_ids,
            env_ids=resolved_env_ids,
        )


def reset_movers(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    human_cfg: SceneEntityCfg = SceneEntityCfg("moving_human"),
    mobile_cfg: SceneEntityCfg = SceneEntityCfg("moving_robot"),
) -> None:
    human = env.scene[human_cfg.name]
    env_ids = _env_ids(env, env_ids, human.device)
    if not hasattr(env, "_mover_phase"):
        env._mover_phase = torch.zeros((env.num_envs, 2), device=human.device)
        env._mover_speed = torch.ones((env.num_envs, 2), device=human.device)
        env._mover_offset = torch.zeros((env.num_envs, 2), device=human.device)
    difficulty = 0.35 + 0.65 * float(getattr(env, "_robustness_difficulty", 0.0))
    env._mover_phase[env_ids] = torch.rand((len(env_ids), 2), device=human.device)
    env._mover_speed[env_ids, 0] = difficulty * torch.empty(
        len(env_ids),
        device=human.device,
    ).uniform_(0.055, 0.11)
    env._mover_speed[env_ids, 1] = difficulty * torch.empty(
        len(env_ids),
        device=human.device,
    ).uniform_(0.07, 0.14)
    env._mover_offset[env_ids, 0] = difficulty * torch.empty(
        len(env_ids),
        device=human.device,
    ).uniform_(-0.12, 0.12)
    env._mover_offset[env_ids, 1] = difficulty * torch.empty(
        len(env_ids),
        device=human.device,
    ).uniform_(-0.16, 0.16)
    move_movers(env, env_ids, human_cfg, mobile_cfg)


def move_movers(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    human_cfg: SceneEntityCfg = SceneEntityCfg("moving_human"),
    mobile_cfg: SceneEntityCfg = SceneEntityCfg("moving_robot"),
) -> None:
    """Advance randomized periodic human and AMR trajectories."""
    human = env.scene[human_cfg.name]
    mobile = env.scene[mobile_cfg.name]
    env_ids = _env_ids(env, env_ids, human.device)
    if not hasattr(env, "_mover_phase"):
        reset_movers(env, env_ids, human_cfg, mobile_cfg)
        return

    time_s = env.episode_length_buf[env_ids].float() * env.step_dt
    human_angle = 2.0 * math.pi * (
        env._mover_speed[env_ids, 0] * time_s + env._mover_phase[env_ids, 0]
    )
    mobile_angle = 2.0 * math.pi * (
        env._mover_speed[env_ids, 1] * time_s + env._mover_phase[env_ids, 1]
    )

    human_pose = human.data.default_root_state[env_ids, :7].clone()
    human_pose[:, 0] = 0.55 + 1.05 * torch.sin(human_angle)
    human_pose[:, 1] = -1.03 + env._mover_offset[env_ids, 0]
    human_pose[:, 2] = 0.0
    human_yaw = torch.where(torch.cos(human_angle) >= 0.0, 0.0, torch.full_like(human_angle, math.pi))
    zeros = torch.zeros_like(human_yaw)
    human_pose[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, human_yaw)
    human_pose[:, :3] += env.scene.env_origins[env_ids]

    mobile_pose = mobile.data.default_root_state[env_ids, :7].clone()
    mobile_pose[:, 0] = 2.02 + env._mover_offset[env_ids, 1]
    mobile_pose[:, 1] = 0.05 + 1.12 * torch.sin(mobile_angle)
    mobile_pose[:, 2] = 0.0
    mobile_yaw = torch.where(
        torch.cos(mobile_angle) >= 0.0,
        torch.full_like(mobile_angle, math.pi / 2.0),
        torch.full_like(mobile_angle, -math.pi / 2.0),
    )
    mobile_pose[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, mobile_yaw)
    mobile_pose[:, :3] += env.scene.env_origins[env_ids]

    human.write_root_pose_to_sim(human_pose, env_ids=env_ids)
    mobile.write_root_pose_to_sim(mobile_pose, env_ids=env_ids)

    human_velocity = torch.zeros((len(env_ids), 6), device=human.device)
    human_velocity[:, 0] = (
        1.05 * 2.0 * math.pi * env._mover_speed[env_ids, 0] * torch.cos(human_angle)
    )
    mobile_velocity = torch.zeros((len(env_ids), 6), device=human.device)
    mobile_velocity[:, 1] = (
        1.12 * 2.0 * math.pi * env._mover_speed[env_ids, 1] * torch.cos(mobile_angle)
    )
    human.write_root_velocity_to_sim(human_velocity, env_ids=env_ids)
    mobile.write_root_velocity_to_sim(mobile_velocity, env_ids=env_ids)


def disturb_cargo(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("cargo"),
    linear_velocity_range: float = 0.035,
) -> None:
    """Apply a small impulse-like velocity disturbance to loose cargo."""
    cargo: RigidObjectCollection = env.scene[asset_cfg.name]
    env_ids = _env_ids(env, env_ids, cargo.device)
    difficulty = 0.20 + 0.80 * float(getattr(env, "_robustness_difficulty", 0.0))
    linear_velocity_range *= difficulty
    velocity = cargo.data.object_com_vel_w[env_ids].clone()
    velocity[..., :2] += torch.empty_like(velocity[..., :2]).uniform_(
        -linear_velocity_range,
        linear_velocity_range,
    )
    velocity[..., 2] += torch.empty_like(velocity[..., 2]).uniform_(-0.008, 0.012)
    cargo.write_object_com_velocity_to_sim(velocity, env_ids=env_ids)


def randomize_lighting(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    dome_intensity_range: tuple[float, float] = (350.0, 900.0),
    key_intensity_range: tuple[float, float] = (1300.0, 2900.0),
) -> None:
    """Randomize global factory illumination."""
    del env, env_ids
    stage = stage_utils.get_current_stage()
    for path, value_range in (
        ("/World/DomeLight", dome_intensity_range),
        ("/World/KeyLight", key_intensity_range),
    ):
        prim = stage.GetPrimAtPath(path)
        attr = prim.GetAttribute("inputs:intensity")
        if attr.IsValid():
            value = float(torch.empty(1).uniform_(*value_range).item())
            attr.Set(value)
