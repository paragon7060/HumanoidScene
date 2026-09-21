"""Reset-time application of v2 spawn plans to Isaac Lab assets."""

from __future__ import annotations

import torch
from isaaclab.envs import mdp as base_mdp
from isaaclab.utils.math import quat_apply, quat_from_euler_xyz, quat_mul

from .assets import CONVEYOR_PART_NAMES, RACK_ROLLER_ASSET_NAMES
from .spawn import SpawnBatch, physical_asset_names, sample_spawn_batch
from ....workcell.workcell_layout import position, scale


SPAWN_BUFFER_FIELDS = (
    "counts",
    "active",
    "box_type_ids",
    "region_ids",
    "pool_ids",
    "rack_local_positions",
    "rack_local_quaternions",
    "rack_xy_delta",
    "rack_yaw_delta",
    "conveyor_xy_delta",
    "conveyor_yaw_delta",
)


def _env_ids(env, env_ids):
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    return torch.as_tensor(env_ids, device=env.device, dtype=torch.long)


def _yaw_quaternion(yaw):
    zeros = torch.zeros_like(yaw)
    return quat_from_euler_xyz(zeros, zeros, yaw)


def _default_pose(asset, env, ids):
    pose = asset.data.default_root_state[ids, :7].clone()
    pose[:, :3] += env.scene.env_origins[ids]
    return pose


def _remember(env, ids, batch: SpawnBatch):
    for field in SPAWN_BUFFER_FIELDS:
        value = getattr(batch, field)
        name = f"_multi_box_{field}"
        target = getattr(env, name, None)
        expected = (env.num_envs, *value.shape[1:])
        if target is None or target.shape != expected or target.dtype != value.dtype:
            target = torch.zeros(expected, dtype=value.dtype, device=env.device)
            if value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
                target.fill_(-1)
            setattr(env, name, target)
        target[ids] = value


def _move_rack(env, ids, batch):
    rack = env.scene["rack"]
    pose = _default_pose(rack, env, ids)
    pose[:, :2] += batch.rack_xy_delta
    delta_q = _yaw_quaternion(batch.rack_yaw_delta)
    pose[:, 3:] = quat_mul(delta_q, pose[:, 3:])
    rack.write_root_pose_to_sim(pose, env_ids=ids)
    for name in RACK_ROLLER_ASSET_NAMES:
        if getattr(env.cfg.scene, name, None) is None:
            continue
        deck = env.scene[name]
        default = _default_pose(deck, env, ids)
        rack_default = _default_pose(rack, env, ids)
        relative = default[:, :3] - rack_default[:, :3]
        deck_pose = default
        deck_pose[:, :3] = pose[:, :3] + quat_apply(delta_q, relative)
        deck_pose[:, 3:] = quat_mul(delta_q, default[:, 3:])
        deck.write_root_pose_to_sim(deck_pose, env_ids=ids)
    return pose


def _move_conveyor(env, ids, batch):
    anchor = torch.tensor(position("conveyor"), dtype=torch.float32, device=env.device)
    delta_q = _yaw_quaternion(batch.conveyor_yaw_delta)
    origins = env.scene.env_origins[ids]
    for name in CONVEYOR_PART_NAMES:
        asset = env.scene[name]
        default = asset.data.default_root_state[ids, :7].clone()
        relative = default[:, :3] - anchor
        pose = default
        pose[:, :3] = origins + anchor + torch.cat(
            (batch.conveyor_xy_delta, torch.zeros(len(ids), 1, device=env.device)), -1)
        pose[:, :3] += quat_apply(delta_q, relative)
        pose[:, 3:] = quat_mul(delta_q, default[:, 3:])
        asset.write_root_pose_to_sim(pose, env_ids=ids)


def _move_active_boxes(env, ids, batch, rack_pose):
    names = physical_asset_names()
    for pool_id, name in enumerate(names):
        matches = batch.active & (batch.pool_ids == pool_id)
        selected = matches.nonzero(as_tuple=False)
        if not len(selected):
            continue
        local_env_ids = selected[:, 0]
        logical_ids = selected[:, 1]
        scene_env_ids = ids[local_env_ids]
        local_position = batch.rack_local_positions[local_env_ids, logical_ids]
        local_quaternion = batch.rack_local_quaternions[local_env_ids, logical_ids]
        root = env.scene[name].data.default_root_state[scene_env_ids].clone()
        root[:, :3] = rack_pose[local_env_ids, :3] + quat_apply(
            rack_pose[local_env_ids, 3:], local_position)
        root[:, 3:7] = quat_mul(rack_pose[local_env_ids, 3:], local_quaternion)
        root[:, 7:] = 0.0
        env.scene[name].write_root_state_to_sim(root, env_ids=scene_env_ids)


def reset_randomized_scene(env, env_ids):
    """Reset dynamics, randomize workcell anchors, and activate the sampled box subset."""
    ids = _env_ids(env, env_ids)
    if not len(ids):
        return
    base_mdp.reset_scene_to_default(env, ids, reset_joint_targets=True)
    batch = sample_spawn_batch(
        env.cfg.multi_box, len(ids), device=env.device, rack_scale=scale("rack"))
    rack_pose = _move_rack(env, ids, batch)
    _move_conveyor(env, ids, batch)
    _move_active_boxes(env, ids, batch, rack_pose)
    _remember(env, ids, batch)
    privileged_grasp = getattr(env, "_multi_box_privileged_grasp", None)
    if privileged_grasp is not None:
        privileged_grasp.reset(ids)
    # A post-reset critic observation may be requested before the global step
    # counter advances. Invalidate shared manager caches so it cannot receive
    # the preceding episode's privileged tensors.
    env._multi_box_privileged_grasp_counter = -1
    env._multi_box_grasp_safety_counter = -1
