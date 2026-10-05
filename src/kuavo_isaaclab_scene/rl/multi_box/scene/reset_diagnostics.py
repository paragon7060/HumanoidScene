"""Read-only, bounded evidence from a requested reset before it respawns."""

from __future__ import annotations

import math

import torch

from ..geometry.pose import (
    normalize_quaternion, quat_apply, quat_conjugate, replace_invalid_poses,
)


def validate_reset_diagnostic_request(waves, *, enabled, training, steps):
    if not enabled:
        return
    if training or steps != 1 or not waves or any(w['split'] != 'validation' for w in waves):
        raise ValueError('Reset diagnostics require frozen DEV waves and --steps 1; no TRAIN or FINAL')


def _finite_values(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def passive_roller_snapshot(env):
    """Measure the support's hidden dynamic state separately from box roots."""
    result = {}
    for name, asset in env.scene.articulations.items():
        if not name.startswith('rack_roller_deck_'):
            continue
        velocity = asset.data.joint_vel.detach().cpu()
        result[name] = dict(joint_names=list(asset.joint_names),
            joint_positions_rad=[_finite_values(row) for row in asset.data.joint_pos.detach().cpu().tolist()],
            joint_velocities_radps=[_finite_values(row) for row in velocity.tolist()],
            maximum_absolute_joint_speed_radps=_finite_values(velocity.abs().amax(-1).tolist()))
    return result


def zero_passive_roller_velocities(env, env_ids):
    """Frozen diagnostic only: preserve poses/angles, remove inherited spin."""
    matched = []
    for name, asset in env.scene.articulations.items():
        if not name.startswith('rack_roller_deck_'):
            continue
        velocity = torch.zeros_like(asset.data.joint_vel[env_ids])
        asset.write_joint_velocity_to_sim(velocity, env_ids=env_ids)
        asset.set_joint_velocity_target(velocity, env_ids=env_ids)
        asset.set_joint_effort_target(velocity, env_ids=env_ids)
        matched.append(name)
    if not matched:
        raise ValueError('Passive roller reset probe requires the live rack roller decks')
    return matched


class ResetFailureCapture:
    """Keep only the first selected-box failure per requested environment.

    A partial respawn calls tracker.reset(), but must not erase this evidence
    or overwrite it with a replacement's failure. The owner explicitly opens
    and closes capture for one whole-layout settling attempt.
    """

    def __init__(self, env):
        self.seen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.records = []

    def record(self, tracker, logical, pose, velocity, type_id, region_id,
               newly_invalid, left_region, timed_out):
        ids = torch.where(newly_invalid & ~self.seen)[0]
        if not len(ids):
            return
        self.seen[ids] = True
        env = tracker.env
        rack = env.scene['rack'].data.root_pose_w[ids]
        robot = env.scene['robot'].data.root_pose_w[ids]
        selected_pose, invalid_box = replace_invalid_poses(pose[ids])
        safe_rack, invalid_rack = replace_invalid_poses(rack)
        local = quat_apply(quat_conjugate(normalize_quaternion(safe_rack[:, 3:])),
                           selected_pose[:, :3] - safe_rack[:, :3])
        pool = env._multi_box_pool_ids[ids, logical[ids]]
        # One transfer for the measured physical state, including raw invalid
        # values. JSON null denotes a nonfinite component, never a repair.
        physical = torch.cat((pose[ids], velocity[ids], rack, robot, local), -1).detach().cpu().tolist()
        labels = torch.stack((ids, logical[ids], pool, type_id[ids], region_id[ids],
                              tracker.invalid_count[ids]), -1).cpu().tolist()
        flags = torch.stack((left_region[ids], timed_out[ids],
            tracker.footprint_in_region[ids], tracker.on_assigned_shelf[ids],
            tracker.stable[ids], invalid_box, invalid_rack,
            ~torch.isfinite(velocity[ids]).all(-1)), -1).cpu().tolist()
        elapsed = tracker.elapsed[ids].cpu().tolist()
        for label, state, flag, seconds in zip(labels, physical, flags, elapsed):
            i, selected, physical_id, kind, region, count = label
            asset = env.scene[tracker.names[physical_id]]
            joints = getattr(asset.data, 'joint_pos', None)
            self.records.append(dict(environment=i, logical_id=selected,
                original_pool_id=physical_id, type_id=kind, region_id=region,
                invalid_count=count, common_step_counter=int(env.common_step_counter),
                elapsed_s=seconds, sample_phase='first_invalid_before_partial_respawn',
                left_region=flag[0], timed_out=flag[1], footprint_in_region=flag[2],
                on_assigned_shelf=flag[3], stable=flag[4], invalid_box_pose=flag[5],
                invalid_rack_pose=flag[6], nonfinite_box_velocity=flag[7],
                box_pose_world=_finite_values(state[:7]),
                box_velocity_world=_finite_values(state[7:13]),
                rack_pose_world=_finite_values(state[13:20]),
                robot_pose_world=_finite_values(state[20:27]),
                rack_local_root_xyz_m=None if flag[5] or flag[6] else _finite_values(state[27:]),
                joint_names=list(getattr(asset, 'joint_names', [])),
                joint_positions_rad=None if joints is None else _finite_values(joints[i].detach().cpu().tolist())))
