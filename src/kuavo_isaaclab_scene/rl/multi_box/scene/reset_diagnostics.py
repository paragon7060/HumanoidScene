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


def validate_rear5_support_gap_diagnostic_request(waves, *, gap_m, reset_enabled, training, steps, other_probe):
    if gap_m is None:
        return
    from ....workcell.rack_box_layout import RACK_SURFACE_CLEARANCE_M
    if not reset_enabled or other_probe:
        raise ValueError('Rear5 support-gap probe requires an otherwise unchanged reset diagnostic')
    validate_reset_diagnostic_request(waves, enabled=True, training=training, steps=steps)
    if not math.isfinite(gap_m) or not 0 <= gap_m <= RACK_SURFACE_CLEARANCE_M:
        raise ValueError('Rear5 support gap must be finite and between0 and the default spawn clearance')


def configure_reset_solver_probe(cfg, waves, *, solver, reset_enabled, training, steps, other_probe=False):
    """Change only the solver for a frozen startup audit, never a training MDP."""
    if solver is None:
        return None
    if solver not in ('PGS', 'TGS') or not reset_enabled or other_probe:
        raise ValueError('Reset solver probe requires an otherwise unchanged frozen reset diagnostic')
    validate_reset_diagnostic_request(waves, enabled=True, training=training, steps=steps)
    original = cfg.sim.physx.solver_type
    if original not in (0, 1):
        raise ValueError('Unknown source physics solver')
    cfg.sim.physx.solver_type = {'PGS': 0, 'TGS': 1}[solver]
    return dict(name='reset_solver_only', source_solver={0: 'PGS', 1: 'TGS'}[original],
                requested_solver=solver, frozen_only=True, Q_import_eligible=False,
                physics_dt_s=cfg.sim.dt, control_dt_s=cfg.sim.dt * cfg.decimation,
                all_iteration_counts_and_limits_unchanged=True,
                box_base_poses_randomization_success_and_safety_unchanged=True)


def _finite_values(values):
    return [float(v) if math.isfinite(float(v)) else None for v in values]


def extend_startup_contact_pair_filters(scene, sensor_names, targets):
    """Extend existing reporters before creation; never touch collision rules."""
    result = {}
    for name in sensor_names:
        cfg = getattr(scene, name)
        cfg.filter_prim_paths_expr = list(dict.fromkeys([*cfg.filter_prim_paths_expr, *targets]))
        result[name] = list(cfg.filter_prim_paths_expr)
    return result


def strongest_normal_contact_pairs(matrix, target_paths, *, count=3):
    """Describe the largest reported pairs, retaining nonfinite evidence."""
    if matrix.ndim != 3 or matrix.shape[-1] != 3 or matrix.shape[1] != len(target_paths):
        raise ValueError('Contact matrix filter axis does not match declared rigid-body targets')
    magnitude = matrix.norm(dim=-1)
    values, indices = magnitude.topk(min(count, len(target_paths)), dim=-1)
    forces = matrix.gather(1, indices[..., None].expand(-1, -1, 3)).detach().cpu().tolist()
    nonfinite = (~torch.isfinite(matrix).all(-1)).sum(-1).cpu().tolist()
    result = []
    for sizes, ids, vectors, invalid_count in zip(values.cpu().tolist(), indices.cpu().tolist(), forces, nonfinite, strict=True):
        pairs = [dict(target_path=target_paths[index], normal_force_world_n=_finite_values(force),
                      normal_force_magnitude_n=_finite_values([size])[0])
                 for size, index, force in zip(sizes, ids, vectors, strict=True)
                 if not math.isfinite(size) or size > 0]
        result.append(dict(strongest_normal_pairs=pairs, nonfinite_filter_count=invalid_count))
    return result


def contact_target_state_lookup(env):
    """CPU evidence from existing live bodies; no USD pose fallback."""
    result = {}
    for name, asset in env.scene.articulations.items():
        prefix = asset.cfg.prim_path
        pose = asset.data.body_link_pose_w.detach().cpu().tolist()
        velocity = asset.data.body_link_vel_w.detach().cpu().tolist()
        for j, body in enumerate(asset.body_names):
            result[(prefix, body)] = dict(asset=name, body=body,
                pose=[row[j] for row in pose], velocity=[row[j] for row in velocity])
    for name, asset in env.scene.rigid_objects.items():
        result[(asset.cfg.prim_path, None)] = dict(asset=name, body=None,
            pose=asset.data.root_pose_w.detach().cpu().tolist(),
            velocity=asset.data.root_vel_w.detach().cpu().tolist())
    return result


def attach_contact_target_states(pair_rows, environment_ids, lookup):
    for row, i in zip(pair_rows, environment_ids, strict=True):
        for pair in row['strongest_normal_pairs']:
            path = pair['target_path']; leaf = path.rsplit('/', 1)[-1]
            candidates = [state for (prefix, body), state in lookup.items()
                          if (body is None and path == prefix) or
                          (body == leaf and path.startswith(prefix + '/'))]
            if len(candidates) != 1:
                pair['target_pose_resolution'] = 'unresolved_or_ambiguous'
                continue
            state = candidates[0]
            pair.update(target_asset=state['asset'], target_body=state['body'],
                target_pose_world=_finite_values(state['pose'][i]),
                target_velocity_world=_finite_values(state['velocity'][i]),
                target_pose_resolution='measured_live_body_link_or_rigid_root')


def support_root_snapshot(env):
    """Observe fixed support base motion and center drift during first contact."""
    result = {}
    baseline = getattr(env, '_reset_support_initial_centers', None)
    if baseline is None:
        baseline = {name: asset.data.body_link_pose_w[..., :3].clone()
                    for name, asset in env.scene.articulations.items() if name.startswith('rack_roller_deck_')}
        env._reset_support_initial_centers = baseline
    for name, asset in env.scene.articulations.items():
        if name not in baseline:
            continue
        base = asset.body_names.index('Base')
        displacement = (asset.data.body_link_pose_w[..., :3] - baseline[name]).norm(dim=-1).amax(-1)
        result[name] = dict(is_fixed_base=asset.is_fixed_base,
            base_pose_world=[_finite_values(row) for row in asset.data.body_link_pose_w[:, base].cpu().tolist()],
            base_velocity_world=[_finite_values(row) for row in asset.data.body_link_vel_w[:, base].cpu().tolist()],
            maximum_link_center_displacement_m=_finite_values(displacement.cpu().tolist()))
    return result


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


def startup_normal_contact_snapshot(env, asset_names, box_sensor_names, robot_sensor_names):
    """Read existing normal-contact reporters; no new sensor or physics writes.

    Net normal forces do not identify the other collider and do not contain
    tangential friction. Keep that limitation explicit in the saved evidence.
    """
    active = env._multi_box_active.detach().cpu()
    pools = env._multi_box_pool_ids.detach().cpu()
    pairs_enabled = getattr(env, '_reset_contact_pair_diagnostics', False)
    target_states = contact_target_state_lookup(env) if pairs_enabled else None
    rows = []
    for pool, (name, sensor_name) in enumerate(zip(asset_names, box_sensor_names, strict=True)):
        selected = torch.nonzero(active & (pools == pool), as_tuple=False)
        if not len(selected):
            continue
        ids = selected[:, 0].to(env.device)
        asset, sensor = env.scene[name], env.scene[sensor_name]
        force = sensor.data.net_forces_w[ids]
        if force.shape[1] != 1 or len(sensor.body_names) != 1:
            raise ValueError('Startup contact trace needs one box-body reporter per physical asset')
        body_velocity = asset.data.body_link_vel_w[ids]
        values = torch.cat((asset.data.root_vel_w[ids], force[:, 0],
            asset.data.joint_vel[ids].abs().amax(-1, keepdim=True),
            body_velocity[..., :3].norm(dim=-1).amax(-1, keepdim=True),
            body_velocity[..., 3:].norm(dim=-1).amax(-1, keepdim=True)), -1).detach().cpu().tolist()
        pair_rows = [{} for _ in values]
        if pairs_enabled:
            pair_rows = strongest_normal_contact_pairs(sensor.data.force_matrix_w[ids, 0], sensor.cfg.filter_prim_paths_expr)
            attach_contact_target_states(pair_rows, selected[:, 0].tolist(), target_states)
        for (i, logical), value, pair_row in zip(selected.tolist(), values, pair_rows, strict=True):
            rows.append(dict(environment=i, logical_id=logical, original_pool_id=pool,
                source_body=sensor.body_names[0], box_velocity_world=_finite_values(value[:6]),
                normal_contact_force_world_n=_finite_values(value[6:9]),
                maximum_absolute_flap_joint_speed_radps=_finite_values(value[9:10])[0],
                maximum_link_linear_speed_mps=_finite_values(value[10:11])[0],
                maximum_link_angular_speed_radps=_finite_values(value[11:12])[0], **pair_row))
    robot = {}
    for name in robot_sensor_names:
        sensor = env.scene[name]
        force = sensor.data.net_forces_w.detach().cpu()
        robot[name] = dict(source_bodies=list(sensor.body_names),
            normal_contact_force_world_n=[[_finite_values(body) for body in row] for row in force.tolist()])
    rollers = {name: _finite_values(asset.data.joint_vel.abs().amax(-1).detach().cpu().tolist())
        for name, asset in env.scene.articulations.items() if name.startswith('rack_roller_deck_')}
    result = dict(boxes=rows, robot=robot, roller_maximum_absolute_speed_radps=rollers,
        measurement='net normal contact force on source bodies; collider identity and tangential friction are not measured')
    if pairs_enabled:
        result.update(support_roots=support_root_snapshot(env),
            measurement='normal contact forces per declared rigid-body filter, plus measured target poses; tangential friction not measured')
    return result


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
