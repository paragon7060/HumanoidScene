"""Read-only root state and commanded wrench at each frozen physics substep."""

import json
import math

import torch

from ..geometry.pose import quat_apply


FORMAT = 'frozen_TRAIN_base_state_and_commanded_wrench_substeps_v1'


def validate_base_substep_trace(indices, *, workplace, training, num_envs):
    if indices is None:
        return None
    if (training or num_envs != 128 or not workplace
            or workplace.get('name') != 'CPU_PhysX_frozen_TRAIN_workplace_search_v1'
            or workplace.get('original_candidate_requests') != 128
            or workplace.get('training') is not False):
        raise ValueError('Base substep trace requires the complete frozen TRAIN workplace search')
    if (not isinstance(indices, list) or not 1 <= len(indices) <= 8
            or any(type(i) is not int or not 0 <= i < num_envs for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('Select one to eight distinct existing environment indices')
    return dict(name=FORMAT, environment_indices=indices, frozen_TRAIN_only=True,
                sampling='before_each_physics_step_after_original_drive_apply',
                root_quaternion_order='wxyz', velocities_frame='world',
                torque_is_command_NOT_measured_reaction=True,
                physics_sensors_actions_and_policy_unchanged=True,
                Q_import_eligible=False, independent_FINAL_used=False,
                log_file='base_substep_trace.log')


def _finite_json(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class BaseSubstepTrace:
    """Observe the original drive; never advance physics or write robot state.

    Logging is enabled only around controlled steps of the frozen search. The
    raw PhysX view is read alongside IsaacLab's cache to distinguish dynamics
    from stale state. No contacts or new sensor reporters are requested.
    """

    def __init__(self, env, output, contract):
        self.env = env
        self.contract = contract
        self.drive = env.action_manager.get_term('base')._drive
        if self.drive is None:
            raise ValueError('Substep wrench trace requires the original dynamic base drive')
        self.ids = list(contract['environment_indices'])
        self.stream = (output / contract['log_file']).open('x')
        self.stream.write(json.dumps(dict(kind='contract', **contract)) + '\n')
        self.original = self.drive.apply
        self.context = None
        self.calls = 0
        self.rows = 0
        self.closed = False

        def observed_apply(velocity):
            result = self.original(velocity)
            self.calls += 1
            if self.context is not None:
                self.capture(velocity)
            return result

        self.observed_apply = observed_apply
        self.drive.apply = observed_apply

    def prepare(self, wave, control_step, active, stages, action):
        self.context = dict(wave=wave, control_step=control_step,
                            active=active, stages=stages, action=action)

    def finish_step(self):
        self.context = None
        self.stream.flush()

    def capture(self, velocity):
        context = self.context
        ids = [i for i in self.ids if bool(context['active'][i])]
        if not ids:
            return
        drive = self.drive
        asset = drive._asset
        data = asset.data
        pose = data.root_pose_w[ids]
        root_velocity = torch.cat((data.root_lin_vel_w[ids], data.root_ang_vel_w[ids]), -1)
        raw_pose = asset.root_physx_view.get_root_transforms()[ids]
        raw_pose = torch.cat((raw_pose[:, :3], raw_pose[:, 6:7], raw_pose[:, 3:6]), -1)
        raw_velocity = asset.root_physx_view.get_root_velocities()[ids]
        force_b = drive._force_b[ids, 0]
        torque_b = drive._torque_b[ids, 0]
        inertia = drive._inertia()[ids]
        tensors = dict(root_pose_world_wxyz=pose, root_velocity_world=root_velocity,
                       raw_PhysX_root_pose_world_wxyz=raw_pose,
                       raw_PhysX_root_COM_velocity_world=raw_velocity,
                       root_COM_pose_world_wxyz=data.root_com_pose_w[ids],
                       commanded_force_body_n=force_b, commanded_torque_body_nm=torque_b,
                       commanded_force_world_n=quat_apply(pose[:, 3:], force_b),
                       commanded_torque_world_nm=quat_apply(pose[:, 3:], torque_b),
                       estimated_world_inertia_diagonal_kg_m2=inertia,
                       total_mass_kg=drive._total_mass[ids],
                       root_target_xy_world_m=drive._target_xy[ids],
                       root_target_yaw_world_rad=drive._target_yaw[ids],
                       root_target_height_world_m=drive._target_height[ids],
                       level_target_quaternion_wxyz=drive._level_quat[ids],
                       processed_planar_velocity_command=velocity[ids],
                       executed_physical_action=context['action'][ids])
        values = _finite_json(tensors)
        finite = torch.stack([torch.isfinite(v).reshape(len(ids), -1).all(-1)
                              for v in tensors.values()]).all(0).tolist()
        for j, i in enumerate(ids):
            record = dict(kind='substep', environment=i, wave=context['wave'],
                          control_step=context['control_step'],
                          physics_apply_call=self.calls,
                          physics_dt_s=self.env.physics_dt,
                          simulation_timestamp_s=float(data._sim_timestamp),
                          phase=context['stages'][i].phase, all_fields_finite=finite[j],
                          root_quaternion_norm=float(pose[j, 3:].norm()),
                          **{k: v[j] for k, v in values.items()})
            self.stream.write(json.dumps(_finite_json(record), allow_nan=False,
                                         separators=(',', ':')) + '\n')
            self.rows += 1

    def close(self):
        if self.closed:
            return
        self.context = None
        self.drive.apply = self.original
        self.stream.write(json.dumps(dict(kind='closed', substep_records=self.rows,
                                         original_drive_apply_restored=True)) + '\n')
        self.stream.close()
        self.closed = True
