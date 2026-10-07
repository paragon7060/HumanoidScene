#!/usr/bin/env python3
"""Relate closed root substeps to real recorded observations, without training."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path

import h5py
import numpy as np

from prepare_size_workplaces import closed_probe
from summarize_batched_staged_run import read_snapshot


def multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), -1)


def conjugate(q):
    result = q.copy()
    result[..., 1:] *= -1
    return result


def rotation_velocity_world(quaternions, dt):
    """World rotation vector of successive wxyz poses; q and -q are equivalent."""
    q = np.asarray(quaternions, dtype=float)
    if q.ndim != 2 or q.shape[1] != 4 or len(q) < 2 or not np.isfinite(q).all() \
            or not math.isfinite(dt) or dt <= 0 or (np.linalg.norm(q, axis=-1) < 1e-8).any():
        raise ValueError('Finite nonzero successive quaternions and positive dt required')
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    delta = multiply(q[1:], conjugate(q[:-1]))
    delta = np.where(delta[:, :1] < 0, -delta, delta)
    length = np.linalg.norm(delta[:, 1:], axis=-1, keepdims=True)
    angle = 2*np.arctan2(length, np.clip(delta[:, :1], 0, 1))
    return delta[:, 1:] * np.divide(angle, length, out=np.full_like(length, 2.),
                                    where=length > 1e-12) / dt


def summarize(parent):
    parent = Path(parent).resolve()
    managed = read_snapshot(parent/'launch.json')
    status = read_snapshot(parent/'status.json')
    run = Path(managed['run']).resolve()
    if (parent.stat().st_uid != os.getuid() or run.parent != parent
            or run.stat().st_uid != os.getuid() or status.get('training_exit_code') != 0
            or Path('/proc', str(status['training_pid'])).exists()):
        raise ValueError('Use our normally closed physical writer; a poll timeout is not termination')
    physical, hashes, _ = closed_probe(run, allow_frozen_base_attitude_probe=True)
    manifest = read_snapshot(run/'manifest.json')
    contract = manifest.get('base_substep_trace')
    if not contract or contract.get('sampling') != 'before_each_physics_step_after_original_drive_apply':
        raise ValueError('Require the recorded unchanged-drive substep format')
    path = run/contract['log_file']
    before = path.stat()
    if before.st_uid != os.getuid():
        raise ValueError('The closed trace must be owned')
    records = [json.loads(line) for line in path.read_text().splitlines()]
    if (records[0].get('kind') != 'contract' or records[-1].get('kind') != 'closed'
            or not records[-1].get('original_drive_apply_restored')
            or records[-1]['substep_records'] != len(records)-2
            or any(r.get('kind') != 'substep' for r in records[1:-1])):
        raise ValueError('Require the complete closed trace, including its footer')
    dt = manifest['physics_dynamics']['physics_dt_s']
    decimation = round(manifest['physics_dynamics']['control_dt_s']/dt)
    cases = {r['environment']: r for r in physical['cases']}
    selected = contract['environment_indices']
    grouped = defaultdict(list)
    for r in records[1:-1]:
        i = r['environment']
        if i not in selected or r['wave'] != 0 or not math.isclose(r['physics_dt_s'],dt):
            raise ValueError('Every row must retain its original frozen TRAIN context')
        grouped[i].append(r)
    results = []
    hdf = run/'executed_transitions.hdf5'
    hdf_before = hdf.stat()
    if hdf_before.st_uid != os.getuid():
        raise ValueError('The closed source observations must be owned')
    with h5py.File(hdf, 'r') as handle:
        episodes = {int(e.attrs['environment']): e for e in handle['episodes'].values()}
        for i in selected:
            case = cases[i]
            if case['initial_invalid']:
                if grouped[i]:raise ValueError('An invalid original request cannot supply substep evidence')
                results.append(dict(environment=i, initial_invalid=True, physical_attempt_retained=True))
                continue
            rows = grouped[i]
            terminal = case['actual_terminal']
            n = terminal['steps']
            if len(rows) != n*decimation or [r['control_step'] for r in rows] != [s for s in range(n) for _ in range(decimation)]:
                raise ValueError('Require every substep of every original control step before termination')
            calls = np.array([r['physics_apply_call'] for r in rows])
            times = np.array([r['simulation_timestamp_s'] for r in rows])
            if not (np.diff(calls)==1).all() or not np.allclose(np.diff(times),dt,atol=1e-7,rtol=0):
                raise ValueError('Never infer derivatives across missing steps or a reset seam')
            finite = all(r['all_fields_finite'] for r in rows)
            result = {k:case[k] for k in ('environment','seed','region','box_type','candidate','success','unsafe','time_out')}
            result.update(original_control_steps=n, substeps=len(rows), all_fields_finite=finite,
                          nonfinite_rows=sum(not r['all_fields_finite'] for r in rows))
            if not finite:
                result['motion_inference_not_qualified']=True
                results.append(result)
                continue
            pose = np.array([r['root_pose_world_wxyz'] for r in rows])
            raw_pose = np.array([r['raw_PhysX_root_pose_world_wxyz'] for r in rows])
            velocity = np.array([r['root_velocity_world'] for r in rows])
            raw_velocity = np.array([r['raw_PhysX_root_COM_velocity_world'] for r in rows])
            observed = episodes[i]['transitions']['actor_obs'][:,43:46]
            if observed.shape != (n,3):
                raise ValueError('Original actor observations must retain every control step')
            error = np.max(np.abs(observed-velocity[::decimation,3:]))
            if error > 1e-5:
                raise ValueError('First substep twist must match the actual original actor observation')
            held = np.array([r['phase']=='held_grasp' for r in rows])
            contiguous_held = held[:-1]&held[1:]
            motion = rotation_velocity_world(pose[:,3:],dt)[contiguous_held]
            measured = velocity[:-1,3:][contiguous_held]
            next_measured = velocity[1:,3:][contiguous_held]
            torque = np.array([r['commanded_torque_world_nm'] for r in rows])[held]
            result.update(actor_observation_twist_max_abs_error_rad_s=float(error),
                cached_raw_position_max_difference_m=float(np.linalg.norm(pose[:,:3]-raw_pose[:,:3],axis=-1).max()),
                cached_raw_world_velocity_max_abs_difference=float(np.abs(velocity-raw_velocity).max()),
                cached_raw_quaternion_same_up_to_sign=bool(np.all(np.minimum(np.linalg.norm(pose[:,3:]-raw_pose[:,3:],axis=-1),np.linalg.norm(pose[:,3:]+raw_pose[:,3:],axis=-1))<1e-5)),
                held_contiguous_pose_derivatives=len(motion))
            if not len(motion):
                result['held_motion_inference_not_qualified'] = 'no_contiguous_held_substeps'
                results.append(result)
                continue
            axis=int(np.argmax(np.median(np.abs(measured),axis=0)))
            high=(np.abs(measured[:,axis])>.1)&(np.abs(next_measured[:,axis])>.1)
            result.update(
                held_dominant_world_angular_axis='xyz'[axis],
                held_high_speed_adjacent_substeps=int(high.sum()),
                held_high_speed_sign_flip_fraction=(float(np.mean(
                    measured[high,axis]*next_measured[high,axis]<0)) if high.any() else None),
                held_angular_velocity_increment_norm_median_rad_s2=float(
                    np.median(np.linalg.norm(next_measured-measured,axis=-1))/dt),
                held_actual_velocity_norm_median_rad_s=float(np.median(np.linalg.norm(measured,axis=-1))),
                held_pose_derived_velocity_norm_median_rad_s=float(np.median(np.linalg.norm(motion,axis=-1))),
                held_actual_velocity_mean_world_rad_s=measured.mean(0).tolist(),
                held_pose_derived_velocity_mean_world_rad_s=motion.mean(0).tolist(),
                held_velocity_pose_derivative_difference_RMS_rad_s=float(np.sqrt(np.mean((measured-motion)**2))),
                held_next_velocity_pose_derivative_difference_RMS_rad_s=float(np.sqrt(np.mean((next_measured-motion)**2))),
                derivative_compared_with_both_endpoint_velocities=True,
                held_commanded_torque_world_median_nm=np.median(torque,axis=0).tolist(),
                held_commanded_torque_world_max_abs_nm=np.abs(torque).max(0).tolist())
            results.append(result)
    after = path.stat();hdf_after=hdf.stat()
    for a,b in ((before,after),(hdf_before,hdf_after)):
        if (a.st_ino,a.st_size,a.st_mtime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns):
            raise ValueError('Original closed trace and HDF must remain unchanged')
    return dict(recorded_UTC=datetime.now(timezone.utc).isoformat(),run_directory_name=run.name,
        whole_original_TRAIN_requests128=True,selected_root_diagnostics_NOT_all_environment_causality=True,
        all198_model_tensors_and_initial_counters_frozen=physical['frozen_tensor_count']==198,
        source_checkpoint_SHA256=hashes['checkpoint'],source_metrics_SHA256=hashes['metrics'],
        base_attitude_gain_probe=manifest.get('base_attitude_gain_probe'),
        selected_environments=selected,results=results,
        actual_actor_observations_cross_checked=True,original_trace_and_HDF_unchanged=True,
        commanded_torque_NOT_measured_reaction=True,causal_control_fix_NOT_proven=True,
        no_Q_or_policy_training_or_synthetic_experience=True,independent_FINAL_unused=True,
        goal_not_complete=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():parser.error('Use a distinct new report file')
    report=summarize(args.experiment_dir)
    with args.output_json.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(selected_environments=report['selected_environments'],
        closed_real_observation_twist_verified=True,goal_not_complete=True)))


if __name__=='__main__':main()
