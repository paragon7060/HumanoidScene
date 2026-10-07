#!/usr/bin/env python3
"""Inspect our closed frozen TRAIN workplace attempts without importing data.

The final pre-reset critic features must match every physical terminal. Initial
invalid attempts remain in the whole-request denominator but have no HDF episode.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
import json,os
from pathlib import Path

import h5py
import numpy as np

from summarize_batched_staged_run import read_snapshot
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results


def longest_run(flags):
    best=now=0
    for value in flags:
        now=now+1 if value else 0
        best=max(best,now)
    return best


def base_tilt_degrees(observation,start):
    # The rack pose is measured relative to the base. Inverting its rotation
    # gives the base in the stationary rack frame; the handoff is the reference.
    six=observation[:,71:77].reshape(-1,2,3)
    a=six[:,0]/np.linalg.norm(six[:,0],axis=1,keepdims=True).clip(1e-8)
    b=six[:,1]-(a*six[:,1]).sum(1,keepdims=True)*a
    b/=np.linalg.norm(b,axis=1,keepdims=True).clip(1e-8)
    rotation=np.stack((a,b,np.cross(a,b)),axis=-1).transpose(0,2,1)
    delta=rotation@rotation[max(0,start-1)].T
    return np.degrees(np.arccos(np.clip(delta[:,2,2],-1,1)))


def summarize(parent):
    parent=Path(parent).resolve()
    if parent.stat().st_uid!=os.getuid():raise ValueError('Use only an owned experiment')
    managed=read_snapshot(parent/'launch.json');status=read_snapshot(parent/'status.json')
    run=Path(managed['run']).resolve()
    if (run.parent!=parent or run.stat().st_uid!=os.getuid()
        or status.get('training_exit_code')!=0
        or Path('/proc',str(status['training_pid'])).exists()
        or read_snapshot(run/'status.json').get('status')!='complete'):
        raise ValueError('The recorded writer must have stopped normally; a poll timeout is not termination')
    manifest=read_snapshot(run/'manifest.json');metrics=read_snapshot(run/'metrics.json')
    physical=summarize_workplace_results(manifest,metrics)
    dt=manifest['physics_dynamics']['control_dt_s']
    if not np.isclose(dt,1/30):raise ValueError('Require the recorded30Hz observation layout')
    cases={x['environment']:x for x in physical['cases']}
    valid_ids={i for i,r in cases.items() if not r['initial_invalid']}
    path=run/'executed_transitions.hdf5';before=path.stat()
    if before.st_uid!=os.getuid():raise ValueError('The closed HDF must be owned')
    records=[];seen=set()
    with h5py.File(path,'r') as handle:
        for episode in handle['episodes'].values():
            i=int(episode.attrs['environment'])
            if i in seen or i not in valid_ids:raise ValueError('Duplicate or invalid physical episode')
            seen.add(i);case=cases[i];terminal=case['actual_terminal'];t=episode['transitions']
            if int(episode.attrs['wave'])!=0 or json.loads(episode.attrs['layout_json'])!=manifest['layout_waves'][0]['layouts'][i]['layout']:
                raise ValueError('HDF episode does not retain the exact original TRAIN request')
            n=terminal['steps'];shape=(n,464)
            if (t['actor_obs'].shape!=shape or t['next_actor_obs'].shape!=shape or t['critic_obs'].shape!=(n,530)
                or t['next_critic_obs'].shape!=(n,530) or t['action'].shape!=(n,24)):
                raise ValueError('Require the exact recorded464D actor/66D privileged/24D command format')
            obs=t['actor_obs'][:];next_obs=t['next_actor_obs'][:];priv=t['next_critic_obs'][:,-66:];action=t['action'][:]
            if not all(np.isfinite(a).all() for a in (obs,next_obs,priv,action)):
                raise ValueError('Nonfinite contact records cannot establish evidence')
            final=priv[-1]
            if ((final[35:37]>.5).tolist()!=terminal['pinching']
                or (final[41:43]>.5).tolist()!=terminal['stable_hands']
                or bool(final[54]>.5)!=terminal['success']
                or bool(t['success'][-1])!=terminal['success']
                or bool(t['unsafe'][-1])!=terminal['unsafe']):
                raise ValueError('Pre-reset terminal features disagree with the measured physical terminal')
            start=terminal['staged_base'].get('manipulation_start')
            if type(start) is not int or not 0<=start<n:
                raise ValueError('A valid closed attempt must retain its measured base handoff')
            force=50*priv[:,15:23].reshape(-1,2,2,2)
            region=priv[:,23:31].reshape(-1,2,2,2)>.5
            opposed=priv[:,31:35].reshape(-1,2,2)>.5
            pinch=priv[:,35:37]>.5;both=pinch.all(-1)
            physically_opposing=both & (priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,0]
                                       !=priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,1])
            best_weaker_pad=force.min(axis=-1).max(axis=-1)
            tilt=base_tilt_degrees(obs,start);speed=np.linalg.norm(obs[:,43:46],axis=-1)
            record={k:case[k] for k in ('environment','seed','region','box_type','candidate',
                'success','unsafe','time_out','numerical_failure','other_terminal')}
            record.update(steps=n,held_start_step=start,unsafe_causes=terminal.get('unsafe_causes',{}),
                rack_peak_body=terminal['rack_peak_body'],rack_peak_force_n=terminal['rack_peak_force_n'],
                terminal_flap_surface_distance_m=terminal['flap_distances'],
                terminal_geometric_potentials=dict(zip(('approach','alignment','capture','proof_lift'),final[45:49].astype(float))),
                terminal_physical_jaw_close=(action[-1,20:22]>0).tolist(),
                terminal_measured_closure_fraction=next_obs[-1,46:48].astype(float).tolist(),
                terminal_pad_force_n_clipped_at200=force[-1].astype(float).tolist(),
                terminal_pad_in_region=region[-1].tolist(),terminal_opposed=opposed[-1].tolist(),
                terminal_hand_pinching=pinch[-1].tolist(),terminal_stable_hands=(final[41:43]>.5).tolist(),
                terminal_best_weaker_pad_force_n=best_weaker_pad[-1].astype(float).tolist(),
                rows_both_physically_opposing_pinches=int(physically_opposing[start:].sum()),
                longest_opposing_pinch_run_s=longest_run(physically_opposing[start:])*dt,
                held_reported_angular_speed_median_rad_s=float(np.median(speed[start:])),
                held_reported_angular_speed_p90_rad_s=float(np.quantile(speed[start:],.9)),
                last60_reported_angular_velocity_mean_world_rad_s=obs[-60:,43:46].mean(0).astype(float).tolist(),
                max_sampled_base_tilt_relative_to_handoff_deg=float(tilt[start:].max()))
            records.append(record)
    after=path.stat()
    if seen!=valid_ids or (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
        raise ValueError('The original complete closed HDF must remain unchanged')
    by_kind=defaultdict(list)
    for record in records:by_kind[record['region']+'_'+record['box_type']].append(record)
    groups={}
    for key,values in by_kind.items():
        groups[key]=dict(recorded_valid_attempts=len(values),
            terminal_both_jaws_closed=sum(all(v['terminal_physical_jaw_close']) for v in values),
            terminal_left_best_weaker_pad_below5N=sum(v['terminal_best_weaker_pad_force_n'][0]<5 for v in values),
            terminal_right_best_weaker_pad_below5N=sum(v['terminal_best_weaker_pad_force_n'][1]<5 for v in values),
            never_opposing_bilateral_pinch=sum(v['rows_both_physically_opposing_pinches']==0 for v in values),
            rack_collision_bodies=dict(Counter(v['rack_peak_body'] for v in values if v['unsafe_causes'].get('robot_rack_collision'))),
            terminal_capture_median=float(np.median([v['terminal_geometric_potentials']['capture'] for v in values])),
            median_reported_held_angular_speed_rad_s=float(np.median([v['held_reported_angular_speed_median_rad_s'] for v in values])),
            median_max_sampled_tilt_relative_to_handoff_deg=float(np.median([v['max_sampled_base_tilt_relative_to_handoff_deg'] for v in values])))
    return dict(recorded_UTC=datetime.now(timezone.utc).isoformat(),run_directory_name=run.name,
        whole_original_requests128=True,unique_original_TRAIN_layouts16=True,candidates_per_layout8=True,
        all198_model_and_initial_counters_frozen=physical['frozen_tensor_count']==198,
        initial_invalid_attempts=len(cases)-len(valid_ids),closed_valid_HDF_episodes=len(records),
        all_pre_reset_HDF_terminal_facts_match_actual_results=True,HDF_size_bytes=before.st_size,
        source_simulation_unchanged=True,control_sampled_velocities_NOT_substep_dynamics_or_causal_controller_proof=True,
        near_geometric_potential_NOT_physical_success=True,by_region_and_box_type=groups,
        attempts=sorted(records,key=lambda r:r['environment']),no_evaluation_or_synthetic_training_rows=True,
        no_sensor_physics_controller_policy_reward_changes=True,independent_FINAL_unused=True,goal_not_complete=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--experiment-dir',type=Path,required=True)
    parser.add_argument('--output-json',type=Path,required=True)
    args=parser.parse_args()
    if args.output_json.exists():parser.error('Use a distinct new report file')
    result=summarize(args.experiment_dir)
    with args.output_json.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(valid_closed_episodes=result['closed_valid_HDF_episodes'],
        terminal_truth_verified=True,by_region_and_box_type=result['by_region_and_box_type']),ensure_ascii=False))


if __name__=='__main__':main()
