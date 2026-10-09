"""Reconstruct closed TRAIN contact attempts from a verified readonly replay copy.

No active HDF/GPU replay reads, policy updates or imports into training occur.
Physical contact labels are diagnostic evidence only, never explorer inputs.
"""
import argparse,os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime
import hashlib,json,sys
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/rl')]
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import PerceivedContactExploration
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import planar_position
from summarize_batched_staged_run import supported_success
from audit_closed_dev_critics import identity, require_closed_run
from export_eval_q_videos import sha256
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
source=parser.add_mutually_exclusive_group(required=True)
source.add_argument('--snapshot-proof',type=Path)
source.add_argument('--closed-run',type=Path,
 help='Inspect an owned, complete original replay after both writers stopped and final checkpoint/log backup was verified; no disk copy is created')
parser.add_argument('--output-json',type=Path,required=True)
parser.add_argument('--summary-json',type=Path)
parser.add_argument('--upright-phase-only',action='store_true',
 help='Validate actual jaw sequence, phase counters and episode mapping for upright data; continuous IK goal equality remains unverified')
args=parser.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only analysis requires empty CUDA_VISIBLE_DEVICES')
if args.output_json.exists() or (args.summary_json is not None and args.summary_json.exists()):raise ValueError('Use unique new output files')
if args.closed_run is not None:
 dest=args.closed_run.resolve()
 require_closed_run(dest)
 path=dest/'staged_goal_experience.pt'
 replay_identity=identity(path)
 m=json.loads((dest/'metrics.json').read_text())
 waves=sorted({r['wave'] for r in m['outcomes'] if r['split']=='train'})
 proof=dict(snapshot=str(path),snapshot_SHA256=sha256(path),source_run=str(dest),
  completed_TRAIN_waves=waves,completed_online_rows=m['learner']['online_rows'])
else:
 dest=args.snapshot_proof.parent
 proof=json.loads(args.snapshot_proof.read_text());path=Path(proof['snapshot'])
 replay_identity=identity(path)
 m=json.loads((dest/'completed_waves_metrics.json').read_text())
assert path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid()
sha=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
assert sha.hexdigest()==proof['snapshot_SHA256']
assert args.closed_run is not None or path.stat().st_mode&0o222==0
s=torch.load(path,map_location='cpu',weights_only=args.closed_run is not None);data=s['executed_goal_transitions']
assert len(data['reward'])==proof['completed_online_rows'] and s['perceived_contact_statistics']==m['learner']['perceived_contact_statistics']
assert all(torch.isfinite(t).all() for t in data.values())
coordinates=PoseGoalCoordinates(exact_projected_base=True)
contract=s['goal_contract'];pilot=SimpleNamespace(coordinates=coordinates,center=torch.tensor(contract['goal_center']),scale=torch.tensor(contract['goal_scale']),agent=SimpleNamespace(action_projector=AbsoluteGoalJawProjector()))
variant=contract['TRAIN_perceived_contact_exploration']['name']
assert variant in ('TRAIN_arm20_perceived_contact_attempt_v1','TRAIN_arm20_perceived_settled_contact_attempt_v2','TRAIN_arm20_perceived_precise_feedback_attempt_v3','TRAIN_arm20_perceived_motion_feedback_attempt_v4','TRAIN_arm20_upright_contact_attempt_v5','TRAIN_arm20_upright_interior_contact_attempt_v6','TRAIN_arm20_upright_predictive_contact_attempt_v7','TRAIN_arm20_upright_whole_arm_rack_clearance_attempt_v8')
clearance=variant.endswith('_v8');predictive=clearance or variant.endswith('_v7');interior=variant.endswith('_v6');upright=predictive or interior or variant.endswith('_v5');settled=not variant.endswith('_v1');motion=upright or variant.endswith('_v4');precise=motion or variant.endswith('_v3')
if args.upright_phase_only and not upright:
 raise ValueError('Phase-only diagnosis is restricted to declared upright v5/v6/v7/v8 data')
if upright:
 from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import upright_support_affine
 pilot.agent.upright_support_anchor,pilot.agent.upright_support_scale=upright_support_affine(pilot.center,pilot.scale,coordinates.links)
waves=proof['completed_TRAIN_waves']
assert waves and len(waves)==len(set(waves))
offset=0;previous_stats=None;records=[];max_error=0.;measured_terminals=0
for wave in waves:
 cases=sorted([r for r in m['outcomes'] if r['wave']==wave],key=lambda r:r['environment'])
 assert len(cases)==128 and {r['environment'] for r in cases}==set(range(128))
 assert all(r['complete'] and not r['result'].get('numerical_failure') for r in cases)
 if args.closed_run is not None:
  manifest=json.loads((dest/'manifest.json').read_text())
  expected={'shelf_2_left/small':16,'shelf_2_right/small':16,
   'shelf_2_left/medium':16,'shelf_2_right/medium':16,'shelf_3_left/small':32,'shelf_3_right/small':32}
  assert Counter(r['layout']['target_region']+'/'+r['layout']['target_box_type'] for r in cases)==expected
  assert manifest['layout_waves'][wave]['split']=='train'
  assert all(r['layout']==manifest['layout_waves'][wave]['layouts'][r['environment']]['layout'] for r in cases)
 for r in cases:
  if not r['initial_layout_valid']:
   assert r['executed_transition_rows']==0 and r['result']['steps']==0 and not supported_success(r)
   assert r.get('collection_policy_mode')=='no_held_mode_draw'
 starts=np.array([r['result']['staged_base']['manipulation_start'] if r['initial_layout_valid'] else 0 for r in cases]);ends=np.array([r['result']['steps'] for r in cases]);chosen=torch.tensor([r['collection_policy_mode']=='perceived_contact_exploration' for r in cases])
 assert all(type(x) is int for x in starts.tolist())
 tracker=None;indices=[[] for _ in cases];phases=[[] for _ in cases];events=[[] for _ in cases];close_gates=[[] for _ in cases];reopens=[[] for _ in cases]
 for step in range(int(ends.max())):
  ids=torch.from_numpy(np.where((starts<=step)&(ends>step))[0]);n=len(ids)
  if not n:continue
  ix=torch.arange(offset,offset+n);offset+=n
  raw=data['critic_obs'][ix,:464];ao=data['actor_obs'][ix];co=data['critic_obs'][ix];goals=data['action'][ix];supp=co[:,533:571];clocks=step-torch.from_numpy(starts[ids.numpy()])
  assert torch.equal(ao[:,:86],raw[:,:86]) and torch.equal(ao[:,474:512],supp)
  assert torch.allclose(co[:,530],clocks.to(co)/900,atol=1e-7,rtol=0)
  xy=raw.new_tensor([cases[i]['result']['staged_base']['base_target_xy_rack_m'] for i in ids]);yaw=raw.new_tensor([cases[i]['result']['staged_base']['base_target_yaw_rack_rad'] for i in ids])
  assert torch.equal(ao[:,-5:-3],xy) and torch.allclose(ao[:,-3],yaw.sin(),atol=1e-7,rtol=0) and torch.allclose(ao[:,-2],yaw.cos(),atol=1e-7,rtol=0)
  pilot.stage=SimpleNamespace(target_xy=xy,target_yaw=yaw)
  command=held_goal_coordinates(coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage)
  if tracker is None:tracker=PerceivedContactExploration(128,raw,previous_stats,settled_close=settled,precise_feedback=precise,motion_feedback=motion,upright_feedback=upright,interior_contact=interior,predictive_feedback=predictive,whole_arm_clearance=clearance)
  pre=tracker.phase[ids].clone()
  pre_closed=tracker.closing[ids].clone()
  pre_closed_ticks=tracker.closed_ticks[ids].clone()
  pre_settled_ticks=tracker.settled_ticks[ids].clone()
  proposed,(_,_,recomputed)=tracker.step(pilot,raw,(command,(ao,torch.full_like(co,float('nan')),goals)),ids,supp,chosen[ids],clocks)
  error=float((recomputed-goals).abs().max());max_error=max(max_error,error)
  if args.upright_phase_only:
   # IK support-boundary decisions can diverge offline even though the exact
   # observed jaw sequence and every integer phase counter still agree. Use
   # recorded body goals below; never claim the continuous trace was verified.
   assert torch.equal(recomputed[:,19:21],goals[:,19:21]),('Actual jaw sequence differs',wave,step)
  else:
   assert torch.allclose(recomputed,goals,atol=3e-5,rtol=0),(wave,step,error)
  closed=(goals[:,19:21]>0).all(-1)&chosen[ids]&((tracker.phase[ids]==1)|(tracker.phase[ids]==2))
  if closed.any():
   relation=supp[:,:36].reshape(n,2,2,9);flaps=torch.stack((tracker.assignment[ids],1-tracker.assignment[ids]),-1)
   selected=relation[torch.arange(n)[:,None],torch.arange(2)[None],flaps]
   relative_R=_rotation_matrix(selected[...,3:]);point_hand=selected[...,:3]+(relative_R@tracker.contact_offsets[ids,...,None]).squeeze(-1)
   current_distance=point_hand.norm(dim=-1).amax(-1)
   rack_R=_rotation_matrix(raw[:,71:77]);fixed=raw[:,None,68:71]+(rack_R[:,None]@tracker.close_targets_rack[ids,...,None]).squeeze(-1)
   fixed_distance=(fixed-raw[:,50:68].reshape(n,2,9)[...,:3]).norm(dim=-1).amax(-1)
   axis_error=torch.acos((relative_R[...,0]*tracker.axes[None]).sum(-1).abs().clamp(0,1)).amax(-1)
   for j in torch.where(closed)[0].tolist():
    close_gates[int(ids[j])].append(dict(closure=raw[j,46:48].tolist(),closed_ticks=int(tracker.closed_ticks[ids[j]]),settled_ticks=int(tracker.settled_ticks[ids[j]]),current_point_distance_m=float(current_distance[j]),fixed_point_distance_m=(float(fixed_distance[j]) if settled and not precise else None),axis_error_rad=float(axis_error[j])))
  for j,i in enumerate(ids.tolist()):
   indices[i].append(int(ix[j]));phases[i].append(int(tracker.phase[i]))
   if pre[j]==1 and tracker.phase[i]==2:
    events[i].append(dict(control_step=step+1,held_clock=int(clocks[j]),measured_closure_fraction=raw[j,46:48].tolist(),physical_close_command=command[j,20:22].tolist(),actual_pre_pinching=(co[j,499:501]>.5).tolist(),actual_next_pinching=(data['next_critic_obs'][ix[j],499:501]>.5).tolist(),actual_pre_stable=(co[j,505:507]>.5).tolist(),actual_pre_weaker_pad_force_n=(50*co[j,479:487].reshape(2,2,2)).min(-1).values.max(-1).values.tolist(),actual_pre_pad_in_region=(co[j,487:495].reshape(2,2,2)>.5).tolist(),actual_pre_opposed=(co[j,495:499].reshape(2,2)>.5).tolist()))
  if precise:
   reopened=chosen[ids]&pre_closed&(pre==1)&(tracker.phase[ids]==1)&~tracker.closing[ids]
   if reopened.any():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cartesian_flap_probe import contact_region_offsets
    from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
    from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import nominal_flap_geometry
    relation=supp[:,:36].reshape(n,2,2,9);flaps=torch.stack((tracker.assignment[ids],1-tracker.assignment[ids]),-1)
    selected=relation[torch.arange(n)[:,None],torch.arange(2)[None],flaps]
    relative_R=_rotation_matrix(selected[...,3:]);point=selected[...,:3]+(relative_R@tracker.contact_offsets[ids,...,None]).squeeze(-1)
    token,valid=target_token(raw);assert valid[reopened].all()
    _,halves,_=nominal_flap_geometry(token[:,5:8],token[:,3:5].argmax(-1))
    nearest_offsets=contact_region_offsets(selected,halves[torch.arange(n)[:,None],flaps])
    nearest=selected[...,:3]+(relative_R@nearest_offsets[...,None]).squeeze(-1)
    angle=torch.acos((relative_R[...,0]*tracker.axes[None]).sum(-1).abs().clamp(0,1))
    for j in torch.where(reopened)[0].tolist():
     # Include the complete immediately preceding closed-command interval.
     # Coordinates are expressed in each row's measured rack frame, so base
     # motion does not masquerade as flap or TCP motion in robot coordinates.
     interval=torch.tensor(indices[int(ids[j])][-(int(pre_closed_ticks[j])+1):])
     interval_raw=data['critic_obs'][interval,:464]
     interval_relation=data['critic_obs'][interval,533:569].reshape(-1,2,2,9)
     interval_flaps=flaps[j]
     interval_selected=interval_relation[torch.arange(len(interval))[:,None],torch.arange(2)[None],interval_flaps[None]]
     interval_tcp=interval_raw[:,50:68].reshape(-1,2,9)
     interval_tcp_R=_rotation_matrix(interval_tcp[...,3:])
     interval_relative_R=_rotation_matrix(interval_selected[...,3:])
     interval_point_hand=interval_selected[...,:3]+(interval_relative_R@tracker.contact_offsets[ids[j],...,None]).squeeze(-1)
     interval_point=interval_tcp[...,:3]+(interval_tcp_R@interval_point_hand[...,None]).squeeze(-1)
     interval_rack_R=_rotation_matrix(interval_raw[:,71:77])
     interval_to_rack=interval_rack_R.transpose(-1,-2)[:,None]
     interval_point_rack=(interval_to_rack@(interval_point-interval_raw[:,None,68:71])[...,None]).squeeze(-1)
     interval_tcp_rack=(interval_to_rack@(interval_tcp[...,:3]-interval_raw[:,None,68:71])[...,None]).squeeze(-1)
     interval_normal=(interval_to_rack@(interval_tcp_R@interval_relative_R)[...,0,None]).squeeze(-1)
     start_normal=interval_normal[0]
     point_delta=interval_point_rack[-1]-interval_point_rack[0]
     tcp_delta=interval_tcp_rack[-1]-interval_tcp_rack[0]
     interval_pending=interval_raw[:,:20]+interval_raw[:,416:436]
     arm_columns=torch.tensor(tracker.kinematics.columns)
     interval_torso=planar_position(interval_raw[:,:2],coordinates.links.to(interval_raw))
     interval_weak=50*data['critic_obs'][interval,479:487].reshape(-1,2,2,2).min(-1).values.max(-1).values
     reopens[int(ids[j])].append(dict(control_step=step+1,held_clock=int(clocks[j]),
      previous_closed_ticks=int(pre_closed_ticks[j]),previous_settled_ticks=int(pre_settled_ticks[j]),
      measured_closure_fraction=raw[j,46:48].tolist(),selected_current_point_distance_m=point[j].norm(dim=-1).tolist(),
      closest_current_panel_point_distance_m=nearest[j].norm(dim=-1).tolist(),closing_line_error_rad=angle[j].tolist(),
      actual_pre_pinching=(co[j,499:501]>.5).tolist(),actual_pre_stable=(co[j,505:507]>.5).tolist(),
      actual_next_pinching=(data['next_critic_obs'][ix[j],499:501]>.5).tolist(),
      selected_point_beyond12mm=bool(point[j].norm(dim=-1).max()>.012),axis_beyond0p30rad=bool(angle[j].max()>.30),
      closest_point_still_within6mm=bool(nearest[j].norm(dim=-1).max()<=.006),actual_executed_jaw_goals=goals[j,19:21].tolist(),
      immediately_preceding_closed_interval=dict(rows_including_reopen=len(interval),
       measured_closure_fraction=interval_raw[:,46:48].tolist(),
       selected_current_point_distance_m=interval_point_hand.norm(dim=-1).tolist(),
       contact_point_rack_m=interval_point_rack.tolist(),tcp_rack_m=interval_tcp_rack.tolist(),
       measured_torso_xz_m=interval_torso.tolist(),
       measured_arm_joints_rad=interval_raw[:,:20][:,arm_columns].tolist(),
       pending_arm_joint_lead_rad=(interval_pending[:,arm_columns]-interval_raw[:,:20][:,arm_columns]).tolist(),
       actual_weaker_pad_force_n=interval_weak.tolist(),
       actual_executed_jaw_goals=data['action'][interval,19:21].tolist(),
       actual_executed_torso_goals_m=(pilot.center+pilot.scale*data['action'][interval])[:,17:19].tolist(),
       flap_point_displacement_rack_m=point_delta.tolist(),tcp_displacement_rack_m=tcp_delta.tolist(),
       flap_point_displacement_along_start_panel_normal_m=(point_delta*start_normal).sum(-1).tolist(),
       tcp_displacement_along_start_panel_normal_m=(tcp_delta*start_normal).sum(-1).tolist())))
 previous_stats=tracker.report()
 for i,case in enumerate(cases):
  if not case['initial_layout_valid']:
   assert not indices[i] and not phases[i]
   records.append(dict(wave=wave,environment=i,seed=case['layout']['seed'],
    region=case['layout']['target_region'],box_type=case['layout']['target_box_type'],
    mode=case['collection_policy_mode'],initial_layout_valid=False,success=False,
    excluded_from_phase_diagnosis_reason='Original initial invalid request has no executed or TRAIN rows',
    original_requested_denominator_retained=True))
   continue
  ix=torch.tensor(indices[i]);assert len(ix)==ends[i]-starts[i]
  priv=data['next_critic_obs'][ix,464:530];final=priv[-1];term=case['result'];raw=data['critic_obs'][ix,:464]
  assert (final[35:37]>.5).tolist()==term['pinching'] and (final[41:43]>.5).tolist()==term['stable_hands'] and bool(final[54]>.5)==term['success']
  measured_terminals+=1
  pinch=priv[:,35:37]>.5;opposing=pinch.all(-1)&(priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,0]!=priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,1]);run=best=0
  for flag in opposing.tolist():run=run+1 if flag else 0;best=max(best,run)
  weaker=50*priv[:,15:23].reshape(-1,2,2,2).min(-1).values.max(-1).values
  guide=(torch.tensor(phases[i])>=0)&(torch.tensor(phases[i])<3)
  geometry={}
  if guide.any():
   relation=data['critic_obs'][ix,533:569].reshape(-1,2,2,9)
   assignment=int(tracker.assignment[i]);flaps=torch.tensor([assignment,1-assignment])
   selected_relation=relation[torch.arange(len(ix))[:,None],torch.arange(2)[None],flaps[None]]
   panel_relative=_rotation_matrix(selected_relation[...,3:])
   point_hand=selected_relation[...,:3]+(panel_relative@tracker.contact_offsets[i,...,None]).squeeze(-1)
   tcp=raw[:,50:68].reshape(-1,2,9);tcp_R=_rotation_matrix(tcp[...,3:])
   points=tcp[...,:3]+(tcp_R@point_hand[...,None]).squeeze(-1)
   rack_R=_rotation_matrix(raw[:,71:77]);outward=rack_R[...,1]
   front=raw[:,68:71]+outward*tracker.front_y
   depth=((front[:,None]-points)*outward[:,None]).sum(-1).clamp_min(0)
   stage_dist=(points+depth[...,None]*outward[:,None]-tcp[...,:3]).norm(dim=-1)
   angle=torch.acos((panel_relative[...,0]*tracker.axes[None]).sum(-1).abs().clamp(0,1))
   first_guide=int(torch.where(guide)[0][0]);last_guide=int(torch.where(guide)[0][-1]);torso=planar_position(raw[:,:2],coordinates.links.to(raw));geometry=dict(
    guided_min_both_stage_distance_m=float(stage_dist[guide].amax(-1).min()),
    guided_min_both_closing_line_error_rad=float(angle[guide].amax(-1).min()),
    guided_min_both_surface_goal_distance_m=float(point_hand[guide].norm(dim=-1).amax(-1).min()),
    last_guided_stage_distance_m=stage_dist[last_guide].tolist(),
    last_guided_closing_line_error_rad=angle[last_guide].tolist(),
    last_guided_surface_goal_distance_m=point_hand[last_guide].norm(dim=-1).tolist(),
    first_guided_measured_torso_xz_m=torso[first_guide].tolist(),
    last_guided_measured_torso_xz_m=torso[last_guide].tolist())
  gates=close_gates[i]
  closure_summary=dict(settled_close_variant=settled,actual_closed_command_guided_rows=len(gates),max_consecutive_projected_closed_ticks=max((r['closed_ticks'] for r in gates),default=0),max_consecutive_measured_settled_ticks=max((r['settled_ticks'] for r in gates),default=0),closed_rows_measured_both_closure_ge_85=sum(min(r['closure'])>=.85 for r in gates),closed_rows_at_least24_closed_ticks=sum(r['closed_ticks']>=24 for r in gates),closed_rows_at_least6_settled_ticks=sum(r['settled_ticks']>=6 for r in gates),closed_rows_current_point_within6mm=sum(r['current_point_distance_m']<=.006 for r in gates),closed_rows_axis_error_within_0p25rad=sum(r['axis_error_rad']<=.25 for r in gates))
  if gates:
   closure_summary.update(measured_closed_command_closure_median=np.median([r['closure'] for r in gates],axis=0).tolist(),minimum_both_current_contact_point_distance_m=min(r['current_point_distance_m'] for r in gates))
   if settled and not precise:closure_summary.update(minimum_both_fixed_close_point_distance_m=min(r['fixed_point_distance_m'] for r in gates),closed_rows_fixed_point_within6mm=sum(r['fixed_point_distance_m']<=.006 for r in gates),closed_rows_all_actual_v2_lift_gates_passed=sum(r['closed_ticks']>=24 and r['settled_ticks']>=6 and r['current_point_distance_m']<=.006 and r['fixed_point_distance_m']<=.006 and r['axis_error_rad']<=.25 for r in gates))
  closure_summary['precise_feedback_variant']=precise
  record=dict(wave=wave,environment=i,seed=case['layout']['seed'],region=case['layout']['target_region'],box_type=case['layout']['target_box_type'],mode=case['collection_policy_mode'],initial_layout_valid=True,success=supported_success(case),unsafe=bool(term['unsafe']),time_out=bool(term['time_out']),unsafe_causes=term['unsafe_causes'],rack_peak_body=term['rack_peak_body'],held_rows=len(ix),reconstructed_guided_rows=sum(0<=p<3 for p in phases[i]),reconstructed_phase_counts=dict(Counter(phases[i])),approach_phase_rows_with_any_commanded_closed_jaw=int(((torch.tensor(phases[i])==0)&(data['action'][ix,19:21]>0).any(-1)).sum()),approach_phase_rows_with_both_commanded_closed_jaws=int(((torch.tensor(phases[i])==0)&(data['action'][ix,19:21]>0).all(-1)).sum()),first_surface_approach_clock=(phases[i].index(1) if 1 in phases[i] else None),lift_attempt_events=events[i],ever_each_hand_pinched=pinch.any(0).tolist(),actual_opposing_pinch_rows=int(opposing.sum()),longest_actual_opposing_pinch_s=best/30,measured_closure_peak=raw[:,46:48].max(0).values.tolist(),best_actual_weaker_pad_force_n=weaker.max(0).values.tolist(),terminal_surface_distance_m=term['flap_distances'],terminal_actual_pinching=term['pinching'],terminal_stable_hands=term['stable_hands'])
  record.update(geometry)
  record['measured_closing_gate_diagnosis']=closure_summary
  record['precise_feedback_reopen_events']=reopens[i]
  records.append(record)
saved_statistics=s['perceived_contact_statistics']
fk_key='measured_fk_max_position_error_m'
assert offset==len(data['reward'])
assert {k:v for k,v in previous_stats.items() if k!=fk_key}=={k:v for k,v in saved_statistics.items() if k!=fk_key}
# CPU reconstruction can round FK differently from the saved CUDA collector.
# Every integer counter and all actual executed goals are checked separately.
fk_stat_difference=abs(previous_stats[fk_key]-saved_statistics[fk_key])
assert fk_stat_difference<=1e-7,(previous_stats,saved_statistics)
selected=[r for r in records if r['mode']=='perceived_contact_exploration'];greedy=[r for r in records if r['mode']=='greedy_current_policy'];lifts=[e for r in selected for e in r['lift_attempt_events']]
def summarize(group):
 return dict(episodes=len(group),success=sum(r['success'] for r in group),unsafe=sum(r['unsafe'] for r in group),time_out=sum(r['time_out'] for r in group),ever_left_pinched=sum(r['ever_each_hand_pinched'][0] for r in group),ever_right_pinched=sum(r['ever_each_hand_pinched'][1] for r in group),ever_both_opposing_pinches=sum(r['actual_opposing_pinch_rows']>0 for r in group),max_opposing_pinch_run_s=max(r['longest_actual_opposing_pinch_s'] for r in group))
result=dict(recorded_at=datetime.now().astimezone().isoformat(),run_directory_name=Path(proof['source_run']).name,snapshot_SHA256=proof['snapshot_SHA256'],completed_TRAIN_waves=waves,original_replay_rows_mapped_to_actual_wave_global_ID_clock_and_staged_target=len(data['reward']),actual_terminal_critic_pinch_stability_success_matches=len(records),reconstructed_helper_counters_exactly_equal_actual_saved_counters=True,recomputed_executed_normalized_goal_max_abs_error=max_error,statistics=previous_stats,selected=summarize(selected),greedy=summarize(greedy),lift_attempts=len(lifts),lift_attempts_pre_actual_bilateral_pinching=sum(all(e['actual_pre_pinching']) for e in lifts),lift_attempts_next_actual_bilateral_pinching=sum(all(e['actual_next_pinching']) for e in lifts),lift_attempt_measured_closure_fraction_min=(float(np.min([e['measured_closure_fraction'] for e in lifts])) if lifts else None),lift_attempt_measured_closure_fraction_median=(np.median([e['measured_closure_fraction'] for e in lifts],axis=0).tolist() if lifts else None),privileged_contact_used_for_analysis_ONLY_NOT_helper_or_policy_inputs=True,read_only_completed_snapshot_no_live_HDF_GPU_replay_or_optimizer_reads=True,no_rows_imported_for_training_or_policy_updates=True,independent_FINAL_unused=True,goal_not_complete=True,cases=records)
result.update(actual_saved_statistics=saved_statistics,reconstructed_helper_FK_max_error_abs_difference_m=fk_stat_difference,reconstructed_helper_FK_stat_tolerance_m=1e-7,reconstructed_helper_statistics_including_float_exact=previous_stats==saved_statistics)
result.update(upright_phase_only_diagnosis=args.upright_phase_only,
    continuous_body_goal_recomputation_validated=not args.upright_phase_only,
    actual_recorded_body_goals_preserved=True,
    exact_recorded_jaw_sequence_validated=args.upright_phase_only,
    continuous_goal_atol_unchanged_for_full_verification=3e-5,
    no_continuous_goal_tolerance_relaxation=True)
assert identity(path)==replay_identity, 'Original closed replay changed during inspection'
if args.closed_run is not None:
 require_closed_run(dest)
 result.update(original_closed_replay_read_without_disk_copy=True,
  original_replay_unchanged=True,original_writers_stopped_and_final_backup_verified=True)
result.update(original_TRAIN_requests=len(records),
 initial_invalid_requests_retained=sum(not r['initial_layout_valid'] for r in records),
 actual_terminal_critic_pinch_stability_success_matches=measured_terminals)
with args.output_json.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
public={k:v for k,v in result.items() if k!='cases'}
public.update(original_TRAIN_requests=len(records),initial_invalid_requests_retained=sum(not r['initial_layout_valid'] for r in records),
 actual_terminal_critic_pinch_stability_success_matches=measured_terminals)
if args.summary_json is not None:
 with args.summary_json.open('x') as f:f.write(json.dumps(public,indent=2,allow_nan=False)+'\n')
print(json.dumps(public,ensure_ascii=False))
