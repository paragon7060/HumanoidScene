"""Reconstruct closed TRAIN contact attempts from a verified readonly replay copy.

No active HDF/GPU replay reads, policy updates or imports into training occur.
Physical contact labels are diagnostic evidence only, never explorer inputs.
"""
import argparse,os
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
from summarize_batched_staged_run import supported_success
torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
parser.add_argument('--snapshot-proof',type=Path,required=True)
parser.add_argument('--output-json',type=Path,required=True)
parser.add_argument('--summary-json',type=Path)
args=parser.parse_args()
if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU-only analysis requires empty CUDA_VISIBLE_DEVICES')
if args.output_json.exists() or (args.summary_json is not None and args.summary_json.exists()):raise ValueError('Use unique new output files')
dest=args.snapshot_proof.parent
proof=json.loads(args.snapshot_proof.read_text());path=Path(proof['snapshot'])
assert path.is_file() and not path.is_symlink() and path.stat().st_uid==os.getuid()
sha=hashlib.sha256()
with path.open('rb') as f:
 for chunk in iter(lambda:f.read(8*1024*1024),b''):sha.update(chunk)
assert sha.hexdigest()==proof['snapshot_SHA256'] and path.stat().st_mode&0o222==0
s=torch.load(path,map_location='cpu',weights_only=False);data=s['executed_goal_transitions'];m=json.loads((dest/'completed_waves_metrics.json').read_text())
assert len(data['reward'])==proof['completed_online_rows'] and s['perceived_contact_statistics']==m['learner']['perceived_contact_statistics']
assert all(torch.isfinite(t).all() for t in data.values())
coordinates=PoseGoalCoordinates(exact_projected_base=True)
contract=s['goal_contract'];pilot=SimpleNamespace(coordinates=coordinates,center=torch.tensor(contract['goal_center']),scale=torch.tensor(contract['goal_scale']),agent=SimpleNamespace(action_projector=AbsoluteGoalJawProjector()))
variant=contract['TRAIN_perceived_contact_exploration']['name']
assert variant in ('TRAIN_arm20_perceived_contact_attempt_v1','TRAIN_arm20_perceived_settled_contact_attempt_v2')
settled=variant.endswith('_v2')
waves=proof['completed_TRAIN_waves']
assert waves and len(waves)==len(set(waves))
offset=0;previous_stats=None;records=[];max_error=0.
for wave in waves:
 cases=sorted([r for r in m['outcomes'] if r['wave']==wave],key=lambda r:r['environment'])
 assert len(cases)==128 and all(r['complete'] and r['initial_layout_valid'] and not r['result'].get('numerical_failure') for r in cases)
 starts=np.array([r['result']['staged_base']['manipulation_start'] for r in cases]);ends=np.array([r['result']['steps'] for r in cases]);chosen=torch.tensor([r['collection_policy_mode']=='perceived_contact_exploration' for r in cases])
 assert all(type(x) is int for x in starts.tolist())
 tracker=None;indices=[[] for _ in cases];phases=[[] for _ in cases];events=[[] for _ in cases]
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
  if tracker is None:tracker=PerceivedContactExploration(128,raw,previous_stats,settled_close=settled)
  pre=tracker.phase[ids].clone()
  proposed,(_,_,recomputed)=tracker.step(pilot,raw,(command,(ao,torch.full_like(co,float('nan')),goals)),ids,supp,chosen[ids],clocks)
  error=float((recomputed-goals).abs().max());max_error=max(max_error,error)
  assert torch.allclose(recomputed,goals,atol=3e-5,rtol=0),(wave,step,error)
  for j,i in enumerate(ids.tolist()):
   indices[i].append(int(ix[j]));phases[i].append(int(tracker.phase[i]))
   if pre[j]==1 and tracker.phase[i]==2:
    events[i].append(dict(control_step=step+1,held_clock=int(clocks[j]),measured_closure_fraction=raw[j,46:48].tolist(),physical_close_command=command[j,20:22].tolist(),actual_pre_pinching=(co[j,499:501]>.5).tolist(),actual_next_pinching=(data['next_critic_obs'][ix[j],499:501]>.5).tolist(),actual_pre_stable=(co[j,505:507]>.5).tolist(),actual_pre_weaker_pad_force_n=(50*co[j,479:487].reshape(2,2,2)).min(-1).values.max(-1).values.tolist(),actual_pre_pad_in_region=(co[j,487:495].reshape(2,2,2)>.5).tolist(),actual_pre_opposed=(co[j,495:499].reshape(2,2)>.5).tolist()))
 previous_stats=tracker.report()
 for i,case in enumerate(cases):
  ix=torch.tensor(indices[i]);assert len(ix)==ends[i]-starts[i]
  priv=data['next_critic_obs'][ix,464:530];final=priv[-1];term=case['result'];raw=data['critic_obs'][ix,:464]
  assert (final[35:37]>.5).tolist()==term['pinching'] and (final[41:43]>.5).tolist()==term['stable_hands'] and bool(final[54]>.5)==term['success']
  pinch=priv[:,35:37]>.5;opposing=pinch.all(-1)&(priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,0]!=priv[:,37:41].reshape(-1,2,2).argmax(-1)[:,1]);run=best=0
  for flag in opposing.tolist():run=run+1 if flag else 0;best=max(best,run)
  weaker=50*priv[:,15:23].reshape(-1,2,2,2).min(-1).values.max(-1).values
  record=dict(wave=wave,environment=i,seed=case['layout']['seed'],region=case['layout']['target_region'],box_type=case['layout']['target_box_type'],mode=case['collection_policy_mode'],success=supported_success(case),unsafe=bool(term['unsafe']),time_out=bool(term['time_out']),unsafe_causes=term['unsafe_causes'],rack_peak_body=term['rack_peak_body'],held_rows=len(ix),reconstructed_guided_rows=sum(0<=p<3 for p in phases[i]),lift_attempt_events=events[i],ever_each_hand_pinched=pinch.any(0).tolist(),actual_opposing_pinch_rows=int(opposing.sum()),longest_actual_opposing_pinch_s=best/30,measured_closure_peak=raw[:,46:48].max(0).values.tolist(),best_actual_weaker_pad_force_n=weaker.max(0).values.tolist(),terminal_surface_distance_m=term['flap_distances'],terminal_actual_pinching=term['pinching'],terminal_stable_hands=term['stable_hands'])
  records.append(record)
assert offset==len(data['reward']) and previous_stats==s['perceived_contact_statistics'],(offset,previous_stats,s['perceived_contact_statistics'])
selected=[r for r in records if r['mode']=='perceived_contact_exploration'];greedy=[r for r in records if r['mode']=='greedy_current_policy'];lifts=[e for r in selected for e in r['lift_attempt_events']]
def summarize(group):
 return dict(episodes=len(group),success=sum(r['success'] for r in group),unsafe=sum(r['unsafe'] for r in group),time_out=sum(r['time_out'] for r in group),ever_left_pinched=sum(r['ever_each_hand_pinched'][0] for r in group),ever_right_pinched=sum(r['ever_each_hand_pinched'][1] for r in group),ever_both_opposing_pinches=sum(r['actual_opposing_pinch_rows']>0 for r in group),max_opposing_pinch_run_s=max(r['longest_actual_opposing_pinch_s'] for r in group))
result=dict(recorded_at=datetime.now().astimezone().isoformat(),run_directory_name=Path(proof['source_run']).name,snapshot_SHA256=proof['snapshot_SHA256'],completed_TRAIN_waves=waves,original_replay_rows_mapped_to_actual_wave_global_ID_clock_and_staged_target=len(data['reward']),actual_terminal_critic_pinch_stability_success_matches=len(records),reconstructed_helper_counters_exactly_equal_actual_saved_counters=True,recomputed_executed_normalized_goal_max_abs_error=max_error,statistics=previous_stats,selected=summarize(selected),greedy=summarize(greedy),lift_attempts=len(lifts),lift_attempts_pre_actual_bilateral_pinching=sum(all(e['actual_pre_pinching']) for e in lifts),lift_attempts_next_actual_bilateral_pinching=sum(all(e['actual_next_pinching']) for e in lifts),lift_attempt_measured_closure_fraction_min=(float(np.min([e['measured_closure_fraction'] for e in lifts])) if lifts else None),lift_attempt_measured_closure_fraction_median=(np.median([e['measured_closure_fraction'] for e in lifts],axis=0).tolist() if lifts else None),privileged_contact_used_for_analysis_ONLY_NOT_helper_or_policy_inputs=True,read_only_completed_snapshot_no_live_HDF_GPU_replay_or_optimizer_reads=True,no_rows_imported_for_training_or_policy_updates=True,independent_FINAL_unused=True,goal_not_complete=True,cases=records)
with args.output_json.open('x') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
public={k:v for k,v in result.items() if k!='cases'}
if args.summary_json is not None:
 with args.summary_json.open('x') as f:f.write(json.dumps(public,indent=2,allow_nan=False)+'\n')
print(json.dumps(public,ensure_ascii=False))
