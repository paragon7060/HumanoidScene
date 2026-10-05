#!/usr/bin/env python3
"""CPU audit of labels from immutable, Drive-verified successful TRAIN paths.

No actor updates, rollout, label edits, or DEV/FINAL imports. The analytic
conditional fractions use the real region/episode/time sampling weights.
The empirical batch mean mirrors the loss reduction over eligible hands.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank,REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--experience',type=Path,required=True)
parser.add_argument('--verified-receipt',type=Path,required=True)
parser.add_argument('--output-json',type=Path,required=True)
parser.add_argument('--sampled-batches',type=int,default=1000)
args=parser.parse_args()
if not 1<=args.sampled_batches<=10000:parser.error('Sampled batches must be in[1,10000]')
experience=args.experience.resolve()
receipt=json.loads(args.verified_receipt.read_text())
entry=next(e for e in receipt['files'] if e['file']==experience.name)
stat=(experience.stat().st_size,experience.stat().st_mtime_ns)
if not receipt['all_files_size_MD5_verified'] or not entry['size_MD5_verified'] \
        or Path(receipt['input_dir']).resolve()!=experience.parent or stat[0]!=entry['bytes']:
    raise ValueError('Immutable experience differs from the verified input receipt')
digest=hashlib.md5()
with experience.open('rb') as stream:
    for part in iter(lambda:stream.read(4*1024**2),b''):digest.update(part)
if digest.hexdigest()!=entry['MD5']:raise ValueError('Immutable TRAIN MD5 differs')
torch.set_num_threads(1)
data=torch.load(experience,map_location='cpu',weights_only=True,mmap=True)
bank=TrainSuccessBank(518,577,data['successful_train_transitions']['config'])
bank.restore(data['successful_train_transitions'])
gate=GoalGripperProjector();available=[r for r in REGIONS if bank.episodes[r]]
result=dict(recorded_at_UTC=datetime.now(timezone.utc).isoformat(),fresh_source_MD5_verified=digest.hexdigest(),
    source_experience_bytes=stat[0],bank=bank.report(),sampling='equal_available_region_then_uniform_episode_then_uniform_time',
    gate='unchanged_production_nominal_assigned_flap_midpoint_distance_le_0p12m',regions={},TRAIN_only=True,
    no_policy_replay_physics_or_labels_modified=True,no_DEV_or_FINAL_imported=True,goal_not_complete=True)
for region in available:
    episodes=[]
    for e in bank.episodes[region]:
        rows=e['rows'];near=gate.near(rows['actor_obs']);close=rows['action'][:,19:21]>0
        episodes.append(dict(identity=e['identity'],rows=len(near),eligible_mass=near.double().mean(0).tolist(),close_eligible_mass=(near&close).double().mean(0).tolist(),
            eligible_rows=near.sum(0).tolist(),close_eligible_rows=(near&close).sum(0).tolist(),
            both_eligible_rows=int(near.all(-1).sum()),both_close_both_eligible_rows=int((near&close).all(-1).sum()),
            commands_by_branch={name:int(((close[:,0]==l)&(close[:,1]==r)).sum()) for name,l,r in [('OO',False,False),('OC',False,True),('CO',True,False),('CC',True,True)]}))
    eligible=torch.tensor([e['eligible_mass'] for e in episodes]).mean(0)
    closed=torch.tensor([e['close_eligible_mass'] for e in episodes]).mean(0)
    result['regions'][region]=dict(episodes=episodes,expected_eligible_mass_per_uniform_episode_time=eligible.tolist(),
        expected_closed_eligible_mass_per_uniform_episode_time=closed.tolist(),
        expected_conditional_close_fraction_per_hand=(closed/eligible.clamp_min(1e-12)).tolist(),
        pooled_eligible_rows=[sum(e['eligible_rows'][h] for e in episodes) for h in range(2)],
        pooled_close_eligible_rows=[sum(e['close_eligible_rows'][h] for e in episodes) for h in range(2)])
torch.manual_seed(20261006);conditional=[];all_eligible=torch.zeros(2,dtype=torch.long);all_closed=torch.zeros(2,dtype=torch.long)
for i in range(args.sampled_batches):
    b=bank.sample(64,'cpu');near=gate.near(b['actor_obs']);closed=b['action'][:,19:21]>0
    all_eligible+=near.sum(0);all_closed+=(near&closed).sum(0)
    conditional.append(float((near&closed).sum()/near.sum().clamp_min(1)))
result['actual_sampler_batches_64']=dict(batches=args.sampled_batches,eligible_hand_rows=all_eligible.tolist(),closed_eligible_hand_rows=all_closed.tolist(),
    conditional_close_fraction_per_hand=(all_closed/all_eligible).tolist(),mean_of_batch_BCE_close_label_fractions=sum(conditional)/len(conditional))
assert stat==(experience.stat().st_size,experience.stat().st_mtime_ns)
out=args.output_json;out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(output=str(out),bank=bank.report(),regions={k:{'episodes':len(v['episodes']),'conditional_close':v['expected_conditional_close_fraction_per_hand']} for k,v in result['regions'].items()},sampler=result['actual_sampler_batches_64']),ensure_ascii=False))
