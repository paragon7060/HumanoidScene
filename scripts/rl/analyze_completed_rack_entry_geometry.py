"""Offline nominal geometry distance; NOT actual collision forces/clearance.

Uses exact saved pre-step robot joints, known authored rack primitive geometry,
STL convex-hull vertices and a conservative forearm swept-radius proxy.
"""
from pathlib import Path
from datetime import datetime
from collections import Counter
import argparse,hashlib,json,os,sys,xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial import ConvexHull
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts/rl')]
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model
from kuavo_isaaclab_scene.robots.end_effector import calibration_definition
from kuavo_isaaclab_scene.rl.multi_box.demo_replay import _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.experiments.tensor_arm_kinematics import TensorArmKinematics,rotation
from summarize_batched_staged_run import supported_success
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';torch.set_num_threads(1)
parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
parser.add_argument('--snapshot-proof',type=Path,required=True)
parser.add_argument('--rack-colliders-json',type=Path,required=True)
parser.add_argument('--output-json',type=Path,required=True)
args=parser.parse_args()
assert not args.output_json.exists(),'Use a distinct output file'
ptr=json.loads(args.snapshot_proof.read_text())
assert len(ptr['completed_TRAIN_waves'])==1,'Use an explicit single completed-wave copy'
wave=ptr['completed_TRAIN_waves'][0]
path=Path(ptr['snapshot']);assert path.stat().st_uid==os.getuid() and path.stat().st_mode&0o222==0 and not path.is_symlink()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
assert sha(path)==ptr['snapshot_SHA256']
state=torch.load(path,map_location='cpu',weights_only=False);data=state['executed_goal_transitions'];N=len(data['reward']);assert N==ptr['completed_online_rows']
metrics=json.loads((path.parent/'completed_waves_metrics.json').read_text())
cases=sorted([x for x in metrics['outcomes'] if x['wave']==wave],key=lambda x:x['environment']);assert len(cases)==128
starts=np.array([x['result']['staged_base']['manipulation_start'] for x in cases]);ends=np.array([x['result']['steps'] for x in cases])
indices=[[] for _ in cases];offset=0
for step in range(int(ends.max())):
 for env in np.where((starts<=step)&(ends>step))[0]:indices[env].append(offset);offset+=1
assert offset==N
for i,ix in enumerate(indices):
 assert len(ix)==ends[i]-starts[i]
 assert torch.allclose(data['critic_obs'][ix,530],torch.arange(len(ix),dtype=data['critic_obs'].dtype)/900,atol=1e-7,rtol=0)
last=torch.tensor([ix[-1] for ix in indices]);raw=data['critic_obs'][last,:464].double();q=raw[:,:20]
assert torch.isfinite(raw).all()
tcp=raw[:,50:68].reshape(128,2,9);tcpR=_rotation_matrix(tcp[...,3:]);rackR=_rotation_matrix(raw[:,71:77])
kin=TensorArmKinematics(dtype=torch.float64);pp,RR,_=kin.fk(q)
maxfk=float((pp-tcp[...,:3]).norm(dim=-1).max());maxR=float((RR-tcpR).abs().max());assert maxfk<.001 and maxR<.001
parentP=q.new_zeros(128,3);parentR=torch.eye(3,dtype=q.dtype).expand(128,3,3).clone()
for xyz,originR,kind,col,axis in kin.parent_chain:
 parentP+=(parentR@xyz[:,None]).squeeze(-1);parentR=parentR@originR
 if kind=='revolute':parentR=parentR@rotation(axis,q[:,col])
 elif kind=='prismatic':parentP+=(parentR@axis[:,None]).squeeze(-1)*q[:,col,None]
forearms=[]
for cols,arm,chain in zip(kin.columns,kin.arms,kin.chains):
 p,R=parentP.clone(),parentR.clone();movable=0;found=None
 for joint,(xyz,originR,axis) in zip(arm.joints,chain):
  p+=(R@xyz[:,None]).squeeze(-1);R=R@originR
  if axis is not None:R=R@rotation(axis,q[:,cols[movable]]);movable+=1
  if joint.child=='zarm_'+arm.side[0]+'4_link':found=(p.clone(),R.clone());break
 assert found is not None;forearms.append(found)
model=resolve_robot_model('s63','leju-twofinger');root=ET.parse(model.urdf_path).getroot()
bychild={j.find('child').get('link'):j for j in root.findall('joint')};bylink={l.get('name'):l for l in root.findall('link')}
def xyz(el):return np.fromstring(el.find('origin').get('xyz','0 0 0'),sep=' ')
closed=calibration_definition(model)['closed_offsets'];palms=[];meshes=[];cylinder=[]
for side,letter in [('left','l'),('right','r')]:
 base=letter+'_twofinger_base';eef='zarm_'+letter+'7_end_effector'
 delta=xyz(bychild[base])-xyz(bychild[eef])-closed[side]
 collision=bylink[base].find('collision');mesh=collision.find('geometry/mesh')
 assert mesh is not None and mesh.get('scale') is None and np.allclose(xyz(collision),0)
 mp=(Path(model.urdf_path).parent/mesh.get('filename')).resolve();blob=mp.read_bytes();count=int.from_bytes(blob[80:84],'little');assert len(blob)==84+50*count
 dtype=np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')]);v=np.frombuffer(blob,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3).astype(float)
 hull=ConvexHull(v);vertices=v[hull.vertices];palms.append(vertices+delta)
 meshes.append(dict(side=side,file=mp.name,SHA256=sha(mp),triangles=count,convex_hull_vertex_count=len(vertices)))
 link=bylink['zarm_'+letter+'4_link'];coll=link.find('collision');cy=coll.find('geometry/cylinder');assert cy is not None
 center=xyz(coll);radius=float(cy.get('radius'));length=float(cy.get('length'));assert radius==.05 and length==.26
 assert np.allclose(np.fromstring(coll.find('origin').get('rpy','0 0 0'),sep=' '),0)
 centerline=np.repeat(center[None],33,axis=0);centerline[:,2]+=np.linspace(-length/2,length/2,33)
 cylinder.append((centerline,radius))
nominal=json.loads(args.rack_colliders_json.read_text())
assert Path(nominal['asset']).is_file() and sha(nominal['asset'])==nominal['asset_SHA256']
assert nominal.get('used_layers_SHA256'),'An extract with composed-layer checksums is required'
assert all(Path(p).is_file() and sha(p)==h for p,h in nominal['used_layers_SHA256'].items())
gs=nominal['colliders'];assert len(gs)==565
centers=[];rotations=[];halves=[];types=[];names=[];axes=[]
for g in gs:
 T=np.array(g['T_asset_from_geometry']);M=T[:3,:3];scale=np.linalg.norm(M,axis=0);R=M/scale[None]
 assert np.allclose(R.T@R,np.eye(3),atol=1e-6) and np.linalg.det(R)>.999
 bounds=np.array(g['local_bounds_m']);localC=bounds.mean(0);h=(bounds[1]-bounds[0])/2
 centers.append(M@localC+T[:3,3]);rotations.append(R);halves.append(h*scale);types.append(g['type']);names.append(g['path']);axes.append(int(np.argmax(h)))
centers=np.array(centers);rotations=np.array(rotations);halves=np.array(halves);cyl=np.array(types)=='Cylinder';ax=np.array(axes);box=~cyl
assert box.sum()==19 and cyl.sum()==546
def nearest(points,radius=0):
 delta=points[:,None]-centers[None]
 local=np.einsum('pkj,kji->pki',delta,rotations)
 d=np.abs(local)-halves[None]
 distances=np.linalg.norm(np.maximum(d,0),axis=-1)+np.minimum(d.max(-1),0)
 for a in range(3):
  mask=cyl&(ax==a)
  if not mask.any():continue
  lateral=[k for k in range(3) if k!=a]
  assert np.allclose(halves[mask][:,lateral[0]],halves[mask][:,lateral[1]],atol=1e-6)
  radial=np.linalg.norm(local[:,mask][:,:,lateral],axis=-1)-halves[mask,lateral[0]][None]
  axial=np.abs(local[:,mask,a])-halves[mask,a][None]
  ds=np.stack((radial,axial),-1)
  distances[:,mask]=np.linalg.norm(np.maximum(ds,0),axis=-1)+np.minimum(ds.max(-1),0)
 distances-=radius
 p,g=np.unravel_index(distances.argmin(),distances.shape)
 return dict(minimum_nominal_proxy_distance_m=float(distances[p,g]),nearest_nominal_collider=names[g],collider_type=types[g],
  sampled_point_asset_frame_m=points[p].tolist(),sampling_method='convex_hull_vertex_signed_distance' if radius==0 else 'forearm33_centerline_swept_radius_conservative')
records=[]
for i,x in enumerate(cases):
 probes={};toRack=rackR[i].numpy().T;origin=raw[i,68:71].numpy()
 for hand,(side,letter) in enumerate([('left','l'),('right','r')]):
  pp=tcpR[i,hand].numpy()@palms[hand].T+tcp[i,hand,:3].numpy()[:,None]
  probes[letter+'_twofinger_base']=nearest((toRack@(pp-origin[:,None])).T)
  p,R=forearms[hand];line,rad=cylinder[hand]
  pp=R[i].numpy()@line.T+p[i].numpy()[:,None]
  probes['zarm_'+letter+'4_link']=nearest((toRack@(pp-origin[:,None])).T,rad)
 r=x['result'];actualbody=r['rack_peak_body'];actualrack=r['unsafe_causes']['robot_rack_collision']
 record=dict(environment=i,region=x['layout']['target_region'],box_type=x['layout']['target_box_type'],mode=x['collection_policy_mode'],
 category='success' if supported_success(x) else 'unsafe' if r['unsafe'] else 'time_out',
 snapshot_row_before_final_step=int(last[i]),actual_terminal_robot_rack_collision=actualrack,
 actual_terminal_peak_body=actualbody,actual_terminal_peak_force_n=r['rack_peak_force_n'],nominal_geometry_probes=probes,
 matching_peak_body_probe=(probes.get(actualbody) if actualrack else None))
 records.append(record)
 if i%32==31:print(json.dumps(dict(phase='nominal_body_geometry',episodes_finished=i+1)),flush=True)
hits=[r for r in records if r['matching_peak_body_probe'] is not None]
by_body={}
for body in sorted({r['actual_terminal_peak_body'] for r in hits}):
 selected=[r for r in hits if r['actual_terminal_peak_body']==body];ds=[r['matching_peak_body_probe']['minimum_nominal_proxy_distance_m'] for r in selected]
 by_body[body]=dict(actual_termination_cases=len(ds),nominal_proxy_distance_median_m=float(np.median(ds)),nominal_proxy_distance_min_m=float(min(ds)),nominal_proxy_distance_max_m=float(max(ds)),nominal_proxy_nonpositive_count=sum(v<=0 for v in ds),closest_authored_parts=dict(Counter(r['matching_peak_body_probe']['nearest_nominal_collider'] for r in selected)))
out=dict(recorded_at=datetime.now().astimezone().isoformat(),verified_readonly_TRAIN_rows=N,completed_episodes_mapped=128,
 source_snapshot_SHA256=ptr['snapshot_SHA256'],source_URDF_SHA256=sha(model.urdf_path),nominal_rack_asset_SHA256=nominal['asset_SHA256'],
 nominal_rack_composed_layers_SHA256=[dict(file=Path(p).name,SHA256=h) for p,h in nominal['used_layers_SHA256'].items()],
 measured_last_pre_step_TCP_FK_max_position_difference_m=maxfk,measured_last_pre_step_TCP_FK_rotation_max_element_difference=maxR,
 rack_static_boxes=19,rack_nominal_roller_cylinders=546,palm_source_meshes=meshes,
 mapped_actual_rack_terminal_bodies=len(hits),matching_terminal_body_nominal_geometry_summary=by_body,records=records,
 limitations=['Last recorded held pose is before final action; post-step collision pose was NOT recorded here.',
 'Rack geometry is nominal authored asset; actual runtime contact offsets/shape approximations/roller states are NOT measured.',
 'Palm convex-hull vertex sampling can miss face/edge intersections and is NOT a complete collision verifier.',
 'Forearm swept-sphere centerline is conservative and differs from actual finite cylinder ends; 33point sampling also approximates line minimum.',
 'Geometric overlaps or separation do NOT replace measured contact force, unsafe termination or real grasp success.'],
 no_live_HDF_GPU_replay_or_policy_updates=True,no_evaluation_rows_imported=True,
 source_snapshot_checksum_reverified_after_diagnosis=sha(path)==ptr['snapshot_SHA256'],
 original_DR_success_safety_preserved=True,new_physical_success_NOT_verified=True,goal_not_complete=True)
target=args.output_json;target.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:out[k] for k in ['recorded_at','mapped_actual_rack_terminal_bodies','matching_terminal_body_nominal_geometry_summary','measured_last_pre_step_TCP_FK_max_position_difference_m']},ensure_ascii=False))
