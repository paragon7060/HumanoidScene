"""Extract nominal composed rack collision geometry without a simulator.

Loads only the installed USD libraries. It does not create Kit/PhysX/CUDA
contexts, change assets, or evaluate collision forces.
"""
from pathlib import Path
import argparse,os,sys,subprocess
ROOT=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
parser.add_argument('--output-json',type=Path,required=True)
args=parser.parse_args()
assert not args.output_json.exists(),'Use a distinct output file'
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cache=Path(sys.prefix)/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages/isaacsim/extscache'
roots=list(cache.glob('omni.usd.libs-*/pxr/Usd/__init__.py'));assert len(roots)==1
root=roots[0].parents[2]
env=os.environ.copy();env['PYTHONPATH']=str(root)+':'+str(ROOT/'src')
env['LD_LIBRARY_PATH']=str(root/'bin')+':'+str(Path(sys.prefix)/'lib')+':'+env.get('LD_LIBRARY_PATH','')
env['RL_OFFLINE_RACK_OUTPUT']=str(args.output_json.resolve())
code=r'''
from pathlib import Path
import numpy as np,json,hashlib,os
from pxr import Usd,UsdGeom,UsdPhysics
from kuavo_isaaclab_scene.core.paths import RACK_ROLLER_RUNTIME_ASSET
stage=Usd.Stage.Open(str(RACK_ROLLER_RUNTIME_ASSET));assert stage
cache=UsdGeom.XformCache();base=stage.GetDefaultPrim();geoms=[]
for p in Usd.PrimRange(base,Usd.TraverseInstanceProxies()):
 if p.GetAttribute('physics:collisionEnabled').Get() is not True:continue
 geom=UsdGeom.Gprim(p)
 if not geom:continue
 T=np.asarray(cache.GetLocalToWorldTransform(p),dtype=float).T
 if p.IsA(UsdGeom.Mesh):
  v=np.array(UsdGeom.Mesh(p).GetPointsAttr().Get(),dtype=float)
  local_bounds=[v.min(0).tolist(),v.max(0).tolist()];kind='Mesh';n=len(v)
 elif p.IsA(UsdGeom.Cylinder):
  c=UsdGeom.Cylinder(p);r=float(c.GetRadiusAttr().Get());h=float(c.GetHeightAttr().Get());axis=c.GetAxisAttr().Get()
  b=np.ones(3)*r;b['XYZ'.index(axis)]=h/2
  local_bounds=[(-b).tolist(),b.tolist()];kind='Cylinder';n=None
 else:
  continue
 geoms.append(dict(path=str(p.GetPath()),type=kind,local_bounds_m=local_bounds,T_asset_from_geometry=T.tolist(),vertex_count=n))
layers={}
for layer in stage.GetUsedLayers():
 p=Path(layer.realPath)
 if p.is_file():layers[str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
assert layers
out=dict(used_layers_SHA256=layers,asset=str(RACK_ROLLER_RUNTIME_ASSET),asset_SHA256=hashlib.sha256(RACK_ROLLER_RUNTIME_ASSET.read_bytes()).hexdigest(),collider_count=len(geoms),colliders=geoms,simulator_NOT_started=True,GPU_mask_empty=True,nominal_authored_asset_NOT_measured_roller_state=True)
Path(os.environ['RL_OFFLINE_RACK_OUTPUT']).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(dict(colliders=len(geoms),types={k:sum(g['type']==k for g in geoms) for k in ['Mesh','Cylinder']},first_meshes=[g for g in geoms if g['type']=='Mesh'][:4])))
'''
raise SystemExit(subprocess.run([sys.executable,'-c',code],env=env).returncode)
