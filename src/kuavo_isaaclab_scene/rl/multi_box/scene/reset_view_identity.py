"""Read actual PhysX row identities and sensor/Articulation source poses."""
import re

import torch

from .reset_diagnostics import _finite_values


def environment_view_order(paths, num_envs):
    paths=list(paths)
    ids=[]
    for path in paths:
        match=re.search(r'/env_(\d+)(?:/|$)',path)
        ids.append(int(match.group(1)) if match else None)
    return dict(prim_paths=paths,environment_ids=ids,
        matches_scene_row_order=len(paths)==num_envs and ids==list(range(num_envs)))


def compare_contact_source_poses(sensor_pose_xyzw, articulation_pose_wxyz):
    if sensor_pose_xyzw.shape!=articulation_pose_wxyz.shape or sensor_pose_xyzw.ndim!=2 \
            or sensor_pose_xyzw.shape[1]!=7:
        raise ValueError('Source pose views must have equal environment rows and pose7')
    a=sensor_pose_xyzw.detach();b=articulation_pose_wxyz.detach()
    q=a[:,[6,3,4,5]];p=b[:,3:]
    valid=torch.isfinite(a).all(-1)&torch.isfinite(b).all(-1)&(q.norm(dim=-1)>1e-6)&(p.norm(dim=-1)>1e-6)
    position=(a[:,:3]-b[:,:3]).norm(dim=-1)
    dot=(q*p).sum(-1).abs()/(q.norm(dim=-1)*p.norm(dim=-1))
    dot=torch.where(valid,dot.clamp(0,1),torch.full_like(dot,float('nan')))
    return dict(position_error_m=_finite_values(position.cpu().tolist()),
        quaternion_absolute_unit_dot=_finite_values(dot.cpu().tolist()),
        invalid_pose_rows=(~valid).cpu().tolist(),
        source_poses_agree=bool((valid&(position<1e-5)&(dot>1-1e-5)).all()),
        measurement_only=True)


def startup_tensor_view_identity_snapshot(env, asset_names, body_sensor_names):
    root_views={}
    for group in (env.scene.articulations,env.scene.rigid_objects):
        for name,asset in group.items():
            root_views[name]=environment_view_order(asset.root_physx_view.prim_paths,env.num_envs)
    specs=[dict(asset_name=asset,sensor_name=sensor,source_body='Body')
        for asset,sensor in zip(asset_names,body_sensor_names,strict=True)]
    specs+=list(getattr(env,'_reset_flap_contact_reporters',()))
    contact_views={}
    for spec in specs:
        asset=env.scene[spec['asset_name']];sensor=env.scene[spec['sensor_name']]
        if list(sensor.body_names)!=[spec['source_body']]:
            raise ValueError('Source identity audit requires its one declared rigid body')
        source=sensor.body_physx_view
        paths=environment_view_order(source.prim_paths,env.num_envs)
        body=asset.body_names.index(spec['source_body'])
        poses=compare_contact_source_poses(source.get_transforms(),asset.data.body_link_pose_w[:,body])
        contact_views[spec['sensor_name']]=dict(asset=spec['asset_name'],body=spec['source_body'],**paths,**poses)
    return dict(environment_origins_world_m=env.scene.env_origins.detach().cpu().tolist(),
        root_views=root_views,contact_source_views=contact_views,
        all_root_environment_orders_match_scene=all(r['matches_scene_row_order'] for r in root_views.values()),
        all_contact_source_environment_orders_match_scene=all(r['matches_scene_row_order'] for r in contact_views.values()),
        all_contact_source_poses_agree=all(r['source_poses_agree'] for r in contact_views.values()),
        actual_backend_prim_paths_used=True,USD_pose_fallback=False,physics_parameters_written=False,
        limitation='Pose disagreement is a readback discrepancy; it alone does not establish a solver/contact cache cause.')
