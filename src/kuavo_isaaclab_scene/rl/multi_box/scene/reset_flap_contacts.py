"""Frozen startup measurements for the previously unobserved flap sources."""
from copy import deepcopy

import torch

from .reset_diagnostics import _finite_values, strongest_normal_contact_pairs, attach_contact_target_states


FLAP_NAMES=('flap_front','flap_back','flap_right','flap_left')


def resolve_startup_flap_contact_paths(records, env_regex_ns):
    """Match InteractiveScene's expansion of sensor configuration paths."""
    if not isinstance(env_regex_ns,str) or not env_regex_ns.startswith('/'):
        raise ValueError('Flap contact attribution requires the actual scene environment namespace')
    resolved=deepcopy(records)
    for record in resolved:
        record['source_path']=record['source_path'].replace('{ENV_REGEX_NS}',env_regex_ns)
        record['other_box_paths']=[p.replace('{ENV_REGEX_NS}',env_regex_ns) for p in record['other_box_paths']]
    return resolved


def add_startup_flap_contact_reporters(scene, body_sensor_names, box_paths, *, physical_pools=None):
    """One source per reporter, as required by PhysX filtered contact views.

    Only frozen diagnostics call this. Existing body/finger reporters, actual
    collision rules, body properties, and all poses are preserved.
    """
    if len(body_sensor_names)!=len(box_paths):
        raise ValueError('Flap reporters require ordered physical-pool geometry')
    pools=set(range(len(box_paths))) if physical_pools is None else set(physical_pools)
    if not pools or any(type(p)!=int or not 0<=p<len(box_paths) for p in pools) \
            or (physical_pools is not None and len(pools)!=len(physical_pools)):
        raise ValueError('Flap source pools must be distinct valid physical-pool IDs')
    records=[]
    for pool,(body_sensor,paths) in enumerate(zip(body_sensor_names,box_paths,strict=True)):
        if set(paths['flaps'])!=set(FLAP_NAMES):
            raise ValueError('Every physical box needs exactly four distinct flap sources')
        source_paths=list(paths['flaps'].values())
        if len(set(source_paths))!=4 or any(p.rsplit('/',1)[-1]!=flap for flap,p in paths['flaps'].items()):
            raise ValueError('Flap source paths must identify distinct literal rigid bodies')
        if pool not in pools:continue
        for flap in FLAP_NAMES:
            name=f'reset_flap_pair_{pool}_{flap}'
            if hasattr(scene,name):raise ValueError('Startup flap reporter already exists')
            cfg=deepcopy(getattr(scene,body_sensor));cfg.prim_path=paths['flaps'][flap]
            setattr(scene,name,cfg)
            records.append(dict(pool=pool,asset_name=paths['asset_name'],sensor_name=name,
                source_body=flap,source_path=cfg.prim_path,
                other_box_paths=[p for other,entry in enumerate(box_paths) if other!=pool
                    for p in [entry['body'],*entry['flaps'].values()]]))
    return records


def startup_flap_contact_snapshot(env, records, target_states):
    """Keep every other-box pair's maximum/count, not only the global top3.

    Zero maximum means no reported other-box normal force in this frame;
    it does not exclude tangential friction or unsampled earlier contacts.
    """
    active=env._multi_box_active.detach().cpu();pools=env._multi_box_pool_ids.detach().cpu()
    rows=[]
    for record in records:
        selected=torch.nonzero(active&(pools==record['pool']),as_tuple=False)
        if not len(selected):continue
        asset=env.scene[record['asset_name']];sensor=env.scene[record['sensor_name']]
        if list(sensor.body_names)!=[record['source_body']]:
            raise ValueError('Flap filtered reporter must resolve to its one declared source body')
        ids=selected[:,0].to(env.device);force=sensor.data.force_matrix_w
        paths=sensor.cfg.filter_prim_paths_expr
        if force is None or force.shape!=(env.num_envs,1,len(paths),3):
            raise ValueError('Flap contact matrix does not match ordered body/filter axes')
        if sensor.data.net_forces_w.shape!=(env.num_envs,1,3):
            raise ValueError('Flap net contact force does not match one source body')
        indices=[paths.index(p) for p in record['other_box_paths']]
        matrix=force[ids,0]
        top=strongest_normal_contact_pairs(matrix,paths)
        other=strongest_normal_contact_pairs(matrix[:,indices],record['other_box_paths'])
        attach_contact_target_states(top,selected[:,0].tolist(),target_states)
        attach_contact_target_states(other,selected[:,0].tolist(),target_states)
        magnitude=matrix[:,indices].norm(dim=-1)
        maxima=magnitude.amax(-1) if indices else magnitude.new_zeros(len(ids))
        counts=(magnitude>0).sum(-1).cpu().tolist()
        body=asset.body_names.index(record['source_body'])
        pose=asset.data.body_link_pose_w[ids,body].detach().cpu().tolist()
        velocity=asset.data.body_link_vel_w[ids,body].detach().cpu().tolist()
        net=sensor.data.net_forces_w[ids,0].detach().cpu().tolist()
        for (i,logical),p,v,n,maximum,count,a,b in zip(selected.tolist(),pose,velocity,net,
                maxima.cpu().tolist(),counts,top,other,strict=True):
            rows.append(dict(environment=i,logical_id=logical,original_pool_id=record['pool'],
                source_asset=record['asset_name'],source_body=record['source_body'],
                source_pose_world=_finite_values(p),source_velocity_world=_finite_values(v),
                net_normal_force_world_n=_finite_values(n),**a,
                maximum_other_box_pair_normal_force_n=_finite_values([maximum])[0],
                nonzero_other_box_pair_count=count,
                other_box_nonfinite_filter_count=b['nonfinite_filter_count'],
                strongest_other_box_normal_pairs=b['strongest_normal_pairs']))
    return rows
