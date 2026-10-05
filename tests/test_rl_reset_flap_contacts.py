from copy import deepcopy
from types import SimpleNamespace
import json

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_flap_contacts import (
    FLAP_NAMES,add_startup_flap_contact_reporters,startup_flap_contact_snapshot,
    resolve_startup_flap_contact_paths,
)


def setup():
    paths=[dict(asset_name=f'box{i}',body=f'/World/envs/env_.*/box{i}/Body',
                flaps={f:f'/World/envs/env_.*/box{i}/{f}' for f in FLAP_NAMES}) for i in range(2)]
    targets=['rack',*[p for box in paths for p in [box['body'],*box['flaps'].values()]]]
    cfg=SimpleNamespace(collision_rules=['unchanged'],
        body0=SimpleNamespace(prim_path=paths[0]['body'],filter_prim_paths_expr=targets.copy(),history_length=1),
        body1=SimpleNamespace(prim_path=paths[1]['body'],filter_prim_paths_expr=targets.copy(),history_length=1))
    records=add_startup_flap_contact_reporters(cfg,['body0','body1'],paths)
    scene={};lookup={}
    for i in range(2):
        pose=torch.zeros(2,5,7);pose[:,:,0]=torch.tensor([100.,200.])[:,None]+i;pose[:,:,3]=1
        vel=torch.full((2,5,6),float(i))
        scene[f'box{i}']=SimpleNamespace(body_names=['Body',*FLAP_NAMES],
            data=SimpleNamespace(body_link_pose_w=pose,body_link_vel_w=vel))
        for j,body in enumerate(['Body',*FLAP_NAMES]):
            lookup[(f'/World/envs/env_.*/box{i}',body)]=dict(asset=f'box{i}',body=body,
                pose=pose[:,j].tolist(),velocity=vel[:,j].tolist())
    for r in records:
        c=getattr(cfg,r['sensor_name'])
        scene[r['sensor_name']]=SimpleNamespace(cfg=c,body_names=[r['source_body']],
            data=SimpleNamespace(force_matrix_w=torch.zeros(2,1,len(targets),3),net_forces_w=torch.zeros(2,1,3)))
    env=SimpleNamespace(num_envs=2,device='cpu',scene=scene,
        _multi_box_active=torch.tensor([[True,True],[True,False]]),
        _multi_box_pool_ids=torch.tensor([[0,1],[1,-1]]))
    return env,cfg,records,lookup


def test_one_source_reporters_preserve_existing_body_filters_and_collision_rules():
    _,cfg,records,_=setup()
    assert len(records)==8 and len({r['source_path'] for r in records})==8
    assert cfg.collision_rules==['unchanged'] and cfg.body0.prim_path.endswith('/Body')
    for r in records:
        c=getattr(cfg,r['sensor_name'])
        assert c.filter_prim_paths_expr==cfg.body0.filter_prim_paths_expr
        assert c.filter_prim_paths_expr is not cfg.body0.filter_prim_paths_expr
        assert len(r['other_box_paths'])==5


def test_other_box_contact_is_not_hidden_by_larger_rack_pairs_and_pool_mapping_is_literal():
    env,_,records,lookup=setup()
    r=next(x for x in records if x['pool']==1 and x['source_body']=='flap_front')
    sensor=env.scene[r['sensor_name']];paths=sensor.cfg.filter_prim_paths_expr
    sensor.data.force_matrix_w[1,0,0,2]=60.
    other=next(p for p in r['other_box_paths'] if p.endswith('/flap_back'))
    sensor.data.force_matrix_w[1,0,paths.index(other),2]=6.
    original=deepcopy(env.scene['box1'].data)
    rows=startup_flap_contact_snapshot(env,records,lookup)
    assert len(rows)==12
    row=next(x for x in rows if x['environment']==1 and x['source_body']=='flap_front')
    assert row['logical_id']==0 and row['original_pool_id']==1
    assert row['strongest_normal_pairs'][0]['normal_force_magnitude_n']==60.
    assert row['maximum_other_box_pair_normal_force_n']==6. and row['nonzero_other_box_pair_count']==1
    pair=row['strongest_other_box_normal_pairs'][0]
    assert pair['target_asset']=='box0' and pair['target_body']=='flap_back'
    assert pair['target_pose_world'][0]==200. and row['source_pose_world'][0]==201.
    torch.testing.assert_close(env.scene['box1'].data.body_link_pose_w,original.body_link_pose_w)
    torch.testing.assert_close(env.scene['box1'].data.body_link_vel_w,original.body_link_vel_w)


def test_zero_other_box_force_and_nonfinite_evidence_are_distinct():
    env,_,records,lookup=setup()
    rows=startup_flap_contact_snapshot(env,records,lookup)
    assert all(r['maximum_other_box_pair_normal_force_n']==0. for r in rows)
    r=records[0];sensor=env.scene[r['sensor_name']]
    j=sensor.cfg.filter_prim_paths_expr.index(r['other_box_paths'][0])
    sensor.data.force_matrix_w[0,0,j,0]=float('nan')
    rows=startup_flap_contact_snapshot(env,records,lookup)
    row=next(x for x in rows if x['environment']==0 and x['logical_id']==0 and x['source_body']=='flap_front')
    assert row['maximum_other_box_pair_normal_force_n'] is None and row['other_box_nonfinite_filter_count']==1
    json.dumps(rows,allow_nan=False)
    assert torch.isnan(sensor.data.force_matrix_w[0,0,j,0])


@pytest.mark.parametrize('invalid',['two_sources','wrong_source','wrong_filter_axis'])
def test_filtered_contact_identity_and_axes_must_match_before_attribution(invalid):
    env,_,records,lookup=setup();sensor=env.scene[records[0]['sensor_name']]
    if invalid=='two_sources':sensor.body_names=['flap_front','flap_back']
    elif invalid=='wrong_source':sensor.body_names=['flap_back']
    else:sensor.data.force_matrix_w=sensor.data.force_matrix_w[:,:,:2]
    with pytest.raises(ValueError):startup_flap_contact_snapshot(env,records,lookup)


def test_duplicate_or_grouped_flap_source_is_rejected():
    _,cfg,_,_=setup()
    bad=[dict(asset_name='box',body='/box/Body',flaps={f:'/box/(flap_front|flap_back)' for f in FLAP_NAMES})]
    with pytest.raises(ValueError,match='literal'):add_startup_flap_contact_reporters(cfg,['body0'],bad)


def test_scene_namespace_expansion_matches_compiled_filter_axis_and_preserves_input():
    env,_,records,lookup=setup();original=deepcopy(records)
    for r in records:
        r['source_path']=r['source_path'].replace('/World/envs/env_.*','{ENV_REGEX_NS}')
        r['other_box_paths']=[p.replace('/World/envs/env_.*','{ENV_REGEX_NS}') for p in r['other_box_paths']]
    unresolved=deepcopy(records)
    resolved=resolve_startup_flap_contact_paths(records,'/World/envs/env_.*')
    assert resolved==original and records==unresolved
    assert len(startup_flap_contact_snapshot(env,resolved,lookup))==12
    with pytest.raises(ValueError,match='namespace'):resolve_startup_flap_contact_paths(records,'relative/path')


def test_focused_sources_keep_other_box_filters_and_literal_pool_identity():
    _,cfg,_,_=setup()
    paths=[dict(asset_name=f'box{i}',body=f'/box{i}/Body',flaps={f:f'/box{i}/{f}' for f in FLAP_NAMES}) for i in range(2)]
    # Start from existing body configs, without reusing prior flap reporters.
    fresh=SimpleNamespace(body0=cfg.body0,body1=cfg.body1)
    records=add_startup_flap_contact_reporters(fresh,['body0','body1'],paths,physical_pools=[1])
    assert len(records)==4 and all(r['pool']==1 for r in records)
    assert all(r['other_box_paths']==[paths[0]['body'],*paths[0]['flaps'].values()] for r in records)
    assert not hasattr(fresh,'reset_flap_pair_0_flap_front')
    for bad in ([],[1,1],[-1],[2]):
        with pytest.raises(ValueError,match='distinct valid'):
            add_startup_flap_contact_reporters(fresh,['body0','body1'],paths,physical_pools=bad)
