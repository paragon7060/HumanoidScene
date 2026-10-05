from copy import deepcopy
from types import SimpleNamespace

import pytest

from kuavo_isaaclab_scene.rl.multi_box.scene.reset_replication import (
    configure_independent_scene_probe, validate_independent_scene_request,
    verify_independent_scene_probe,
)
from test_rl_reset_world_frame import passive_probe


def test_independent_scene_requires_explicit_matched_state_and_frozen_DEV():
    waves=[dict(split='validation',layouts=[dict(layout=dict(seed=123))])]
    base=dict(enabled=True,reset_enabled=True,training=False,steps=1)
    validate_independent_scene_request(waves,passive_probe(),**base)
    for changed in (dict(training=True),dict(reset_enabled=False),dict(steps=2),dict(other_probe=True)):
        with pytest.raises(ValueError):validate_independent_scene_request(waves,passive_probe(),**(base|changed))
    with pytest.raises(ValueError):validate_independent_scene_request(waves,None,**base)
    frame=passive_probe();frame['probe_type']='original_world_frame'
    with pytest.raises(ValueError):validate_independent_scene_request(waves,frame,**base)
    with pytest.raises(ValueError):validate_independent_scene_request([waves[0]|dict(split='holdout')],passive_probe(),**base)


def test_independent_scene_changes_replication_only_and_preserves_default():
    scene=SimpleNamespace(replicate_physics=True,filter_collisions=True,clone_in_fabric=False,num_envs=16)
    cfg=SimpleNamespace(scene=scene,sim=SimpleNamespace(dt=1/120),decimation=4)
    assert configure_independent_scene_probe(cfg,enabled=False) is None and scene.replicate_physics
    audit=configure_independent_scene_probe(cfg,enabled=True)
    assert not scene.replicate_physics and scene.filter_collisions and not scene.clone_in_fabric
    assert cfg.sim.dt==1/120 and cfg.decimation==4 and scene.num_envs==16
    assert audit['Q_import_eligible'] is False


@pytest.mark.parametrize('changed',[dict(filter_collisions=False),dict(clone_in_fabric=True),dict(replicate_physics=False)])
def test_nonstandard_source_flags_rejected_before_mutation(changed):
    scene=SimpleNamespace(**(dict(replicate_physics=True,filter_collisions=True,clone_in_fabric=False)|changed))
    original=deepcopy(vars(scene))
    with pytest.raises(ValueError):configure_independent_scene_probe(SimpleNamespace(scene=scene),enabled=True)
    assert vars(scene)==original


def collision_scene(global_invert=False,group_invert=False):
    from pxr import Sdf, Usd, UsdPhysics
    stage=Usd.Stage.CreateInMemory();stage.DefinePrim('/World/collisions','Scope')
    physics_scene=UsdPhysics.Scene.Define(stage,'/World/physicsScene').GetPrim()
    physics_scene.CreateAttribute('physxScene:invertCollisionGroupFilter',Sdf.ValueTypeNames.Bool).Set(global_invert)
    paths=['/World/envs/env_0','/World/envs/env_1']
    group_paths=['/World/collisions/env0','/World/collisions/env1']
    for i,path in enumerate(paths):
        stage.DefinePrim(path,'Xform')
        group=UsdPhysics.CollisionGroup.Define(stage,group_paths[i])
        group.GetCollidersCollectionAPI().CreateIncludesRel().SetTargets([path])
        group.CreateInvertFilteredGroupsAttr().Set(group_invert)
        group.CreateFilteredGroupsRel().SetTargets([group_paths[i if global_invert or group_invert else 1-i]])
    flags=SimpleNamespace(replicate_physics=False,filter_collisions=True,clone_in_fabric=False)
    return SimpleNamespace(scene=SimpleNamespace(cfg=flags,env_prim_paths=paths,physics_scene_path='/World/physicsScene'),sim=SimpleNamespace(stage=stage))


@pytest.mark.parametrize('global_invert,group_invert',[(False,False),(True,False),(False,True)])
def test_actual_USD_collision_groups_cover_all_independent_environment_pairs(global_invert,group_invert):
    env=collision_scene(global_invert,group_invert)
    audit=dict(requested_scene_config=dict(replicate_physics=False,filter_collisions=True,clone_in_fabric=False))
    result=verify_independent_scene_probe(env,audit)
    assert result['all_environment_pairs_collision_filtered']
    assert len(result['actual_environment_collision_groups'])==2


def test_missing_collision_filter_is_rejected_even_if_config_flag_is_true():
    from pxr import UsdPhysics
    env=collision_scene();group=UsdPhysics.CollisionGroup.Get(env.sim.stage,'/World/collisions/env0')
    group.GetFilteredGroupsRel().SetTargets([])
    audit=dict(requested_scene_config=dict(replicate_physics=False,filter_collisions=True,clone_in_fabric=False))
    with pytest.raises(ValueError,match='filtering is incomplete'):verify_independent_scene_probe(env,audit)
