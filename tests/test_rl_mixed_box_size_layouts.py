"""Size resets must select real supported assets without fabricating experience."""
from dataclasses import replace
import importlib.util
from pathlib import Path

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.demo_replay import load_v2_grasp_demonstrations, _rotation_matrix
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import (
    GraspLayout, layout_reset_observation, validate_layout_footprints,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import select_reference_episode
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells, physical_pool_id
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M


ROOT=Path(__file__).resolve().parents[1]


def front_beam_separation(actor, token):
    """SAT of the actual body envelope and the authored shelf02 front beam."""
    rack=_rotation_matrix(actor[71:77])
    rotation=rack.T@_rotation_matrix(token[15:21])
    position=rack.T@(token[12:15]-actor[68:71])
    width,depth,height=token[5:8]
    corners=torch.cartesian_prod(torch.tensor([-1.,1.]),torch.tensor([-1.,1.]),torch.tensor([-.005,.995]))
    corners=corners*torch.stack((width/2,depth/2,height))
    body=corners@rotation.T+position
    # Verified from rack_roller_runtime.usda, not from the reset implementation.
    beam=torch.cartesian_prod(torch.tensor([-.77,-.07]),torch.tensor([-.020426,-.010426]),
                             torch.tensor([1.047287,1.087287]))
    axes=torch.eye(3);other=rotation.T
    cross=torch.linalg.cross(axes[:,None,:],other[None,:,:]).reshape(-1,3)
    axes=torch.cat((axes,other,cross));length=axes.norm(dim=-1)
    axes=axes[length>1e-7]/length[length>1e-7,None]
    a=body@axes.T;b=beam@axes.T
    overlap=torch.minimum(a.max(0).values,b.max(0).values)-torch.maximum(a.min(0).values,b.min(0).values)
    return float(-overlap.min())


def recipe_module():
    spec=importlib.util.spec_from_file_location('mixed_size_recipe',ROOT/'scripts/rl/prepare_region_grasp_layouts.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def source_rows():
    batch,_=load_v2_grasp_demonstrations(ROOT/'examples/demos/v2_grasp_quest_success.hdf5',self_collision_enabled=False)
    return [select_reference_episode(batch,i)['actor_obs'][0] for i in range(2)]


def test_mixed_size_recipe_covers_real_middle_assets_and_keeps_all_randomized_requests(tmp_path):
    module=recipe_module();recipe=module.prepare(tmp_path/'mixed',920000,32,32,32,mixed_middle_box_types=True)
    sources=source_rows();saved=[x.clone() for x in sources];spec=MultiBoxSpec();cells=logical_cells(spec)
    seeds={};counts={}
    for row in recipe['records']:
        layout=GraspLayout(**row['layout']).validate();source=sources[row['reference_episode_index']]
        reset=layout_reset_observation(source,layout,spec)
        target=int(reset[400:412].argmax());token=reset[86:350].reshape(12,22)[target]
        kind=('small','medium')[int(token[3:5].argmax())]
        assert kind==layout.target_box_type and cells[target].region_name==layout.target_region
        torch.testing.assert_close(token[5:8],torch.tensor(BOX_DIMENSIONS_M[kind]),atol=1e-6,rtol=0)
        assert physical_pool_id(cells[target],int(token[3:5].argmax())) in range(18)
        validate_layout_footprints(reset)
        if kind=='medium':
            # Both halves and every original split/yaw/lateral/base draw must
            # start outside the physical beam, including its2mm contact skin.
            assert front_beam_separation(reset,token)>.002
        assert torch.equal(reset[:20],source[:20])  # Same neutral joints, no success reset state.
        assert -.04<=layout.lateral_m<=-.02
        seeds.setdefault(row['group'],set()).add(layout.seed)
        key=(row['group'],layout.target_region,kind);counts[key]=counts.get(key,0)+1
    for group in ('train','development','holdout'):
        for region in module.REGIONS:
            assert counts[group,region,'small']==(32 if region.startswith('shelf_3') else 16)
            assert counts.get((group,region,'medium'),0)==(0 if region.startswith('shelf_3') else 16)
    assert not(seeds['train']&seeds['development'] or seeds['train']&seeds['holdout'] or seeds['development']&seeds['holdout'])
    assert all(torch.equal(a,b) for a,b in zip(sources,saved))
    assert recipe['box_fixed'] is False and recipe['synthetic_Q_transitions'] is False
    assert recipe['measured_size_specific_waypoints_required_before_matching_SAC'] is True


def test_size_change_preserves_source_bottom_plane_and_orientation():
    source=source_rows()[0];spec=MultiBoxSpec()
    layout=GraspLayout(930000,'train',0.,target_region='shelf_2_right')
    old=layout_reset_observation(source,layout,spec)
    new=layout_reset_observation(source,replace(layout,target_box_type='medium'),spec)
    target=int(old[400:412].argmax());a=old[86:350].reshape(12,22)[target];b=new[86:350].reshape(12,22)[target]
    normal=_rotation_matrix(a[15:21])[:,2]
    old_bottom=a[12:15]-normal*.005*a[7];new_bottom=b[12:15]-normal*.005*b[7]
    torch.testing.assert_close(normal.dot(old_bottom),normal.dot(new_bottom),atol=2e-7,rtol=0)
    torch.testing.assert_close(a[15:21],b[15:21],atol=0,rtol=0)
    assert int(b[3:5].argmax())==1


@pytest.mark.parametrize('region',['shelf_2_left','shelf_2_right'])
def test_larger_body_at_small_root_penetrates_beam_but_neutral_size_reset_does_not(region):
    source=source_rows()[0];spec=MultiBoxSpec()
    layout=GraspLayout(930001,'train',0.,target_region=region)
    small=layout_reset_observation(source,layout,spec)
    old=small[86:350].reshape(12,22)[int(small[400:412].argmax())].clone()
    old[5:8]=torch.tensor(BOX_DIMENSIONS_M['medium'])
    assert front_beam_separation(small,old)<-.001
    medium=layout_reset_observation(source,replace(layout,target_box_type='medium'),spec)
    new=medium[86:350].reshape(12,22)[int(medium[400:412].argmax())]
    assert front_beam_separation(medium,new)>.002
    same=layout_reset_observation(source,replace(layout,target_box_type='small'),spec)
    assert torch.equal(small,same)


@pytest.mark.parametrize('kind,region', [('large','shelf_2_right'),('medium','shelf_3_left'),('medium',None)])
def test_unsupported_physical_size_region_pairs_fail_before_generation(kind,region):
    with pytest.raises(ValueError):GraspLayout(1,'train',0.,target_region=region,target_box_type=kind).validate()


def test_legacy_recipes_omit_size_override_and_odd_mixed_counts_are_rejected(tmp_path):
    module=recipe_module();old=module.prepare(tmp_path/'old',940000,2,1,2)
    assert all('target_box_type' not in r['layout'] for r in old['records'])
    with pytest.raises(ValueError,match='even'):
        module.prepare(tmp_path/'bad',940000,2,1,2,mixed_middle_box_types=True)
    assert not (tmp_path/'bad').exists()
