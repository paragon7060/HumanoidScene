"""Keep physical diagnostic geometry separate from neutral policy estimates."""
import math
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.debug.flap_geometry import compare_flap_centers
from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import nominal_flap_geometry


def inputs():
    box=torch.tensor([[1.,2.,3.,1.,0.,0.,0.]])
    size=torch.tensor([[.266,.30,.24]]);kind=torch.tensor([0])
    centers,_,axes=nominal_flap_geometry(size,kind)
    panels=box[:,None].expand(-1,2,-1).clone()
    tcp=panels.clone();tcp[...,:3]+=centers
    return box,panels,centers,axes,size,kind,tcp


def test_neutral_panels_equal_nominal_and_preserve_each_hand_pair():
    result=compare_flap_centers(*inputs())
    assert not result['center_error_m'].any()
    assert not result['normal_error_rad'].any()
    assert result['hand_actual_center_distance_m'].shape==(1,2,2)
    assert (result['hand_actual_center_distance_m'][0].diagonal()==0).all()
    assert result['hand_actual_center_distance_m'][0,0,1]>.2


def test_bent_panel_changes_actual_midpoint_normal_without_changing_nominal():
    values=list(inputs());before=compare_flap_centers(*values)
    values[1]=values[1].clone();values[1][0,0,:3]+=.02
    values[1][0,0,3:]=torch.tensor([math.cos(math.pi/8),0.,math.sin(math.pi/8),0.])
    after=compare_flap_centers(*values)
    assert torch.equal(after['nominal_center_pose_world'],before['nominal_center_pose_world'])
    assert after['center_error_m'][0,0]>.01
    assert after['normal_error_rad'][0,0]==pytest.approx(math.pi/4,abs=1e-6)
    assert not after['center_error_m'][0,1]


@pytest.mark.parametrize('bad',['shape','nonfinite','quaternion','axis'])
def test_invalid_geometry_cannot_be_presented_as_a_measurement(bad):
    values=list(inputs())
    if bad=='shape':values[1]=values[1][:,:1]
    elif bad=='nonfinite':values[1][0,0,0]=float('nan')
    elif bad=='quaternion':values[1][0,0,3:]=0
    else:values[3]=values[3].clone();values[3][0,0]=3
    with pytest.raises(ValueError):compare_flap_centers(*values)
