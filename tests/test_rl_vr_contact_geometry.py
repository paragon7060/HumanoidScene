"""Bent panels must change contact poses without migrating old observation semantics."""
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.vr_reference import vr_contact_geometry
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import matrix6,yaw_matrix
from kuavo_isaaclab_scene.rl.multi_box.observations.contracts import flap_observation_contract


def observed_panels():
    raw=torch.zeros(1,464)
    identity=matrix6(torch.eye(3))
    raw[:,50:68]=torch.cat((torch.zeros(3),identity)).repeat(2)
    raw[:,71:77]=identity
    raw[:,86]=1;raw[:,86+15:86+21]=identity
    raw[:,388]=1;raw[:,400]=1;raw[:,386]=1
    relations=raw[:,350:386].reshape(1,2,2,9)
    relations[...,3:]=identity
    relations[0,0,0,:3]=torch.tensor([.3,.1,.9])
    relations[0,1,1,:3]=torch.tensor([.3,-.1,.9])
    relations[0,0,0,3:]=matrix6(yaw_matrix(.6,raw))
    relations[0,1,1,3:]=matrix6(yaw_matrix(-.4,raw))
    return raw


def test_articulated_guide_rotates_each_hand_offset_in_its_panel_frame():
    raw=observed_panels();relative=torch.eye(3).repeat(2,1,1)
    offset=torch.tensor([[.03,0.,0.],[.02,0.,0.]])
    _,nominal,nominal_rotation,valid=vr_contact_geometry(raw,.5,relative,offset,'demo','nominal')
    _,actual,actual_rotation,_=vr_contact_geometry(raw,.5,relative,offset,'demo','articulated')
    centers=torch.tensor([[[.3,.1,.9],[.3,-.1,.9]]])
    rotations=torch.stack((yaw_matrix(.6,raw),yaw_matrix(-.4,raw)))[None]
    torch.testing.assert_close(nominal,centers+offset)
    torch.testing.assert_close(nominal_rotation,torch.eye(3).repeat(1,2,1,1))
    torch.testing.assert_close(actual,centers+(rotations@offset[None,...,None]).squeeze(-1))
    torch.testing.assert_close(actual_rotation,rotations)
    assert valid.all()


def test_center_goal_ignores_nominal_contact_calibration_but_keeps_live_panel_orientation():
    raw=observed_panels()
    _,goals,rotation,valid=vr_contact_geometry(raw,.5,torch.eye(3).repeat(2,1,1),
        torch.full((2,3),10.),'center','articulated')
    torch.testing.assert_close(goals,torch.tensor([[[.3,.1,.9],[.3,-.1,.9]]]))
    assert valid.all() and not torch.equal(rotation[:,0],rotation[:,1])


def test_missing_articulated_assignment_holds_current_tcp_and_forbids_closing():
    raw=observed_panels();raw[:,386:388]=0
    tcp,goals,rotation,valid=vr_contact_geometry(raw,.5,torch.eye(3).repeat(2,1,1),
        torch.ones(2,3),'demo','articulated')
    assert not valid.any()
    torch.testing.assert_close(goals,tcp[...,:3])
    torch.testing.assert_close(rotation,torch.eye(3).repeat(1,2,1,1))


def test_observation_metadata_distinguishes_same_width_contracts_and_excludes_contact_truth():
    nominal=flap_observation_contract('nominal');actual=flap_observation_contract('articulated')
    assert nominal['observation_contract']=='neutral_flap_center_controller_state_actual_base_twist_v2'
    assert actual['observation_contract']=='perceived_articulated_flap_center_controller_state_actual_base_twist_v3'
    for contract in (nominal,actual):
        assert contract['flap_perception_contract']['contact_force_or_success_in_actor'] is False
    assert not nominal['flap_perception_contract']['simulator_pose_proxy']
    assert actual['flap_perception_contract']['simulator_pose_proxy']
    with pytest.raises(ValueError):flap_observation_contract('unknown')
