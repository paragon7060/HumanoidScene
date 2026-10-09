"""Known geometry derivatives, pending-target cost and selected-only commands."""
import pytest
import torch

from test_rl_upright_contact_sac import contact_fixture, contact_step
from test_rl_cartesian_flap_probe import rotation6
from kuavo_isaaclab_scene.rl.multi_box.experiments.rack_entry_clearance import (
    RackEntryClearance, box_signed_distance_gradient, regularize_clearance,
    rack_clearance_contract, projected_pending_arm_lead,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.tensor_arm_kinematics import TensorArmKinematics, rotation
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import (
    URDFWholeArmClearanceSACPilot, URDFPredictiveContactSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates


def test_rotated_box_distance_gradient_matches_finite_difference_inside_and_outside():
    dtype=torch.float64
    centers=torch.tensor([[.3,-.2,.5]],dtype=dtype)
    R=rotation(torch.tensor([0.,0.,1.],dtype=dtype),torch.tensor([.4],dtype=dtype))
    half=torch.tensor([[.1,.2,.3]],dtype=dtype)
    local=torch.tensor([[[.13,.24,.02],[.02,.15,.01],[-.14,.02,.31]]],dtype=dtype)
    points=torch.einsum('bij,npj->npi',R,local)+centers
    d,g=box_signed_distance_gradient(points,centers,R,half)
    assert torch.allclose(d[0,:,0],torch.tensor([.05,-.05,(.04**2+.01**2)**.5],dtype=dtype),atol=1e-12)
    for col in range(3):
        moved=points.clone();moved[...,col]+=1e-7
        fd=(box_signed_distance_gradient(moved,centers,R,half)[0]-d)/1e-7
        assert torch.allclose(fd,g[...,col],atol=2e-6,rtol=0)


def test_palm_and_forearm_point_joint_jacobians_match_actual_URDF_FK():
    kin=TensorArmKinematics(dtype=torch.float64);geometry=RackEntryClearance(kin)
    q=torch.zeros(2,20,dtype=torch.float64);q[:,:3]=torch.tensor([-.2,.3,-.1])
    for hand,cols in enumerate(kin.columns):
        q[:,cols]=(kin.lower[hand]+kin.upper[hand])/2
    q[1,kin.columns[0]]+=.013
    rack=q.new_zeros(2,9);rack[:,:3]=q.new_tensor([.17,-.43,.11]);rack[:,3:]=rotation6(torch.eye(3,dtype=q.dtype).expand(2,3,3))
    measured=geometry.measure(q,rack);points=geometry.selected_points(q,measured['local_points'])
    for variable,col in enumerate(sum(kin.columns,[])):
        moved=q.clone();moved[:,col]+=1e-6
        fd=((geometry.selected_points(moved,measured['local_points'])-points)/1e-6*measured['root_normals']).sum(-1)
        assert torch.allclose(fd,measured['gradients'][...,variable],atol=3e-7,rtol=0)
    # Arm joints after elbow4 cannot move its cylinder proxy.
    assert measured['gradients'][:,2,4:7].eq(0).all()
    assert measured['gradients'][:,3,11:14].eq(0).all()


def test_soft_cost_respects_pending_inward_lead_and_keeps_far_tasks_exact():
    H=torch.eye(2).unsqueeze(0)*.0025;b=torch.tensor([[-.000025,0.]])
    g=torch.zeros(1,4,2);g[:,0,0]=1.
    distances=torch.tensor([[.012,.2,.2,.2]])
    lead=torch.tensor([[-.01,0.]])
    A,r,active=regularize_clearance(H,b,g,distances,lead)
    delta=torch.linalg.solve(A,r[...,None]).squeeze(-1)
    # Existing inward lead is cancelled enough to approach the 10mm margin.
    assert active.sum()==1 and .0095<distances[0,0]+lead[0,0]+delta[0,0]<.0105
    far=torch.ones(1,4)
    A,r,active=regularize_clearance(H,b,g,far,torch.zeros_like(lead))
    assert not active.any() and torch.equal(A,H) and torch.equal(r,b)
    with pytest.raises(ValueError):regularize_clearance(H,b,g,far*float('nan'),lead)


def test_variant_keeps_v7_gates_density_agent_and_default_contract():
    flags=dict(settled_close=True,precise_feedback=True,motion_feedback=True,upright_feedback=True,predictive_feedback=True)
    old=perceived_contact_contract(**flags)
    assert old==perceived_contact_contract(**flags,whole_arm_clearance=False)
    new=perceived_contact_contract(**flags,whole_arm_clearance=True)
    assert all(new[k]==v for k,v in old.items() if k!='name')
    assert new['whole_arm_rack_clearance']==rack_clearance_contract()
    assert URDFWholeArmClearanceSACPilot.agent_class is URDFPredictiveContactSACPilot.agent_class
    assert staged_policy_class(URDFWholeArmClearanceSACPilot.artifact_type) is URDFWholeArmClearanceSACPilot
    with pytest.raises(ValueError):perceived_contact_contract(whole_arm_clearance=True)
    with pytest.raises(ValueError):perceived_contact_contract(**flags,whole_arm_clearance=1)


def test_near_rack_commands_are_selected_only_bounded_and_decode_saved_goals():
    values=contact_fixture(ready=True);pilot,raw,result,supp,prior=values
    tracker=PerceivedContactExploration(128,raw,settled_close=True,precise_feedback=True,motion_feedback=True,
        upright_feedback=True,predictive_feedback=True,whole_arm_clearance=True)
    tracker.axes=prior.axes.clone()
    geometry=RackEntryClearance(tracker.kinematics)
    frames=geometry.frames(raw[:,:20]);p,R,_=frames[0]
    point=p[0]+R[0]@geometry.palms[0][0]
    raw[:,71:77]=rotation6(torch.eye(3).expand(2,3,3))
    raw[0,68:71]=point-torch.tensor([.021,.0,1.0825])
    result=(held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*result[1][2],pilot.stage),result[1])
    physical,(_,critic,goals)=contact_step((pilot,raw,result,supp,tracker),0,torch.tensor([True,False]))
    assert torch.equal(goals[1],result[1][2][1]) and torch.equal(physical[1],result[0][1])
    assert torch.equal(physical[:,[0,1,2,3,22,23]],result[0][:,[0,1,2,3,22,23]])
    assert torch.equal(physical,held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage))
    assert torch.isnan(critic).all() and goals.abs().max()<=1 and physical.abs().max()<=1
    assert tracker.upright_control.clearance.guided_rows==1
    assert tracker.upright_control.clearance.active_body_rows>0


def test_unassisted_evaluation_never_constructs_geometry_helper(monkeypatch):
    marker=object();monkeypatch.setattr(URDFStrongSuccessSACPilot,'act',lambda *a,**k:marker)
    pilot=object.__new__(URDFWholeArmClearanceSACPilot);pilot.training=False
    assert pilot.act(None,None,0) is marker


def test_geometry_fingerprint_is_checked_before_any_proposal(monkeypatch,tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments import rack_entry_clearance as module
    altered=tmp_path/'geometry.json';altered.write_text('{}')
    monkeypatch.setattr(module,'GEOMETRY_PATH',altered)
    with pytest.raises(ValueError,match='fingerprint'):
        RackEntryClearance(TensorArmKinematics())


def test_pending_geometry_matches_executed_goal_clip_and_saturated_sensitivity():
    kin=TensorArmKinematics(dtype=torch.float64);raw=torch.zeros(1,464,dtype=torch.float64)
    for h,columns in enumerate(kin.columns):raw[:,columns]=(kin.lower[h]+kin.upper[h])/2
    columns=sum(kin.columns,[])
    raw[:,416+columns[0]]=.48;raw[:,416+columns[1]]=-.48;raw[:,416+columns[2]]=.025
    lead,sensitivity=projected_pending_arm_lead(raw,kin)
    assert torch.allclose(lead[0,:3],raw.new_tensor([.08,-.08,.025]))
    assert torch.equal(sensitivity[0,:3],raw.new_tensor([0,0,1]))
    H=torch.eye(14,dtype=raw.dtype)[None];b=raw.new_zeros(1,14)
    full=raw.new_zeros(1,4,14);full[:,0,0]=1
    A,r,active=regularize_clearance(H,b,full*sensitivity[:,None],raw.new_tensor([[.001,.3,.3,.3]]),lead,pending_geometry_gradients=full)
    assert active[0,0] and torch.equal(A,H) and torch.equal(r,b)
