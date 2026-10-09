"""Short measured-panel anticipation; actual gates and unassisted SAC stay intact."""
import pytest
import torch

from test_rl_upright_contact_sac import contact_fixture, contact_step
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, bounded_panel_prediction, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import (
    URDFPredictiveContactSACPilot, URDFInteriorContactSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates


def pair():
    old=contact_fixture(ready=True)
    pilot,raw,result,supp,prior=contact_fixture(ready=True)
    new=PerceivedContactExploration(128,raw,settled_close=True,precise_feedback=True,
        motion_feedback=True,upright_feedback=True,predictive_feedback=True)
    new.axes=prior.axes.clone()
    return old,(pilot,raw,result,supp,new)


def test_predictive_variant_preserves_original_gates_limits_and_SAC_agent():
    flags=dict(settled_close=True,precise_feedback=True,motion_feedback=True,
               upright_feedback=True)
    old=perceived_contact_contract(**flags)
    assert old==perceived_contact_contract(**flags,predictive_feedback=False)
    new=perceived_contact_contract(**flags,predictive_feedback=True)
    assert all(new[k]==v for k,v in old.items() if k!='name')
    assert new['closed_panel_prediction_horizon_control_ticks']==3
    assert new['closed_panel_prediction_maximum_offset_m']==.008
    assert new['closing_and_lift_gates_use_current_unpredicted_perception']
    assert URDFPredictiveContactSACPilot.agent_class is URDFInteriorContactSACPilot.agent_class
    assert staged_policy_class(URDFPredictiveContactSACPilot.artifact_type) is URDFPredictiveContactSACPilot
    with pytest.raises(ValueError):perceived_contact_contract(predictive_feedback=True)
    with pytest.raises(ValueError):perceived_contact_contract(**flags,predictive_feedback=1)
    with pytest.raises(ValueError):perceived_contact_contract(**flags,interior_contact=True,predictive_feedback=True)


def test_prediction_tracks_known_linear_motion_and_caps_bilateral_outliers():
    delta=torch.tensor([[[.001,0,0],[.010,.010,0]],[[0,0,0],[-.002,0,0]]])
    offset=bounded_panel_prediction(delta,horizon_ticks=3,maximum_offset_m=.008)
    assert torch.allclose(offset[0,0],torch.tensor([.003,0,0]))
    assert torch.allclose(offset[1,1],torch.tensor([-.006,0,0]))
    assert offset.norm(dim=-1).max()<=.008001
    assert torch.equal(offset[1,0],torch.zeros(3))
    assert torch.allclose(offset[0,1]/offset[0,1].norm(),delta[0,1]/delta[0,1].norm())
    with pytest.raises(ValueError):bounded_panel_prediction(delta*float('nan'),horizon_ticks=3,maximum_offset_m=.008)


def test_stationary_closing_matches_previous_commands_without_extra_RNG():
    old,new=pair();rng=torch.get_rng_state()
    for clock in range(5):
        a=contact_step(old,clock);b=contact_step(new,clock)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1][2],b[1][2])
    assert torch.equal(rng,torch.get_rng_state())
    assert new[-1].statistics==old[-1].statistics
    assert new[-1].phase[torch.tensor([5,19])].eq(1).all()


def test_predicted_closing_is_selected_only_bounded_and_records_exact_executed_goals():
    old,new=pair();chosen=torch.tensor([True,False])
    for values in (old,new):contact_step(values,0,chosen)
    for values in (old,new):values[3][:,:36].reshape(2,2,2,9)[0,...,1]+=.001
    a=contact_step(old,1,chosen);b=contact_step(new,1,chosen)
    pilot,raw,result,supp,tracker=new
    assert not torch.equal(a[1][2][0,1:15],b[1][2][0,1:15])
    assert torch.equal(a[1][2][:,19:21],b[1][2][:,19:21])
    assert torch.equal(a[1][2][:,17:19],b[1][2][:,17:19])
    assert torch.equal(b[0][1],result[0][1]) and torch.equal(b[1][2][1],result[1][2][1])
    assert torch.equal(b[0][:,[0,1,2,3,22,23]],result[0][:,[0,1,2,3,22,23]])
    assert torch.equal(b[0],held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*b[1][2],pilot.stage))
    assert b[0].abs().max()<=1 and b[1][2].abs().max()<=1
    assert torch.isnan(b[1][1]).all()  # Privileged critic is not a prediction input.


def test_skipped_clock_does_not_extrapolate_stale_motion():
    old,new=pair()
    for values in (old,new):contact_step(values,0)
    for values in (old,new):values[3][:,:36].reshape(2,2,2,9)[...,1]+=.001
    a=contact_step(old,2);b=contact_step(new,2)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1][2],b[1][2])


def test_prediction_never_overrides_actual_distance_reopening_or_unsettled_lift_gate():
    old,new=pair()
    for values in (old,new):contact_step(values,0)
    for values in (old,new):values[3][:,:36].reshape(2,2,2,9)[...,0]+=.02
    a=contact_step(old,1);b=contact_step(new,1)
    assert torch.equal(a[0],b[0]) and torch.equal(a[1][2],b[1][2])
    assert b[1][2][:,19:21].eq(-1).all()
    assert new[-1].statistics['attempted_lift_episodes']==0


def test_predictive_evaluation_does_not_call_contact_helper(monkeypatch):
    marker=object()
    monkeypatch.setattr(URDFStrongSuccessSACPilot,'act',lambda *a,**k:marker)
    pilot=object.__new__(URDFPredictiveContactSACPilot);pilot.training=False
    assert pilot.act(None,None,0) is marker
