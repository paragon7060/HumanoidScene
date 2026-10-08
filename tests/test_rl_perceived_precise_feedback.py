"""Avoid empty closure after a perceived flap moves away from a fixed point."""
import pytest
import torch

from test_rl_perceived_contact_exploration import fixture
from test_rl_cartesian_flap_probe import rotation6
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration,perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_perceived_contact_sac import URDFPreciseFeedbackSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def setup(distance=.001):
    pilot,raw,result,supp,old=fixture(ready=True)
    supp[:,:36].reshape(2,2,2,9)[...,:3]=torch.tensor([distance,0,0])
    tracker=PerceivedContactExploration(128,raw,settled_close=True,precise_feedback=True)
    tracker.axes=old.axes.clone()
    return pilot,raw,result,supp,tracker


def step(values,clock,chosen=None):
    pilot,raw,result,supp,tracker=values
    return tracker.step(pilot,raw,result,torch.tensor([5,19]),supp,
        torch.ones(2,dtype=torch.bool) if chosen is None else chosen,clock)


def test_twelve_mm_is_not_inside_precise_closing_gate_even_if_source_jaws_close():
    values=setup(.012);tracker=values[-1]
    for clock in range(30):
        _,(_,_,goals)=step(values,clock)
        assert goals[:,19:21].eq(-1).all()
    assert tracker.phase[[5,19]].eq(1).all()
    assert tracker.closed_ticks.sum()==tracker.statistics['attempted_lift_episodes']==0


def test_contact_goal_follows_current_panel_during_closure():
    values=setup();supp=values[3];tracker=values[-1]
    first=step(values,0)[1][2]
    supp[:,:36].reshape(2,2,2,9)[...,0]+=.003
    second=step(values,1)[1][2]
    assert tracker.closing[[5,19]].all()
    assert second[:,19:21].eq(1).all()
    assert not torch.equal(first[:,1:15],second[:,1:15])


def test_lost_geometry_reopens_both_jaws_and_restarts_settling():
    values=setup();raw=values[1];supp=values[3];tracker=values[-1]
    raw[:,46:48]=.98
    for clock in range(10):step(values,clock)
    assert tracker.closed_ticks[[5,19]].eq(10).all()
    supp[:,:36].reshape(2,2,2,9)[...,0]+=.02
    _,(_,_,goals)=step(values,10)
    assert goals[:,19:21].eq(-1).all()
    assert not tracker.closing.any() and not tracker.closed_ticks.any() and not tracker.settled_ticks.any()
    assert tracker.statistics['attempted_lift_episodes']==0


def test_axis_feedback_continues_while_closing_before_lift():
    values=setup();supp=values[3]
    first=step(values,0)[1][2]
    angle=torch.tensor(.10);c,s=angle.cos(),angle.sin()
    R=torch.stack((c,c*0,s,c*0,c*0+1,c*0,-s,c*0,c)).reshape(3,3)
    supp[:,:36].reshape(2,2,2,9)[...,3:]=rotation6(R)
    second=step(values,1)[1][2]
    assert second[:,19:21].eq(1).all()
    assert not torch.equal(first[:,1:15],second[:,1:15])


def test_closed_command_does_not_replace_measured_settling_or_24_tick_hold():
    values=setup();raw=values[1];tracker=values[-1];raw[:,46:48]=.5
    for clock in range(40):
        step(values,clock);assert tracker.statistics['attempted_lift_episodes']==0
    raw[:,46:48]=.98
    for clock in range(40,46):
        step(values,clock);assert tracker.statistics['attempted_lift_episodes']==0
    step(values,46);assert tracker.statistics['attempted_lift_episodes']==2


def test_selection_actual_actions_other_channels_and_inputs_are_preserved():
    values=setup(.012);pilot,raw,result,supp,tracker=values
    before=(raw.clone(),supp.clone(),torch.get_rng_state())
    physical,(actor,critic,goals)=step(values,0,torch.tensor([True,False]))
    assert torch.equal(physical[1],result[0][1]) and torch.equal(goals[1],result[1][2][1])
    assert torch.equal(physical[:,[0,1,2,3,18,19,22,23]],result[0][:,[0,1,2,3,18,19,22,23]])
    assert torch.equal(held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage),physical)
    assert torch.isnan(critic).all() and goals.abs().max()<=1
    assert all(torch.equal(a,b) for a,b in zip(before,(raw,supp,torch.get_rng_state())))


def test_distinct_artifact_and_frozen_eval_never_uses_contact_feedback(monkeypatch):
    old=perceived_contact_contract(settled_close=True)
    new=perceived_contact_contract(settled_close=True,precise_feedback=True)
    assert old['close_point_tolerance_m']==.018 and 'world_frame_close_latch' not in old
    assert new['close_point_tolerance_m']==.006 and not new['world_frame_close_latch']
    assert not new['privileged_pinch_contact_reward_or_success_inputs']
    assert staged_policy_class(URDFPreciseFeedbackSACPilot.artifact_type) is URDFPreciseFeedbackSACPilot
    marker=object();monkeypatch.setattr(URDFStrongSuccessSACPilot,'act',lambda *args,**kwargs:marker)
    pilot=object.__new__(URDFPreciseFeedbackSACPilot);pilot.training=False
    assert pilot.act(None,None,0) is marker
    with pytest.raises(ValueError):perceived_contact_contract(precise_feedback=True)
