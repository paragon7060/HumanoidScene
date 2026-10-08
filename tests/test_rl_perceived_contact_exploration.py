"""Actual action identity, deployable inputs, stable IDs and no eval assistance."""
from types import SimpleNamespace

import pytest
import torch

from test_rl_cartesian_flap_probe import region_fixture, rotation6
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract, contact_statistics,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_perceived_contact_sac import URDFPerceivedContactSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def fixture(*,ready=False):
    pilot,raw,_,original,supplemental=region_fixture()
    tracker=PerceivedContactExploration(128,raw)
    for hand,cols in enumerate(tracker.kinematics.columns):
        raw[:,cols]=((tracker.kinematics.lower[hand]+tracker.kinematics.upper[hand])/2).clamp(-.4,.4)
    p,R,_=tracker.kinematics.fk(raw[:,:20])
    raw[:,50:68]=torch.cat((p,rotation6(R)),-1).flatten(1)
    joint,torso,_,_,_=pilot.coordinates.current(raw)
    pilot.center=torch.cat((joint[0],torso[0],torch.zeros(2)))
    pilot.center[1:15]=((tracker.kinematics.lower+tracker.kinematics.upper)/2).flatten()
    pilot.scale=torch.ones(21)
    pilot.scale[1:15]=((tracker.kinematics.upper-tracker.kinematics.lower)/2).flatten()
    pilot.agent.action_projector=lambda actor,requested:requested.clamp(-1,1)
    critic=torch.full_like(original[1][1],float('nan'))
    goals=original[1][2].clone()
    physical=held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage)
    result=physical,(original[1][0],critic,goals)
    if ready:
        tracker.axes[:]=torch.tensor([1.,0,0])
        relation=supplemental[:,:36].reshape(2,2,2,9)
        relation[...,:3]=torch.tensor([.001,0,0])
        raw[:,69]=p[...,1].amin(-1)-tracker.front_y-.3
        physical=held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage)
        result=physical,result[1]
    return pilot,raw,result,supplemental,tracker


def test_actual_bounded_actions_exactly_decode_and_other_channels_inputs_RNG_unchanged():
    pilot,raw,original,supplemental,tracker=fixture()
    ids=torch.tensor([5,19]);chosen=torch.tensor([True,False])
    before=(raw.clone(),supplemental.clone(),original[0].clone(),original[1][2].clone(),torch.get_rng_state())
    physical,(actor,critic,goals)=tracker.step(pilot,raw,original,ids,supplemental,chosen,torch.tensor([100,100]))
    assert tracker.phase[5]>=0 and tracker.phase[19]==-1 and tracker.phase[0]==-1
    assert torch.equal(physical[1],original[0][1]) and torch.equal(goals[1],original[1][2][1])
    assert torch.equal(physical[:,[0,1,2,3,18,19,22,23]],original[0][:,[0,1,2,3,18,19,22,23]])
    assert goals.abs().max()<=1 and goals[:,19:].abs().eq(1).all()
    decoded=held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*goals,pilot.stage)
    assert torch.equal(decoded,physical)
    assert actor is original[1][0] and critic is original[1][1]
    assert torch.isnan(critic).all()  # Critic/contact inputs were never read.
    assert all(torch.equal(a,b) for a,b in zip(before,[raw,supplemental,original[0],original[1][2],torch.get_rng_state()]))


def test_close_command_attempts_lift_without_claiming_real_pinch_and_expires():
    pilot,raw,result,supplemental,tracker=fixture(ready=True)
    ids=torch.tensor([5,19]);chosen=torch.ones(2,dtype=torch.bool)
    for clock in range(7):
        _,(_,_,goals)=tracker.step(pilot,raw,result,ids,supplemental,chosen,clock)
        assert goals[:,19:].eq(1).all() and tracker.statistics['attempted_lift_episodes']==0
    tracker.step(pilot,raw,result,ids,supplemental,chosen,7)
    assert tracker.statistics['attempted_lift_episodes']==2
    assert tracker.phase[ids].eq(2).all()
    assert perceived_contact_contract()['commanded_close_NOT_confirmed_pinch_or_success']
    assert 'confirmed_lift_episodes' not in tracker.report()
    assert tracker.step(pilot,raw,result,ids,supplemental,chosen,47) is result
    assert tracker.phase[ids].eq(3).all() and tracker.statistics['expired_attempts']==2
    tracker.step(pilot,raw,result,ids,supplemental,chosen,0)
    assert tracker.phase[ids].lt(2).all() and tracker.closed_ticks[ids].eq(1).all()


def test_production_jaw_gate_cannot_be_overridden_by_the_contact_attempt():
    pilot,raw,result,supplemental,tracker=fixture(ready=True)
    def projection(actor,goals):
        out=goals.clamp(-1,1).clone();out[:,20]=-1;return out
    pilot.agent.action_projector=projection
    for clock in range(12):
        _,(_,_,goals)=tracker.step(pilot,raw,result,torch.tensor([5,19]),supplemental,torch.ones(2,dtype=torch.bool),clock)
        assert goals[:,20].eq(-1).all()
    assert tracker.statistics['attempted_lift_episodes']==0 and tracker.closed_ticks.sum()==0


def test_selected_bad_TCP_and_unbounded_legacy_coordinates_rejected():
    pilot,raw,result,supplemental,tracker=fixture()
    raw[:,50]+=.02
    with pytest.raises(ValueError,match='does not match'):
        tracker.step(pilot,raw,result,torch.tensor([5,19]),supplemental,torch.ones(2,dtype=torch.bool),0)
    pilot,raw,result,supplemental,tracker=fixture()
    pilot.scale[1:15]*=.001
    with pytest.raises(ValueError,match='bounded full URDF'):
        tracker.step(pilot,raw,result,torch.tensor([5,19]),supplemental,torch.ones(2,dtype=torch.bool),0)


def test_frozen_eval_never_invokes_geometry_or_selection(monkeypatch):
    marker=object()
    monkeypatch.setattr(URDFStrongSuccessSACPilot,'act',lambda *args,**kwargs:marker)
    pilot=object.__new__(URDFPerceivedContactSACPilot)
    pilot.training=False
    assert pilot.act(None,None,0) is marker
    assert staged_policy_class(URDFPerceivedContactSACPilot.artifact_type) is URDFPerceivedContactSACPilot
    assert staged_policy_class(URDFStrongSuccessSACPilot.artifact_type) is URDFStrongSuccessSACPilot


def test_unknown_statistics_and_privileged_inputs_are_not_part_of_the_contract():
    c=perceived_contact_contract()
    assert not c['privileged_pinch_contact_reward_or_success_inputs'] and not c['curriculum']
    assert c['actor_Q_targets_density_entropy_and_greedy_eval_unchanged']
    with pytest.raises(ValueError):contact_statistics({'attempted_lift_episodes':1})
