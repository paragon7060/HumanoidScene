"""Prevent lift during actual jaw travel and avoid chasing a moving panel."""
import pytest
import torch

from test_rl_perceived_contact_exploration import fixture
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration,perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_perceived_contact_sac import URDFSettledContactSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def setup():
    pilot,raw,result,supplemental,old=fixture(ready=True)
    tracker=PerceivedContactExploration(128,raw,settled_close=True)
    tracker.axes=old.axes.clone()
    return pilot,raw,result,supplemental,tracker


def step(values,clock):
    pilot,raw,result,supplemental,tracker=values
    return tracker.step(pilot,raw,result,torch.tensor([5,19]),supplemental,
        torch.ones(2,dtype=torch.bool),clock)


def test_closed_commands_with_half_closed_measured_jaws_cannot_trigger_lift():
    values=setup();raw=values[1];tracker=values[-1]
    raw[:,46:48]=.5
    for clock in range(40):step(values,clock)
    assert tracker.closed_ticks[[5,19]].eq(40).all()
    assert tracker.statistics['attempted_lift_episodes']==0
    assert tracker.settled_ticks[[5,19]].eq(0).all()
    raw[:,46:48]=.98
    for clock in range(40,46):
        step(values,clock)
        assert tracker.statistics['attempted_lift_episodes']==0
    step(values,46)
    assert tracker.statistics['attempted_lift_episodes']==2


def test_position_settling_does_not_replace_minimum_closed_hold_time():
    values=setup();raw=values[1];tracker=values[-1];raw[:,46:48]=.98
    for clock in range(23):
        step(values,clock)
        assert tracker.statistics['attempted_lift_episodes']==0
    step(values,23)
    assert tracker.statistics['attempted_lift_episodes']==2
    assert tracker.phase[[5,19]].eq(2).all()


def test_high_but_still_moving_closure_restarts_settling_window():
    values=setup();raw=values[1];tracker=values[-1]
    for clock in range(40):
        raw[:,46:48]=.94 if clock%2 else .98
        step(values,clock)
    assert tracker.statistics['attempted_lift_episodes']==0
    assert tracker.settled_ticks.sum()==0


def test_panel_motion_keeps_contact_target_fixed_and_prevents_empty_lift():
    values=setup();raw=values[1];supp=values[3];tracker=values[-1]
    raw[:,46:48]=.98
    first=step(values,0)[1][2]
    before=tracker.close_targets_rack.clone()
    supp[:,:36].reshape(2,2,2,9)[...,:3]+=.03
    for clock in range(1,30):
        current=step(values,clock)[1][2]
        assert torch.equal(current[:,1:15],first[:,1:15])
    assert torch.equal(before,tracker.close_targets_rack)
    assert tracker.statistics['attempted_lift_episodes']==0


def test_original_jaw_gate_interrupts_latched_closure():
    values=setup();pilot=values[0];tracker=values[-1]
    step(values,0);assert tracker.closing[[5,19]].all()
    def gate(actor,goals):
        out=goals.clamp(-1,1).clone();out[:,20]=-1;return out
    pilot.agent.action_projector=gate
    step(values,1)
    assert not tracker.closing.any() and not tracker.closed_ticks.any()
    assert tracker.statistics['attempted_lift_episodes']==0


def test_explicit_variant_keeps_v1_contract_and_checkpoints_distinct():
    old=perceived_contact_contract();new=perceived_contact_contract(settled_close=True)
    assert old['sustained_projected_closed_command_ticks_before_attempt_lift']==8
    assert old['max_guided_control_ticks']==180
    assert new['sustained_projected_closed_command_ticks_before_attempt_lift']==24
    assert new['measured_jaw_settling_required']
    assert not new['privileged_pinch_contact_reward_or_success_inputs']
    assert new['settled_closure_NOT_confirmed_pinch_or_success']
    assert staged_policy_class(URDFSettledContactSACPilot.artifact_type) is URDFSettledContactSACPilot
    with pytest.raises(ValueError):perceived_contact_contract(settled_close=1)
