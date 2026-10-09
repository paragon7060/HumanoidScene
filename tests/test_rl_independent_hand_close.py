"""Asynchronous TRAIN closure cannot bypass projection or bilateral lift gates."""
import pytest
import torch

from test_rl_upright_contact_sac import contact_fixture, contact_step
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import (
    URDFIndependentHandCloseSACPilot, URDFWholeArmClearanceSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates

FLAGS=dict(settled_close=True,precise_feedback=True,motion_feedback=True,
           upright_feedback=True,predictive_feedback=True,whole_arm_clearance=True)


def setup(*, individual=True, right_ready=False):
    pilot,raw,result,supp,prior=contact_fixture(ready=True)
    tracker=PerceivedContactExploration(128,raw,**FLAGS,independent_hand_close=individual)
    tracker.axes=prior.axes.clone()
    if not right_ready:supp[:,:36].reshape(2,2,2,9)[:,1,...,0]=.020
    return pilot,raw,result,supp,tracker


def test_new_option_keeps_original_thresholds_distribution_and_artifacts_distinct():
    old=perceived_contact_contract(**FLAGS)
    assert old==perceived_contact_contract(**FLAGS,independent_hand_close=False)
    new=perceived_contact_contract(**FLAGS,independent_hand_close=True)
    for key in ['close_point_tolerance_m','closed_reacquire_maximum_point_distance_m',
                'sustained_projected_closed_command_ticks_before_attempt_lift',
                'measured_minimum_closure_fraction','consecutive_measured_settled_ticks_before_attempt_lift',
                'original_nominal_jaw_gate_PD_limits_DR_reward_success_safety_preserved']:
        assert new[key]==old[key]
    assert URDFIndependentHandCloseSACPilot.agent_class is URDFWholeArmClearanceSACPilot.agent_class
    assert staged_policy_class(URDFIndependentHandCloseSACPilot.artifact_type) is URDFIndependentHandCloseSACPilot
    with pytest.raises(ValueError):perceived_contact_contract(independent_hand_close=True)
    with pytest.raises(ValueError):perceived_contact_contract(**FLAGS,independent_hand_close=1)


def test_prepared_hand_closes_while_other_waits_selected_only_with_exact_saved_goals():
    values=setup();pilot,raw,result,supp,tracker=values
    before=[raw.clone(),supp.clone(),torch.get_rng_state()]
    out=contact_step(values,0,torch.tensor([True,False]))
    assert torch.equal(out[1][2][0,19:21],torch.tensor([1.,-1.]))
    assert torch.equal(out[0][1],result[0][1]) and torch.equal(out[1][2][1],result[1][2][1])
    assert torch.equal(out[0][:,[0,1,2,3,22,23]],result[0][:,[0,1,2,3,22,23]])
    assert torch.equal(out[0],held_goal_coordinates(pilot.coordinates,raw,pilot.center+pilot.scale*out[1][2],pilot.stage))
    assert torch.isnan(out[1][1]).all() and torch.equal(before[2],torch.get_rng_state())
    assert torch.equal(raw,before[0]) and torch.equal(supp,before[1])
    assert tracker.individual_close_statistics['unilateral_closed_rows']==1


def test_one_settled_closed_hand_cannot_start_bilateral_hold_or_lift():
    values=setup();values[1][:,46:48]=.98;tracker=values[-1]
    for clock in range(45):contact_step(values,clock)
    assert tracker.hand_closing[[5,19],0].all() and not tracker.hand_closing[[5,19],1].any()
    assert not tracker.closed_ticks.any() and not tracker.settled_ticks.any()
    assert tracker.statistics['attempted_lift_episodes']==0
    values[3][:,:36].reshape(2,2,2,9)[:,1,...,0]=.001
    for clock in range(45,68):
        contact_step(values,clock)
        assert tracker.statistics['attempted_lift_episodes']==0
    contact_step(values,68)
    assert tracker.statistics['attempted_lift_episodes']==2


def test_each_hand_reacquires_without_reopening_its_still_prepared_partner():
    values=setup(right_ready=True);contact_step(values,0)
    values[3][:,:36].reshape(2,2,2,9)[:,0,...,0]=.020
    out=contact_step(values,1)
    assert out[1][2][:,19].eq(-1).all() and out[1][2][:,20].eq(1).all()
    assert not values[-1].closed_ticks.any() and not values[-1].settled_ticks.any()


def test_original_projection_interrupts_only_that_hand_latch():
    values=setup(right_ready=True);pilot=values[0];tracker=values[-1]
    contact_step(values,0)
    def gate(actor,goals):
        out=goals.clone();out[:,19]=-1;return out
    pilot.agent.action_projector=gate
    contact_step(values,1)
    assert not tracker.hand_closing[[5,19],0].any() and tracker.hand_closing[[5,19],1].all()
    assert not tracker.closed_ticks.any()
    pilot.agent.action_projector=lambda actor,goals:goals.clamp(-1,1)
    values[3][:,:36].reshape(2,2,2,9)[:,0,...,0]=.010
    out=contact_step(values,2)
    assert out[1][2][:,19].eq(-1).all()  # A rejected latch cannot reuse 12mm hysteresis.


def test_clock_reset_drops_individual_hysteresis_state():
    values=setup();tracker=values[-1];contact_step(values,10)
    values[3][:,:36].reshape(2,2,2,9)[:,0,...,0]=.010
    assert contact_step(values,11)[1][2][:,19].eq(1).all()
    out=contact_step(values,0)
    assert out[1][2][:,19:21].eq(-1).all() and not tracker.hand_closing.any()


def test_stationary_both_ready_preserves_old_commands_without_extra_randomness():
    old=setup(individual=False,right_ready=True);new=setup(right_ready=True)
    rng=torch.get_rng_state()
    for clock in range(10):
        a=contact_step(old,clock);b=contact_step(new,clock)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1][2],b[1][2])
    assert old[-1].statistics==new[-1].statistics and torch.equal(rng,torch.get_rng_state())


def test_unassisted_evaluation_never_constructs_contact_helper(monkeypatch):
    marker=object();monkeypatch.setattr(URDFStrongSuccessSACPilot,'act',lambda *a,**k:marker)
    pilot=object.__new__(URDFIndependentHandCloseSACPilot);pilot.training=False
    assert pilot.act(None,None,0) is marker
