"""Distinguish stalled perception gates without assisting control or eval."""
import torch
from test_rl_perceived_precise_feedback import setup, step


def counts(values):
    return values[-1].gate_report()['counts']


def test_missing_precise_position_is_distinct_from_missing_axis_and_closure():
    values=setup(.012)
    values[1][:,46:48]=.98
    for clock in range(30):step(values,clock)
    c=counts(values)
    assert c['contact_phase_rows']==60 and c['front_stage_rows']==0
    assert c['left_axis_ready_rows']==c['right_axis_ready_rows']==60
    assert c['left_point_ready_rows']==c['right_point_ready_rows']==0
    assert c['measured_closure_ready_rows']==60
    assert c['both_returned_closed_rows']==c['lift_transition_rows']==0


def test_original_jaw_projector_is_distinct_from_geometric_readiness():
    values=setup()
    def gate(actor,goals):
        out=goals.clamp(-1,1).clone();out[:,20]=-1;return out
    values[0].agent.action_projector=gate
    for clock in range(30):step(values,clock)
    c=counts(values)
    assert c['both_geometry_ready_rows']==60
    assert c['right_requested_closed_rows']==c['right_projector_blocked_rows']==60
    assert c['left_projector_blocked_rows']==c['both_projected_closed_rows']==0
    assert c['left_returned_closed_rows']==60 and c['right_returned_closed_rows']==0
    assert c['lift_transition_rows']==0


def test_held_command_does_not_hide_unsettled_measured_jaws():
    values=setup();values[1][:,46:48]=.5
    for clock in range(30):step(values,clock)
    c=counts(values)
    assert c['both_returned_closed_rows']==60
    assert c['closed_hold_ready_rows']==14
    assert c['measured_closure_ready_rows']==c['measured_settled_hold_ready_rows']==0
    assert c['lift_transition_rows']==0


def test_actual_transition_is_counted_once_and_global_ids_are_preserved():
    values=setup();values[1][:,46:48]=.98
    for clock in range(30):step(values,clock,torch.tensor([True,False]))
    report=values[-1].gate_report();c=report['counts']
    assert c['lift_transition_rows']==values[-1].statistics['attempted_lift_episodes']==1
    assert c['selected_rows']==30 and c['lift_phase_rows']==6
    assert [r['environment'] for r in report['by_environment']]==[5]
    assert report['overlapping_counts_NOT_exclusive_failure_causes']
    assert report['closure_or_command_NOT_physical_pinch']


def test_diagnostic_history_is_not_a_control_input_and_does_not_advance_rng():
    a,b=setup(),setup();a[1][:,46:48]=b[1][:,46:48]=.98
    b[-1].gate_counts.fill_(12345)
    before=torch.get_rng_state().clone()
    for clock in range(30):
        xa,xb=step(a,clock),step(b,clock)
        assert torch.equal(xa[0],xb[0]) and torch.equal(xa[1][2],xb[1][2])
    assert a[-1].statistics==b[-1].statistics
    assert torch.equal(a[-1].phase,b[-1].phase)
    assert torch.equal(before,torch.get_rng_state())
    # Publishing a snapshot cannot mutate counters or cumulative restore state.
    saved=b[-1].gate_counts.clone();report=b[-1].gate_report()
    report['by_environment'][0]['counts']['selected_rows']=0
    assert torch.equal(saved,b[-1].gate_counts)
