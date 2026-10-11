"""First rack warning is tied to the measured control packet, not terminal peak."""
from copy import deepcopy
import json

import pytest

from rack_contact_summary import RackContactSummary


def packet(step, force, phase='held_grasp', body='right_gripper'):
    return dict(steps=step, rack_peak_force_n=force, rack_peak_body=body,
        staged_base=dict(phase=phase), flap_distances=[.03, .06],
        pinching=[False, False], stable_hands=[False, False], opposing_flaps=False,
        hold_time_s=0., proof_lift=False, rack_clearance_m=.01, success=False, unsafe=False)


def test_first_approach_warning_held_warning_and_later_terminal_peak_are_distinct():
    t = RackContactSummary(1, 10.)
    t.observe(0, packet(1, .2, 'base_approach'))
    t.observe(0, packet(2, 2.5, 'base_approach', 'left_arm4'))
    t.observe(0, packet(3, 2., body='left_gripper'))
    terminal = packet(4, 30., body='right_arm4'); terminal['unsafe'] = True
    t.observe(0, terminal)
    r = t.snapshot(0)
    assert r['first_warning']['control_step'] == 2
    assert r['first_warning']['max_rack_force_body'] == 'left_arm4'
    assert r['first_held_warning']['control_step'] == 3
    assert r['largest_measured_force']['control_step'] == 4
    assert r['largest_measured_force']['unsafe']
    assert not r['first_warning']['unsafe']
    assert r['observed_control_steps'] == 4


def test_threshold_is_strict_and_diagnostic_only_and_empty_environment_stays_empty():
    t = RackContactSummary(2, 10.)
    t.observe(1, packet(1, 1.))
    t.observe(1, packet(2, 1.0001))
    assert t.snapshot(1)['first_warning']['control_step'] == 2
    assert t.snapshot(0)['observed_control_steps'] == 0
    assert t.snapshot(0)['first_warning'] is None
    assert t.contract['failure_force_n'] == 10.
    assert not t.contract['threshold_changes_success_safety_reward']


def test_pre_reset_measurement_is_copied_and_cannot_take_post_reset_hand_or_force_state():
    t = RackContactSummary(1, 10.); r = packet(1, 12.)
    before = deepcopy(r); t.observe(0, r)
    assert r == before
    r['rack_peak_force_n'] = 999.; r['pinching'][0] = True; r['flap_distances'][0] = 9.
    saved = t.snapshot(0)
    assert saved['first_warning']['max_rack_force_n'] == 12.
    assert saved['first_warning']['pinching_hands'] == [False, False]
    assert saved['first_warning']['flap_distances_m'] == [.03, .06]
    saved['first_warning']['pinching_hands'][1] = True
    assert t.snapshot(0)['first_warning']['pinching_hands'] == [False, False]


def test_duplicate_packets_and_fresh_wave_cannot_inherit_an_old_event():
    t = RackContactSummary(1, 10.)
    t.observe(0, packet(5, 4.)); t.observe(0, packet(5, 99.)); t.observe(0, packet(4, 99.))
    assert t.snapshot(0)['observed_control_steps'] == 1
    assert t.snapshot(0)['largest_measured_force']['max_rack_force_n'] == 4.
    fresh = RackContactSummary(1, 10.)
    fresh.observe(0, packet(1, .5))
    assert fresh.snapshot(0)['first_warning'] is None


def test_nonfinite_diagnostics_cannot_pollute_json_or_create_a_contact_event():
    t = RackContactSummary(1, 10.)
    t.observe(0, packet(1, float('nan'))); t.observe(0, packet(2, -1.))
    r = packet(3, 2.); r['flap_distances'][0] = float('inf'); t.observe(0, r)
    saved = t.snapshot(0)
    assert saved['invalid_force_samples'] == 2
    assert saved['first_warning']['control_step'] == 3
    assert saved['first_warning']['flap_distances_m'] == [None, .06]
    json.dumps(saved, allow_nan=False)


def test_warning_scales_with_existing_safety_threshold_without_overwriting_it():
    t = RackContactSummary(1, .1)
    assert t.contract['warning_force_n'] == pytest.approx(.01)
    assert t.contract['failure_force_n'] == .1
    with pytest.raises(ValueError): RackContactSummary(0, 10.)
    with pytest.raises(ValueError): RackContactSummary(1, float('nan'))
