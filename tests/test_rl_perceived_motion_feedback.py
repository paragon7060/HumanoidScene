"""Bounded closure tracking without changing the actual grasp contract."""
import pytest
import torch

from test_rl_perceived_contact_exploration import fixture
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_perceived_contact_sac import (
    URDFMotionFeedbackSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def setup(*, motion=True, distance=.004):
    pilot, raw, result, supp, old = fixture(ready=True)
    supp[:, :36].reshape(2, 2, 2, 9)[..., :3] = torch.tensor([distance, 0, 0])
    tracker = PerceivedContactExploration(128, raw, settled_close=True,
        precise_feedback=True, motion_feedback=motion)
    tracker.axes = old.axes.clone()
    return pilot, raw, result, supp, tracker


def step(values, clock, chosen=None, ids=None):
    pilot, raw, result, supp, tracker = values
    return tracker.step(pilot, raw, result,
        torch.tensor([5, 19]) if ids is None else ids, supp,
        torch.ones(2, dtype=torch.bool) if chosen is None else chosen, clock)


def test_unchanged_stationary_panel_reduces_position_error_faster_only_while_closed():
    v3, v4 = setup(motion=False), setup()
    old_goals, new_goals = step(v3, 0)[1][2], step(v4, 0)[1][2]
    assert new_goals[:, 19:].eq(1).all()
    assert not torch.equal(new_goals[:, 1:15], old_goals[:, 1:15])
    # Open approach retains the previous action exactly; it does not speed
    # entry into rack obstacles to compensate for a closure-only defect.
    v3, v4 = setup(motion=False, distance=.02), setup(distance=.02)
    assert torch.equal(step(v3, 0)[0], step(v4, 0)[0])


def test_current_panel_motion_is_used_only_for_consecutive_ticks():
    a, b = setup(distance=.001), setup(distance=.001)
    step(a, 0); step(b, 0)
    for values in (a, b):
        values[3][:, :36].reshape(2, 2, 2, 9)[..., 0] += .003
    continuous = step(a, 1)[1][2]
    skipped = step(b, 2)[1][2]
    assert not torch.equal(continuous[:, 1:15], skipped[:, 1:15])
    assert continuous[:, 19:].eq(1).all() and skipped[:, 19:].eq(1).all()


def test_clock_reset_does_not_reuse_previous_episode_motion():
    a, fresh = setup(distance=.001), setup(distance=.004)
    step(a, 4)
    a[3][:, :36].reshape(2, 2, 2, 9)[..., 0] += .003
    assert torch.equal(step(a, 0)[0], step(fresh, 0)[0])


def test_no_close_or_lift_without_original_precise_geometry_and_measured_settling():
    values = setup(distance=.013)
    for clock in range(30):
        assert step(values, clock)[1][2][:, 19:].eq(-1).all()
    values = setup(); raw, tracker = values[1], values[-1]
    raw[:, 46:48] = .5
    for clock in range(40):
        step(values, clock)
        assert tracker.statistics['attempted_lift_episodes'] == 0
    raw[:, 46:48] = .98
    for clock in range(40, 46):
        step(values, clock)
        assert tracker.statistics['attempted_lift_episodes'] == 0
    step(values, 46)
    assert tracker.statistics['attempted_lift_episodes'] == 2


def test_lost_panel_still_reopens_and_production_jaw_gate_still_wins():
    values = setup(); step(values, 0)
    values[3][:, :36].reshape(2, 2, 2, 9)[..., 0] += .02
    assert step(values, 1)[1][2][:, 19:].eq(-1).all()
    assert not values[-1].closing.any()
    values = setup()
    def gate(actor, goals):
        out = goals.clamp(-1, 1).clone(); out[:, 20] = -1; return out
    values[0].agent.action_projector = gate
    for clock in range(30):
        assert step(values, clock)[1][2][:, 20].eq(-1).all()
    assert values[-1].statistics['attempted_lift_episodes'] == 0


def test_deployable_inputs_global_ids_untouched_channels_and_actual_commands():
    values = setup(); pilot, raw, result, supp, tracker = values
    before = (raw.clone(), supp.clone(), torch.get_rng_state())
    command, (actor, critic, goals) = step(values, 0, torch.tensor([True, False]))
    assert torch.equal(command[1], result[0][1])
    assert tracker.previous_target_clock[5] == 0 and tracker.previous_target_clock[19] == -1
    assert tracker.previous_target_clock[0] == -1
    assert torch.equal(command[:, [0, 1, 2, 3, 18, 19, 22, 23]], result[0][:, [0, 1, 2, 3, 18, 19, 22, 23]])
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw,
        pilot.center + pilot.scale * goals, pilot.stage))
    assert command.abs().max() <= 1 and goals.abs().max() <= 1
    assert torch.isnan(critic).all()
    assert all(torch.equal(x, y) for x, y in zip(before, (raw, supp, torch.get_rng_state())))


def test_opt_in_saved_contract_and_unassisted_evaluation(monkeypatch):
    old = perceived_contact_contract(settled_close=True, precise_feedback=True)
    new = perceived_contact_contract(settled_close=True, precise_feedback=True, motion_feedback=True)
    assert 'closed_position_feedback_gain_per_s' not in old
    for key in ('close_point_tolerance_m', 'closed_reacquire_maximum_point_distance_m',
                'measured_minimum_closure_fraction', 'consecutive_measured_settled_ticks_before_attempt_lift',
                'joint_increment_cap_rad', 'pending_target_lead_cap_rad'):
        assert old[key] == new[key]
    assert not new['privileged_pinch_contact_reward_or_success_inputs']
    with pytest.raises(ValueError): perceived_contact_contract(motion_feedback=True)
    assert staged_policy_class(URDFMotionFeedbackSACPilot.artifact_type) is URDFMotionFeedbackSACPilot
    marker = object()
    monkeypatch.setattr(URDFStrongSuccessSACPilot, 'act', lambda *args, **kwargs: marker)
    pilot = object.__new__(URDFMotionFeedbackSACPilot); pilot.training = False
    assert pilot.act(None, None, 0) is marker
