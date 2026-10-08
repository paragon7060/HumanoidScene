"""Interior TRAIN grasp targets and unchanged upright SAC/evaluation semantics."""
import pytest
import torch

from test_rl_upright_contact_sac import contact_fixture, contact_step
from kuavo_isaaclab_scene.rl.multi_box.experiments.cartesian_flap_probe import contact_region_offsets
from kuavo_isaaclab_scene.rl.multi_box.experiments.kinematic_exploration import target_token
from kuavo_isaaclab_scene.rl.multi_box.geometry.grasp import nominal_flap_geometry
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import (
    URDFInteriorContactSACPilot, URDFUprightContactSACPilot, UprightServoGuardSAC,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates


def interior_fixture(*, ready=False):
    pilot, raw, result, supp, previous = contact_fixture(ready=ready)
    tracker = PerceivedContactExploration(128, raw, settled_close=True, precise_feedback=True,
        motion_feedback=True, upright_feedback=True, interior_contact=True)
    if ready: tracker.axes = previous.axes.clone()
    return pilot, raw, result, supp, tracker


def test_interior_contract_keeps_all_existing_gates_and_upright_distribution():
    flags = dict(settled_close=True, precise_feedback=True, motion_feedback=True, upright_feedback=True)
    old = perceived_contact_contract(**flags)
    assert old == perceived_contact_contract(**flags, interior_contact=False)
    new = perceived_contact_contract(**flags, interior_contact=True)
    changed = {'name', 'contact_point'}
    assert all(new[k] == v for k, v in old.items() if k not in changed)
    assert new['contact_tangent_margin_m'] == .020 and 'contact_tangent_margin_m' not in old
    assert not new['privileged_pinch_contact_reward_or_success_inputs'] and not new['curriculum']
    assert URDFInteriorContactSACPilot.agent_class is URDFUprightContactSACPilot.agent_class is UprightServoGuardSAC
    assert staged_policy_class(URDFInteriorContactSACPilot.artifact_type) is URDFInteriorContactSACPilot
    with pytest.raises(ValueError): perceived_contact_contract(interior_contact=True)
    with pytest.raises(ValueError): perceived_contact_contract(**flags, interior_contact=1)


def test_near_corner_target_has_room_for_close_tolerance_without_forcing_midpoint():
    rel = torch.zeros(1, 2, 9); rel[..., 3:] = torch.tensor([1., 0, 0, 0, 1, 0])
    rel[..., :3] = torch.tensor([.001, -.099, -.049])
    half = torch.tensor([.002, .10, .05]).expand(1, 2, 3)
    old = contact_region_offsets(rel, half)
    new = contact_region_offsets(rel, half, margin=.020)
    assert torch.equal(new[..., 0], torch.zeros(1, 2))
    assert torch.allclose(new[..., 1:], torch.tensor([.08, .03]).expand(1, 2, 2))
    # This is TCP geometry, not evidence of actual pad contact or success.
    assert ((half[..., 1:] - (new[..., 1:].abs() + .006)) >= .013999).all()
    assert ((half[..., 1:] - (old[..., 1:].abs() + .006)) < 0).all()
    rel[..., 1:3] = torch.tensor([-.025, .013])
    center_near = contact_region_offsets(rel, half, margin=.020)
    assert torch.allclose(center_near[..., 1:], torch.tensor([.025, -.013]).expand(1, 2, 2))


def test_actual_interior_proposal_decodes_exactly_and_preserves_selection_inputs_and_RNG():
    values = interior_fixture(); pilot, raw, result, supp, tracker = values
    relation = supp[:, :36].reshape(2, 2, 2, 9)
    relation[..., :3] = torch.tensor([.001, -.099, -.049])
    relation[..., 3:] = torch.tensor([1., 0, 0, 0, 1, 0])
    before = raw.clone(), supp.clone(), torch.get_rng_state(), result[0].clone(), result[1][2].clone()
    command, (actor, critic, goals) = contact_step(values, 0, torch.tensor([True, False]))
    assert torch.equal(command[1], result[0][1]) and torch.equal(goals[1], result[1][2][1])
    assert actor is result[1][0] and critic is result[1][1] and torch.isnan(critic).all()
    assert torch.equal(command[:, [0,1,2,3,22,23]], result[0][:, [0,1,2,3,22,23]])
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * goals, pilot.stage))
    token, valid = target_token(raw); assert valid.all()
    _, half, _ = nominal_flap_geometry(token[:, 5:8], token[:, 3:5].argmax(-1))
    margin = half[0, :, 1:] - tracker.contact_offsets[5, :, 1:].abs()
    assert (margin >= .019999).all() and tracker.phase[19] == -1
    assert all(torch.equal(x, y) for x, y in zip(before,
        (raw, supp, torch.get_rng_state(), result[0], result[1][2])))


def test_captured_interior_point_stays_panel_fixed_until_episode_reset():
    values = interior_fixture(); pilot, raw, result, supp, tracker = values
    contact_step(values, 0); original = tracker.contact_offsets[5].clone()
    supp[0, :36].reshape(2, 2, 9)[..., 1] += .01
    contact_step(values, 1)
    assert torch.equal(original, tracker.contact_offsets[5])
    tracker.contact_offsets[5] += 1
    contact_step(values, 0)
    token, _ = target_token(raw); _, half, _ = nominal_flap_geometry(token[:, 5:8], token[:, 3:5].argmax(-1))
    assert (half[0, :, 1:] - tracker.contact_offsets[5, :, 1:].abs() >= .019999).all()


def test_interior_closing_holds_torso_but_still_requires_measured_settling():
    values = interior_fixture(ready=True); pilot, raw, result, supp, tracker = values
    first = contact_step(values, 0)[1][2]
    changed = result[1][2].clone(); changed[:, 17:19] = .9
    values = pilot, raw, (result[0], (result[1][0], result[1][1], changed)), supp, tracker
    for clock in range(1, 25):
        goals = contact_step(values, clock)[1][2]
        assert torch.equal(goals[:, 17:19], first[:, 17:19])
    assert tracker.statistics['attempted_lift_episodes'] == 0
    assert tracker.phase[torch.tensor([5, 19])].eq(1).all()


def test_interior_evaluation_never_uses_the_TRAIN_helper(monkeypatch):
    marker = object()
    monkeypatch.setattr(URDFStrongSuccessSACPilot, 'act', lambda *a, **k: marker)
    pilot = object.__new__(URDFInteriorContactSACPilot); pilot.training = False
    assert pilot.act(None, None, 0) is marker
