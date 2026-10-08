"""Actual support/Jacobian consistency and bounded upright TRAIN proposals."""
from copy import deepcopy
import math

import pytest
import torch

from test_rl_servo_critic import decoder_fixture
from test_rl_urdf_servo_guard_sac import guard
from test_rl_perceived_contact_exploration import fixture
from kuavo_isaaclab_scene.rl.algorithms.common import gaussian_log_prob
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_upright_contact_sac import (
    UprightServoGuardSAC, URDFUprightContactSACPilot, upright_support_affine, ABSOLUTE_COLUMNS,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.perceived_contact_exploration import (
    PerceivedContactExploration, perceived_contact_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import URDFStrongSuccessSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import held_goal_coordinates
from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import planar_position


def agent_fixture(n=8):
    raw, encoder = decoder_fixture(n)
    encoder.center[17:19] = torch.tensor([.041708149, .647084117])
    encoder.scale[17:19] = torch.tensor([.033473842, .134806409])
    old = guard(encoder)
    a = UprightServoGuardSAC(518, 4, 21, old.config, action_projector=old.action_projector)
    a.correction_radius = .3
    a.executed_body_anchor = lambda raw: raw[:, :19].clamp(-1, 1)
    a.goal_servo_critic_encoder = encoder
    a.body_std_calibration.copy_(old.body_std_calibration)
    a.configure_upright_support(encoder.center, encoder.scale, encoder.coordinates.links)
    with torch.no_grad():
        a.actor.network[-1].weight[:19].zero_(); a.actor.network[-1].bias[:19].zero_()
    return raw, encoder, a


def test_full_torso_support_intersects_real_height_without_extending_physical_travel():
    raw, enc, a = agent_fixture()
    raw[:, 1:15] = torch.linspace(-.95, .95, 14)
    raw[:, 17:19] = 0.
    normalized = a.actor_normalizer(raw)
    body = a.continuous_sample(normalized, deterministic=True, raw=raw)[0]
    assert torch.allclose(body, a.executed_body_anchor(raw), atol=1e-7)
    bound = enc.coordinates.links[:, 1].sum() + .46
    assert (enc.center + enc.scale)[18] > bound + .009
    for sign in (-1., 1.):
        offset = torch.zeros(len(raw), 19); offset[:, ABSOLUTE_COLUMNS] = sign * 100
        full = a.continuous_sample(normalized, deterministic=True, body_latent_offset=offset, raw=raw)[0]
        xz = enc.center[17:19] + enc.scale[17:19] * full[:, 17:19]
        assert (xz[:, 1] <= bound + 1e-7).all()
        assert (full[:, 17] * sign > .999).all()
        if sign > 0: assert torch.allclose(xz[:, 1], bound.expand(len(raw)), atol=1e-7)
        assert torch.equal(full[:, [0, 15, 16]], body[:, [0, 15, 16]])


def test_torso_boundary_source_projection_has_finite_density_and_escape_gradient():
    raw, enc, a = agent_fixture(1); raw[:, 18] = 1.
    offset = torch.zeros(1, 19, requires_grad=True)
    body, logp, _ = a.continuous_sample(a.actor_normalizer(raw), deterministic=True,
        body_latent_offset=offset, raw=raw)
    gradient = torch.autograd.grad(body[:, 17:19].sum(), offset)[0]
    assert gradient[:, 17:19].gt(0).all() and torch.isfinite(logp).all()
    assert torch.isfinite(a.continuous_entropy_target(a.actor_normalizer(raw), raw)).all()
    assert body[0, 18] < (a.upright_support_anchor + a.upright_support_scale)[1]


def test_density_and_entropy_use_the_actual_torso_affine_Jacobian():
    raw, enc, a = agent_fixture(); raw[:, 17:19] = .4
    normalized = torch.zeros_like(raw)
    mean, std, _ = a.continuous_parameters(normalized, raw)
    body, logp, _ = a.continuous_sample(normalized, deterministic=True, raw=raw)
    jac = 2 * (math.log(2) - mean - torch.nn.functional.softplus(-2 * mean))
    affine, scale = a.anchor_and_scale(raw)
    expected = ((gaussian_log_prob(mean, mean, std.exp()) - jac - scale.clamp_min(1e-8).log())
                * (scale > 1e-8)).sum(-1)
    assert torch.allclose(expected, logp, atol=1e-5)
    assert torch.allclose(body, affine + scale * mean.tanh(), atol=1e-7)
    assert scale[0, 18] < 1 and scale[0, 17] == 1
    other = raw.clone(); other[:, 17:19] = -.4
    assert not torch.equal(mean[:, 17:19], a.continuous_parameters(normalized, other)[0][:, 17:19])
    with pytest.raises(ValueError): a.continuous_parameters(normalized)


def test_real_actor_Q_target_and_success_update_resume_share_absolute_torso_distribution():
    raw, enc, a = agent_fixture()
    calls = []; original = a.continuous_parameters
    def record(normalized, raw=None):
        assert raw is not None; calls.append(len(raw)); return original(normalized, raw)
    a.continuous_parameters = record
    action = a.act(raw, True)
    term = torch.tensor([True] * 4 + [False] * 4)
    next_raw = raw.clone(); next_raw[term] = 0
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=action,
        next_actor_obs=next_raw, next_critic_obs=torch.zeros(8, 4),
        reward=torch.zeros(8), terminated=term)
    before = deepcopy(batch)
    report = a.update(batch, successful_train=dict(actor_obs=raw, action=action.clone()),
                      success_goal_weight=2., success_jaw_weight=.05)
    assert report['actor_updated'] and report['success_servo_interval_coefficient'] == 1.
    assert 4 in calls and all(torch.equal(v, batch[k]) for k, v in before.items())
    restored = agent_fixture()[2]; restored.restore(a.checkpoint())
    assert torch.equal(restored.act(raw, True), a.act(raw, True))
    with pytest.raises(ValueError): guard(enc).restore(a.checkpoint())


def contact_fixture(*, ready=False):
    pilot, raw, result, supp, old = fixture(ready=ready)
    # This synthetic torso configuration is reachable. The assertion checks
    # physical command identity, not task feasibility or grasp success.
    pilot.center[17:19] = planar_position(raw[:1, :2], pilot.coordinates.links)[0]
    pilot.scale[17:19] = .1
    anchor, scale = upright_support_affine(pilot.center, pilot.scale, pilot.coordinates.links)
    pilot.agent.upright_support_anchor = anchor; pilot.agent.upright_support_scale = scale
    tracker = PerceivedContactExploration(128, raw, settled_close=True, precise_feedback=True,
        motion_feedback=True, upright_feedback=True)
    if ready: tracker.axes = old.axes.clone()
    return pilot, raw, result, supp, tracker


def contact_step(values, clock, chosen=None):
    pilot, raw, result, supp, tracker = values
    return tracker.step(pilot, raw, result, torch.tensor([5, 19]), supp,
        torch.ones(2, dtype=torch.bool) if chosen is None else chosen, clock)


def test_joint_torso_stage_proposal_is_physically_bounded_and_selected_only():
    values = contact_fixture(); pilot, raw, result, supp, tracker = values
    before = raw.clone(), supp.clone(), torch.get_rng_state()
    command, (_, critic, goals) = contact_step(values, 0, torch.tensor([True, False]))
    assert torch.equal(command[1], result[0][1]) and torch.isnan(critic).all()
    assert tracker.upright_control.initialized[5] and not tracker.upright_control.initialized[19]
    assert torch.equal(command[:, [0,1,2,3,22,23]], result[0][:, [0,1,2,3,22,23]])
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw, pilot.center + pilot.scale * goals, pilot.stage))
    torso_target = pilot.center[17:19] + pilot.scale[17:19] * goals[0,17:19]
    measured = planar_position(raw[:1,:2], pilot.coordinates.links)[0]
    assert (torso_target - measured).abs().max() <= .00101
    assert command.abs().max() <= 1 and goals.abs().max() <= 1
    assert all(torch.equal(x,y) for x,y in zip(before, (raw, supp, torch.get_rng_state())))


def test_closing_holds_measured_torso_instead_of_following_clock_source_then_resets():
    values = contact_fixture(ready=True); pilot, raw, result, _, tracker = values
    first = contact_step(values, 0)[1][2]
    changed = result[1][2].clone(); changed[:,17:19] = .9
    values = pilot, raw, (result[0], (result[1][0], result[1][1], changed)), values[3], tracker
    second = contact_step(values, 1)[1][2]
    assert tracker.phase[torch.tensor([5,19])].eq(1).all()
    assert torch.equal(first[:,17:19], second[:,17:19])
    tracker.upright_control.target[5] += .01
    reset = contact_step(values, 0)[1][2]
    assert torch.equal(reset[:,17:19], first[:,17:19])
    assert tracker.statistics['attempted_lift_episodes'] == 0


def test_projected_IK_outside_support_keeps_valid_goal_and_arm_solution_per_row(monkeypatch):
    values = contact_fixture(); pilot, raw, result, supp, tracker = values
    from kuavo_isaaclab_scene.rl.multi_box.experiments import upright_contact_control as module
    original = module.upright_joint_step
    def projected(current, target, pitch, links, limits):
        torso, achieved = original(current, target, pitch, links, limits)
        # A speed-limited recovery from measured physics outside the rectangular
        # goal support cannot jump straight back to an allowed target.
        if len(current) == 2:
            achieved = achieved.clone(); achieved[0, 0] = pilot.center[17] - .2
        return torso, achieved
    monkeypatch.setattr(module, 'upright_joint_step', projected)
    before = raw.clone(), supp.clone(), torch.get_rng_state()
    command, (_, _, goals) = contact_step(values, 0)
    control = tracker.upright_control
    assert control.projected_proposal_rejections == 1
    assert torch.equal(control.target[5], control.origin[5])
    assert not torch.equal(control.target[19], control.origin[19])
    assert torch.isfinite(command).all() and command.abs().max() <= 1
    assert torch.equal(command, held_goal_coordinates(pilot.coordinates, raw,
        pilot.center + pilot.scale * goals, pilot.stage))
    assert all(torch.equal(x, y) for x, y in zip(before, (raw, supp, torch.get_rng_state())))


def test_nonfinite_upright_IK_still_fails_instead_of_sending_invalid_command(monkeypatch):
    values = contact_fixture()
    from kuavo_isaaclab_scene.rl.multi_box.experiments import upright_contact_control as module
    original = module.upright_joint_step
    def invalid(current, target, pitch, links, limits):
        torso, achieved = original(current, target, pitch, links, limits)
        if len(current) == 2:
            achieved = achieved.clone(); achieved[0, 0] = float('nan')
        return torso, achieved
    monkeypatch.setattr(module, 'upright_joint_step', invalid)
    with pytest.raises(ValueError, match='finite fixed-pitch'):
        contact_step(values, 0)


def test_explicit_variant_contract_and_unassisted_eval_preserve_old_defaults(monkeypatch):
    with pytest.raises(ValueError): perceived_contact_contract(upright_feedback=True)
    old = perceived_contact_contract(settled_close=True, precise_feedback=True, motion_feedback=True)
    new = perceived_contact_contract(settled_close=True, precise_feedback=True, motion_feedback=True, upright_feedback=True)
    assert 'torso_goal' not in old and not new['curriculum']
    for key in ('close_point_tolerance_m', 'closed_reacquire_maximum_point_distance_m',
                'measured_minimum_closure_fraction', 'joint_increment_cap_rad'):
        assert old[key] == new[key]
    assert staged_policy_class(URDFUprightContactSACPilot.artifact_type) is URDFUprightContactSACPilot
    marker = object()
    monkeypatch.setattr(URDFStrongSuccessSACPilot, 'act', lambda *a, **k: marker)
    pilot = object.__new__(URDFUprightContactSACPilot); pilot.training = False
    assert pilot.act(None, None, 0) is marker
