"""Physical noise units, density/entropy and actual successful-servo gradients."""
from copy import deepcopy

import pytest
import torch

from test_rl_servo_critic import decoder_fixture
from test_rl_servo_success_retention import retained_agent, goals
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetentionGentleSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_servo_guard_sac import (
    ServoGuardCorrectionSAC, URDFServoGuardSACPilot, calibrated_body_noise_scales,
)


def guard(encoder):
    a = ServoGuardCorrectionSAC(518, 4, 21, SACConfig(hidden=16, entropy_backup=False,
        initial_policy_std=.02, min_policy_std=.0075, max_policy_std=.03),
        action_projector=AbsoluteGoalJawProjector())
    a.correction_radius = .3
    a.executed_body_anchor = lambda raw: raw.new_zeros(len(raw), 19)
    a.goal_servo_critic_encoder = encoder
    a.body_std_calibration[1:15] = .1
    return a


def test_noise_has_source_physical_scale_without_restricting_wide_mean_goals():
    source = torch.full((21,), .05)
    target = torch.full((21,), .05); target[1:15] = 2.
    factor = calibrated_body_noise_scales(source, target)
    assert torch.equal(factor[[0, 15, 16, 17, 18]], torch.ones(5))
    assert torch.allclose(target[1:15] * factor[1:15] * .02, source[1:15] * .02)
    assert (target[1:15] * .3 > source[1:15] * .3).all()
    assert torch.equal(calibrated_body_noise_scales([1] * 21, [1] * 21), torch.ones(19))


@pytest.mark.parametrize('fault', ['zero', 'nan', 'shape'])
def test_invalid_physical_noise_mapping_is_rejected(fault):
    source = torch.ones(21); target = torch.ones(21)
    if fault == 'zero': source[2] = 0
    elif fault == 'nan': target[1] = torch.nan
    else: target = target[:19]
    with pytest.raises(ValueError): calibrated_body_noise_scales(source, target)


def test_same_mean_and_jaws_but_calibrated_gaussian_density_and_entropy_target():
    raw, encoder = decoder_fixture(8)
    old, new = retained_agent(encoder), guard(encoder)
    new.actor.load_state_dict(old.actor.state_dict())
    old_ao = old.actor_normalizer(old.actor_features(raw))
    new_ao = new.actor_normalizer(new.actor_features(raw))
    om, os, oj = old.parameters_at(old_ao); nm, ns, nj = new.parameters_at(new_ao)
    assert torch.equal(om, nm) and torch.equal(oj, nj)
    assert torch.allclose(ns.exp(), os.exp() * new.body_std_calibration)
    assert torch.equal(old.act(raw, True), new.act(raw, True))
    ob, op, _ = old.continuous_sample(old_ao, deterministic=True, raw=raw)
    nb, np, _ = new.continuous_sample(new_ao, deterministic=True, raw=raw)
    assert torch.equal(ob, nb)
    log_scale = new.body_std_calibration.log().sum()
    assert torch.allclose(np - op, (-log_scale).expand_as(op), atol=1e-5)
    ot = old.continuous_entropy_target(old_ao, raw)
    nt = new.continuous_entropy_target(new_ao, raw)
    cap_adjustment = (.03 ** 2 - (.03 * new.body_std_calibration).square()).sum()
    assert torch.allclose(nt - ot, (log_scale + cap_adjustment).expand_as(ot), atol=1e-5)


def test_wrong_saturated_command_escape_and_real_update_retain_actual_labels():
    raw, encoder = decoder_fixture(8); a = guard(encoder)
    request = torch.zeros(8, 19); request[:, 1] = -.9; request.requires_grad_(True)
    labels = goals(8); labels[:, 1] = .2
    loss = a.success_body_loss(raw, request, labels)
    gradient = torch.autograd.grad(loss, request)[0]
    assert gradient[:, 1].lt(0).all() and torch.isfinite(gradient).all()
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=a.act(raw, True),
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8, 4),
        reward=torch.zeros(8), terminated=torch.zeros(8, dtype=torch.bool))
    before = deepcopy(batch); labels_before = labels.clone()
    report = a.update(batch, successful_train=dict(actor_obs=raw, action=labels),
        success_goal_weight=2., success_jaw_weight=.05)
    assert report['actor_updated'] and report['success_servo_interval_weight'] == .2
    assert report['success_servo_interval_coefficient'] == .1
    assert report['success_goal_loss'] == pytest.approx(
        report['success_absolute_goal_MSE'] + .1 * report['success_servo_interval_Huber'])
    assert torch.equal(labels, labels_before) and all(torch.equal(v, batch[k]) for k, v in before.items())
    restored = guard(encoder); restored.restore(a.checkpoint())
    assert torch.equal(restored.act(raw, True), a.act(raw, True))
    assert torch.equal(restored.body_std_calibration, a.body_std_calibration)
    with pytest.raises(ValueError): retained_agent(encoder).restore(a.checkpoint())


def test_guard_and_original_policy_dispatch_remain_separate():
    assert staged_policy_class(URDFServoGuardSACPilot.artifact_type) is URDFServoGuardSACPilot
    assert staged_policy_class(ServoRetentionGentleSACPilot.artifact_type) is ServoRetentionGentleSACPilot
