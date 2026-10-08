"""Full support, actual-state probability and source-preserving local controls."""
from copy import deepcopy
import math

import pytest
import torch

from test_rl_servo_critic import decoder_fixture
from test_rl_urdf_servo_guard_sac import guard
from kuavo_isaaclab_scene.rl.algorithms.common import gaussian_log_prob
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_full_arm_sac import (
    FullArmServoGuardSAC, URDFFullArmSACPilot, ANCHOR_MARGIN,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def full(encoder):
    old = guard(encoder)
    result = FullArmServoGuardSAC(518, 4, 21, old.config,
        action_projector=old.action_projector)
    result.load_state_dict(old.state_dict())
    result.correction_radius = .3
    result.executed_body_anchor = lambda raw: raw[:, :19].clamp(-1, 1)
    result.goal_servo_critic_encoder = encoder
    with torch.no_grad():
        result.actor.network[-1].weight[:19].zero_()
        result.actor.network[-1].bias[:19].zero_()
    return result


def test_initial_source_and_jaws_preserved_but_full_arm_targets_available():
    raw, encoder = decoder_fixture(8); a = full(encoder)
    raw[:, 1:15] = torch.linspace(-.95, .95, 14)
    norm = a.actor_normalizer(a.actor_features(raw))
    body, _, logits = a.continuous_sample(norm, deterministic=True, raw=raw)
    assert torch.allclose(body, a.executed_body_anchor(raw), atol=1e-7)
    assert torch.equal(logits, a.parameters_at(norm)[2])
    for sign in [-1., 1.]:
        offset = torch.zeros(8, 19); offset[:, 1:15] = sign * 80
        candidate, _, _ = a.continuous_sample(norm, deterministic=True,
            body_latent_offset=offset, raw=raw)
        assert (candidate[:, 1:15] * sign > .999).all()
        assert torch.equal(candidate[:, [0,15,16,17,18]], body[:, [0,15,16,17,18]])


def test_physical_local_derivative_matches_old_guard_without_restricting_support():
    raw, encoder = decoder_fixture(8); a = full(encoder)
    raw[:, 1:15] = torch.linspace(-.95, .95, 14)
    normalized = a.actor_normalizer(a.actor_features(raw))
    offset = torch.zeros(8, 19, requires_grad=True)
    body = a.continuous_sample(normalized, deterministic=True,
        body_latent_offset=offset, raw=raw)[0]
    gradient = torch.autograd.grad(body.sum(), offset)[0]
    expected = (1 - a.executed_body_anchor(raw).abs()).clamp(max=.3)
    assert torch.allclose(gradient, expected, atol=2e-6)


@pytest.mark.parametrize('anchor', [-1., -.999, 0., .999, 1.])
def test_source_boundaries_have_finite_density_and_escape_gradient(anchor):
    raw, encoder = decoder_fixture(1); raw[:,1:15] = anchor; a = full(encoder)
    normalized = a.actor_normalizer(a.actor_features(raw))
    offset = torch.zeros(1,19,requires_grad=True)
    body, logp, _ = a.continuous_sample(normalized, deterministic=True,
        body_latent_offset=offset, raw=raw)
    gradient = torch.autograd.grad(body[:,1:15].sum(),offset)[0][:,1:15]
    assert (gradient>0).all() and torch.isfinite(logp).all()
    assert torch.isfinite(a.continuous_entropy_target(normalized,raw)).all()
    assert (body[:,1:15]-anchor).abs().max() <= 2 * ANCHOR_MARGIN


def test_density_uses_actual_raw_source_and_only_non_arm_affine_jacobian():
    raw, encoder = decoder_fixture(8); a = full(encoder)
    raw[:,1:15] = .7
    # Supply identical normalized features but different measured anchor values;
    # reconstructing raw from a clipped normalizer would produce identical means.
    normalized = torch.zeros_like(raw)
    mean, logstd, _ = a.continuous_parameters(normalized,raw)
    other = raw.clone(); other[:,1:15] = -.7
    assert not torch.equal(mean,a.continuous_parameters(normalized,other)[0])
    body, logp, _ = a.continuous_sample(normalized, deterministic=True,raw=raw)
    correction = 2*(math.log(2)-mean-torch.nn.functional.softplus(-2*mean))
    _, scale = a.anchor_and_scale(raw)
    expected = ((gaussian_log_prob(mean,mean,logstd.exp())-correction-
                 scale.clamp_min(1e-8).log())*(scale>1e-8)).sum(-1)
    assert torch.allclose(logp,expected,atol=1e-5)
    assert torch.allclose(body[:,1:15],raw[:,1:15],atol=1e-7)
    with pytest.raises(ValueError,match='raw'):a.continuous_parameters(normalized)


def test_actual_actor_target_and_success_update_use_same_full_goals_and_resume():
    raw, encoder = decoder_fixture(8); a = full(encoder)
    calls=[]
    original=a.continuous_parameters
    def record(normalized,raw=None):
        assert raw is not None
        calls.append(raw.clone())
        return original(normalized,raw)
    a.continuous_parameters=record
    action=a.act(raw,True)
    term=torch.tensor([True]*4+[False]*4)
    next_raw=raw.clone();next_raw[term]=0
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,4),action=action,
        next_actor_obs=next_raw,next_critic_obs=torch.zeros(8,4),
        reward=torch.zeros(8),terminated=term)
    before=deepcopy(batch);labels=action.clone()
    report=a.update(batch,successful_train=dict(actor_obs=raw,action=labels),
        success_goal_weight=2.,success_jaw_weight=.05)
    assert report['actor_updated'] and report['success_servo_interval_weight']==.2
    assert any(len(c)==4 for c in calls) and all(c[:,-6].eq(1).all() for c in calls)
    assert all(torch.equal(v,batch[k]) for k,v in before.items())
    restored=full(encoder);restored.restore(a.checkpoint())
    assert torch.equal(restored.act(raw,True),a.act(raw,True))
    with pytest.raises(ValueError):guard(encoder).restore(a.checkpoint())
    assert staged_policy_class(URDFFullArmSACPilot.artifact_type) is URDFFullArmSACPilot
