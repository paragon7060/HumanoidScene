"""A separate objective preserves actions/Q while strengthening true success retention."""
from copy import deepcopy

import pytest
import torch

from test_rl_servo_critic import decoder_fixture
from test_rl_servo_success_retention import goals
from test_rl_urdf_full_arm_sac import full
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import BoundedCorrectionHybridSAC
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_strong_success_sac import (
    StrongSuccessFullArmSAC, URDFStrongSuccessSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_full_arm_sac import URDFFullArmSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_servo_guard_sac import servo_guard_contract


def paired(encoder):
    old = full(encoder)
    new = StrongSuccessFullArmSAC(518, 4, 21, old.config,
        action_projector=old.action_projector)
    new.load_state_dict(old.state_dict())
    new.correction_radius = old.correction_radius
    new.executed_body_anchor = old.executed_body_anchor
    new.goal_servo_critic_encoder = encoder
    return old, new


def test_same_greedy_noise_density_entropy_and_Q_targets_before_updates():
    raw, encoder = decoder_fixture(8); old, new = paired(encoder)
    raw[:, 1:15] = torch.linspace(-.95, .95, 14)
    assert torch.equal(old.act(raw, True), new.act(raw, True))
    generator = torch.Generator().manual_seed(36)
    noise = torch.randn(8, 19, generator=generator)
    for agent in (old, new):
        normalized = agent.actor_normalizer(agent.actor_features(raw))
        sample = agent.continuous_sample(normalized, noise=noise, raw=raw)
        target_entropy = agent.continuous_entropy_target(normalized, raw)
        if agent is old: reference, entropy = sample, target_entropy
        else:
            assert all(torch.equal(a,b) for a,b in zip(reference,sample))
            assert torch.equal(entropy,target_entropy)
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8,4), action=old.act(raw,True),
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8,4),
        reward=torch.arange(8).float(), terminated=torch.tensor([True]*4+[False]*4))
    batch['next_actor_obs'][:4]=0
    torch.manual_seed(54); target_old=old.critic_target(batch,old.log_alpha.exp(),old.log_alpha_discrete.exp())[0]
    torch.manual_seed(54); target_new=new.critic_target(batch,new.log_alpha.exp(),new.log_alpha_discrete.exp())[0]
    assert torch.equal(target_old,target_new)


def test_only_servo_interval_component_changes_and_wrong_command_can_escape():
    raw,encoder=decoder_fixture(8);old,new=paired(encoder)
    request=torch.zeros(8,19);request[:,1]=-.9;request.requires_grad_(True)
    labels=goals(8);labels[:,1]=.2;labels_before=labels.clone()
    original=BoundedCorrectionHybridSAC.success_body_loss(old,raw,request,labels)
    old_loss=old.success_body_loss(raw,request,labels)
    new_loss=new.success_body_loss(raw,request,labels)
    assert float(new_loss-original)==pytest.approx(float((old_loss-original)*10),rel=1e-5)
    gradient=torch.autograd.grad(new_loss,request)[0]
    assert gradient[:,1].lt(0).all() and torch.isfinite(gradient).all()
    assert torch.equal(labels,labels_before)
    assert new.hybrid_contract['URDF_servo_guard']==servo_guard_contract(1.)
    assert old.hybrid_contract['URDF_servo_guard']==servo_guard_contract()


def test_actual_update_reports_coefficient_and_resume_rejects_other_objective():
    raw,encoder=decoder_fixture(8);old,new=paired(encoder)
    labels=goals(8);labels[:,1]=.2
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,4),action=new.act(raw,True),
        next_actor_obs=raw.clone(),next_critic_obs=torch.zeros(8,4),
        reward=torch.zeros(8),terminated=torch.zeros(8,dtype=torch.bool))
    before=deepcopy(batch)
    report=new.update(batch,successful_train=dict(actor_obs=raw,action=labels),
        success_goal_weight=2.,success_jaw_weight=.05)
    assert report['actor_updated'] and report['success_servo_interval_coefficient']==1.
    assert report['success_servo_interval_weight']==2.
    assert report['success_goal_loss']==pytest.approx(
        report['success_absolute_goal_MSE']+report['success_servo_interval_Huber'])
    assert all(torch.equal(v,batch[k]) for k,v in before.items())
    _,restored=paired(encoder);restored.restore(new.checkpoint())
    assert torch.equal(restored.act(raw,True),new.act(raw,True))
    with pytest.raises(ValueError):old.restore(new.checkpoint())
    with pytest.raises(ValueError):restored.restore(old.checkpoint())
    assert staged_policy_class(URDFStrongSuccessSACPilot.artifact_type) is URDFStrongSuccessSACPilot
    assert staged_policy_class(URDFFullArmSACPilot.artifact_type) is URDFFullArmSACPilot


@pytest.mark.parametrize('invalid',[True,.3,0.,float('nan')])
def test_unknown_or_bool_coefficient_contracts_rejected(invalid):
    with pytest.raises(ValueError):servo_guard_contract(invalid)
