"""Successful command equivalence, escape gradients and real trainer wiring."""
from copy import deepcopy
import pytest
import torch

from test_rl_servo_critic import decoder_fixture,make_agent
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_actions import servo_raw_actor
from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import planar_position
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import (
    servo_interval_loss,ServoRetainedCorrectionSAC,ServoRetentionGentleSACPilot,
)


def goals(n=1):
    value=torch.zeros(n,21);value[:,19:]=-1
    return value


def test_unclipped_deltas_share_real_pending_target_joint_and_upright_torso_units():
    raw,encoder=decoder_fixture(8)
    torch.manual_seed(13);g=goals(8);g[:,:19]=torch.rand(8,19)*1.8-.9
    measured=servo_raw_actor(torch.cat((raw[:,:474],raw[:,-6:]),-1))
    targets=measured[:,:20]+measured[:,416:436];physical=encoder.center+encoder.scale*g
    expected=torch.cat(((physical[:,:17]-targets[:,encoder.coordinates.joints.joint_columns])/
        raw.new_tensor(encoder.coordinates.joints.scales),
        (physical[:,17:19]-planar_position(targets[:,:2],encoder.coordinates.links))/(.1/30)),-1)
    assert torch.equal(encoder.unclipped_body(raw,g),expected)
    assert torch.equal(encoder(raw,g),torch.cat((expected,g[:,19:]),-1).clamp(-1,1))


@pytest.mark.parametrize('sign',[-1.,1.])
def test_equivalent_saturated_command_has_zero_interval_loss_without_inventing_inverse_goal(sign):
    raw,encoder=decoder_fixture(1);label=goals();predicted=goals()
    label[:,1]=sign*.3;predicted[:,1]=sign*.9
    assert not torch.equal(label,predicted) and torch.equal(encoder(raw,label),encoder(raw,predicted))
    assert servo_interval_loss(encoder,raw,predicted,label)==0


@pytest.mark.parametrize('sign',[-1.,1.])
def test_wrong_saturated_prediction_has_escape_gradient_while_clipped_MSE_does_not(sign):
    raw,encoder=decoder_fixture(1);label=goals();predicted=goals()
    label[:,1]=sign*.3;predicted[:,1]=-sign*.9;predicted.requires_grad_(True)
    clipped=(encoder(raw,predicted)[:,:19]-encoder(raw,label)[:,:19]).square().mean()
    assert torch.autograd.grad(clipped,predicted)[0][0,1]==0
    loss=servo_interval_loss(encoder,raw,predicted,label)
    gradient=torch.autograd.grad(loss,predicted)[0]
    assert loss>0 and gradient[0,1]*sign<0 and torch.isfinite(gradient).all()
    assert gradient[:,19:].eq(0).all()


def test_unsaturated_label_has_unique_drive_delta_and_recorded_label_is_detached():
    raw,encoder=decoder_fixture(1);label=goals();predicted=goals()
    label[:,1]=.1;predicted[:,1]=-.05
    label.requires_grad_(True);predicted.requires_grad_(True)
    loss=servo_interval_loss(encoder,raw,predicted,label)
    pg,lg=torch.autograd.grad(loss,[predicted,label],allow_unused=True)
    assert loss>0 and pg[0,1]<0 and lg is None
    assert servo_interval_loss(encoder,raw,label.detach(),label.detach())==0


def retained_agent(encoder):
    a=ServoRetainedCorrectionSAC(518,4,21,SACConfig(hidden=16,entropy_backup=False,
        initial_policy_std=.02,min_policy_std=.0075,max_policy_std=.03),
        action_projector=AbsoluteGoalJawProjector())
    a.correction_radius=.3;a.executed_body_anchor=lambda raw:raw.new_zeros(len(raw),19)
    a.goal_servo_critic_encoder=encoder
    return a


def test_only_opted_in_agent_adds_servo_loss_and_resume_rejects_legacy_objective():
    raw,encoder=decoder_fixture(8);old=make_agent(encoder);new=retained_agent(encoder)
    request=torch.zeros(8,19);request[:,1]=-.9
    label=goals(8);label[:,1]=.2
    legacy=old.success_body_loss(raw,request,label)
    assert legacy==pytest.approx(float((request*.3-label[:,:19]).square().mean()))
    assert new.success_body_loss(raw,request,label)>legacy
    state=new.checkpoint();restored=retained_agent(encoder);restored.restore(state)
    assert torch.equal(new.act(raw,True),restored.act(raw,True))
    with pytest.raises(ValueError):old.restore(state)
    with pytest.raises(ValueError):new.restore(old.checkpoint())


def test_production_actor_update_uses_new_loss_and_preserves_real_replay():
    raw,encoder=decoder_fixture(8);a=retained_agent(encoder)
    labels=goals(8);labels[:,1]=.2
    term=torch.zeros(8,dtype=torch.bool)
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,4),action=a.act(raw,True),
        next_actor_obs=raw.clone(),next_critic_obs=torch.zeros(8,4),reward=torch.zeros(8),terminated=term)
    before=deepcopy(batch);model=deepcopy(a.actor.state_dict())
    report=a.update(batch,successful_train=dict(actor_obs=raw,action=labels),
        success_goal_weight=1.,success_jaw_weight=.05)
    assert report['actor_updated'] and report['success_goal_loss']>0
    assert report['success_servo_interval_weight']==.01
    assert report['success_goal_loss']==pytest.approx(
        report['success_absolute_goal_MSE']+.01*report['success_servo_interval_Huber'])
    assert all(torch.equal(v,batch[k]) for k,v in before.items())
    assert any(not torch.equal(v,a.actor.state_dict()[k]) for k,v in model.items())


def test_fresh_initializer_preserves_models_and_rejects_trained_Q():
    from test_rl_body_policy_spread import initializer_inputs
    from prepare_gentle_servo_actor import prepare as gentle_prepare
    from prepare_servo_retention_actor import prepare
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
    state,replay=initializer_inputs()
    quarter,experience,_=gentle_prepare(state,replay)
    quarter['hybrid_contract']={'test_fixture_only':True}
    quarter_before=deepcopy(quarter)
    new,new_experience=prepare(quarter,experience)
    assert staged_policy_class(new['artifact_type']) is ServoRetentionGentleSACPilot
    assert all(torch.equal(v,new['model'][k]) for k,v in quarter_before['model'].items())
    assert quarter['goal_contract']==quarter_before['goal_contract']
    assert new_experience['goal_contract']==new['goal_contract']
    assert new['hybrid_contract']['success_body_retention']==new['goal_contract']['success_body_retention']
    contaminated=deepcopy(quarter);contaminated['critic_updates']=1
    with pytest.raises(ValueError,match='Trained'):prepare(contaminated,experience)
