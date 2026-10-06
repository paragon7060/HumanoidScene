"""Real decoder invariants; synthetic fixtures are not grasp success data."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import (
    AbsoluteGoalJawProjector,BoundedCorrectionHybridSAC,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_actions import servo_raw_actor,goal_body_command
from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import planar_position
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import BodyServoCriticEncoder,servo_critic_contract


def decoder_fixture(n=8):
    coordinates=PoseGoalCoordinates(exact_projected_base=True)
    raw=torch.zeros(n,518)
    raw[:,0]=.3;raw[:,1]=-.2;raw[:,2]=-.1
    raw[:,144]=1;raw[:,173]=1;raw[:,-6]=1;raw[:,-1]=.3
    # Include pending drive target errors; measured positions alone are wrong.
    raw[:,154]=.013
    nominal=torch.cat((raw[:,:474],raw[:,-6:]),-1)
    measured=servo_raw_actor(nominal)
    targets=measured[:,:20]+measured[:,416:436]
    center=torch.cat((targets[0,coordinates.joints.joint_columns],
        planar_position(targets[:1,:2],coordinates.links)[0],torch.zeros(2)))
    scale=torch.full((21,),.1);scale[19:]=1
    return raw,BodyServoCriticEncoder(coordinates,center,scale)


def test_encoder_matches_existing_decoder_with_real_pending_targets_and_binary_jaws():
    raw,encoder=decoder_fixture()
    torch.manual_seed(14)
    goals=torch.rand(8,21)*1.8-.9;goals[:,19:]=torch.tensor([1.,-1.])
    nominal=torch.cat((raw[:,:474],raw[:,-6:]),-1)
    expected=goal_body_command(encoder.coordinates,nominal,encoder.center+encoder.scale*goals)
    assert torch.equal(encoder(raw,goals),expected)
    assert (expected.abs()<=1).all() and (expected[:,19:].abs()==1).all()
    assert encoder.contract==servo_critic_contract()


def test_same_clipped_command_has_identical_critic_input_and_exact_zero_gradient():
    raw,encoder=decoder_fixture(1)
    a=torch.zeros(1,21);a[:,19:]=1;a[:,1]=.5
    b=a.clone();b[:,1]=.9
    assert not torch.equal(a,b) and torch.equal(encoder(raw,a),encoder(raw,b))
    a.requires_grad_(True);encoded=encoder(raw,a)
    derivative=torch.autograd.grad(encoded[0,1],a)[0]
    assert derivative.eq(0).all()  # Both lie beyond the actual arm step limit.
    a=a.detach();a[:,1]=.05;a.requires_grad_(True)
    derivative=torch.autograd.grad(encoder(raw,a)[0,1],a)[0]
    assert derivative[0,1]==pytest.approx(.1/.02)
    assert torch.count_nonzero(derivative)==1


@pytest.mark.parametrize('fault',['approach','invalid_telemetry','nonfinite','wrong_width','missing'])
def test_encoder_rejects_nonphysical_state_or_terminal_placeholder(fault):
    raw,encoder=decoder_fixture(1);goals=torch.zeros(1,21);goals[:,19:]=-1
    if fault=='approach':raw[:,-6]=0
    elif fault=='invalid_telemetry':raw[:,173]=0
    elif fault=='nonfinite':raw[:,0]=torch.nan
    elif fault=='wrong_width':raw=raw[:,:517]
    else:raw=None
    with pytest.raises(ValueError):encoder(raw,goals)


def make_agent(encoder=None):
    agent=BoundedCorrectionHybridSAC(518,4,21,SACConfig(hidden=16,entropy_backup=False,
        min_policy_std=.03,max_policy_std=.12,initial_policy_std=.08),
        action_projector=AbsoluteGoalJawProjector())
    agent.correction_radius=.3
    agent.executed_body_anchor=lambda raw:raw.new_zeros(len(raw),19)
    if encoder is not None:agent.goal_servo_critic_encoder=encoder
    return agent


def test_encoded_Q_respects_actual_command_equivalence_and_old_default_keeps_goals():
    raw,encoder=decoder_fixture(1);agent=make_agent(encoder)
    a=torch.zeros(1,4,21);a[:,:,19:]=1;a[:,:,1]=.5
    b=a.clone();b[:,:,1]=.9
    critic=torch.zeros(1,4)
    assert torch.equal(agent.branch_values(critic,a,raw=raw),agent.branch_values(critic,b,raw=raw))
    old=make_agent()
    assert old.critic_action_features(raw,a[:,0]) is a[:,0] or torch.equal(old.critic_action_features(raw,a[:,0]),a[:,0])
    assert old.hybrid_contract['Q_action_coordinates']=='actual_absolute_projected_goals'
    assert 'critic_action_encoding' not in old.hybrid_contract
    assert agent.hybrid_contract['critic_action_encoding']==encoder.contract
    with pytest.raises(ValueError,match='continuous-only'):agent.restore(old.checkpoint())


def test_one_step_target_actor_and_nstep_all_use_measured_servo_and_skip_terminal_placeholders():
    raw,encoder=decoder_fixture(8);agent=make_agent(encoder)
    calls=[]
    class RecordingEncoder:
        contract=encoder.contract
        def __call__(self,obs,goals):
            actual=encoder(obs,goals);calls.append((obs.clone(),goals.detach().clone(),actual.detach().clone()))
            return actual
    agent.goal_servo_critic_encoder=RecordingEncoder()
    action=agent.act(raw,True)
    term=torch.tensor([True,True,True,True,False,False,False,False])
    next_raw=raw.clone();next_raw[term]=0
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,4),action=action,
        next_actor_obs=next_raw,next_critic_obs=torch.zeros(8,4),reward=torch.zeros(8),terminated=term)
    original=deepcopy(batch)
    # The base class strictly opts into auxiliary validation; this test uses
    # the same measured-nstep configuration as the production subclass.
    agent.measured_train_credit_enabled=True
    aux=deepcopy(batch);aux.update(n_steps=torch.full((8,),16),
        bootstrap_discount=torch.full((8,),agent.config.gamma**16).masked_fill(term,0.))
    report=agent.update(batch,critic_auxiliary=aux,critic_auxiliary_weight=.1)
    assert report['actor_updated'] and report['measured_nstep_rows']==8
    # Target branches16, current8, nstep target16, nstep current8, actor32.
    assert [len(c[0]) for c in calls]==[16,8,16,8,32]
    assert all(c[0][:,-6].eq(1).all() for c in calls)
    for obs,goals,actual in calls:assert torch.equal(actual,encoder(obs,goals))
    assert all(torch.equal(v,batch[k]) for k,v in original.items())
    state=agent.checkpoint();restored=make_agent(encoder);restored.restore(state)
    assert torch.equal(agent.act(raw,True),restored.act(raw,True))


def test_only_fresh_initial_checkpoint_can_enable_encoder_and_resume_preserves_models(tmp_path):
    from test_rl_actual_flap_reanchored_sac import learned_source
    from prepare_servo_critic_actor import prepare
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_servo_critic_sac import ServoCriticReanchoredSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
    _,old,warm,physical,stage,_=learned_source(tmp_path)
    old.directory.mkdir();old.save(final=True)
    old_cp=next(old.directory.glob('checkpoint_*.pt'))
    initial=torch.load(old_cp,weights_only=True)
    replay=torch.load(old.directory/'staged_goal_experience.pt',weights_only=True)
    state,experience=prepare(initial,replay)
    directory=tmp_path/'encoded';directory.mkdir()
    cp=directory/'checkpoint_00000000.pt';torch.save(state,cp)
    torch.save(experience,directory/'staged_goal_experience.pt')
    candidate=ServoCriticReanchoredSACPilot(warm,physical,tmp_path/'resume',stage,checkpoint=cp)
    assert candidate.contract==state['goal_contract']
    assert candidate.replay.size==0 and candidate.actor_updates==candidate.critic_updates==0
    assert not any(o.state for o in candidate.agent.optimizers)
    assert staged_policy_class(candidate.artifact_type) is ServoCriticReanchoredSACPilot
    raw=torch.zeros(8,518);raw[:,144]=1;raw[:,173]=1;raw[:,-6]=1;raw[:,-1]=.3
    assert torch.equal(old.agent.act(raw,True),candidate.agent.act(raw,True))
    for k,v in initial['model'].items():assert torch.equal(v,state['model'][k])
    with pytest.raises(ValueError,match='Old observation/control'):
        ServoCriticReanchoredSACPilot(warm,physical,tmp_path/'wrong',stage,checkpoint=old_cp)
    with pytest.raises(ValueError,match='Learned Q'):
        prepare(initial|dict(critic_updates=1),replay)
    contaminated=deepcopy(initial);contaminated['optimizers'][1]['state'][0]={'step':torch.tensor(1.)}
    with pytest.raises(ValueError,match='Learned Q'):prepare(contaminated,replay)
