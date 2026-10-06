"""Smaller Gaussian keeps means and jaws, and reaches all learner samplers."""
from copy import deepcopy
from dataclasses import asdict
import math

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.algorithms.hybrid_goal_sac import HybridGoalSAC
from kuavo_isaaclab_scene.rl.multi_box.experiments.body_policy_spread import (
    quarter_body_policy_config,shift_body_log_std,validate_quarter_policy_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.gentle_servo_critic_sac import GentleServoCriticSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


class AllNear:
    name='test_all_near_binary_jaws'
    free_grippers=True
    def __call__(self,raw,actions):return actions
    def entropy_mask(self,raw):return torch.ones(len(raw),21)


def agents():
    cfg=SACConfig(hidden=16,initial_policy_std=.1,min_policy_std=.03,max_policy_std=.12,entropy_backup=False)
    old=HybridGoalSAC(7,10,21,cfg,'cpu',action_projector=AllNear())
    new=HybridGoalSAC(7,10,21,quarter_body_policy_config(cfg),'cpu',action_projector=AllNear())
    model,_=shift_body_log_std(old.state_dict());new.load_state_dict(model)
    return old,new


def test_real_hybrid_mean_and_jaws_unchanged_and_gaussian_std_is_exactly_one_quarter():
    old,new=agents();raw=torch.randn(23,7)
    assert torch.equal(old.act(raw,True),new.act(raw,True))
    om,os,oj=old.parameters_at(raw);nm,ns,nj=new.parameters_at(raw)
    assert torch.equal(om,nm) and torch.equal(oj,nj)
    torch.testing.assert_close(ns.exp(),.25*os.exp(),atol=1e-8,rtol=2e-6)
    assert new.target_entropy_per_dim==pytest.approx(old.target_entropy_per_dim+math.log(.25),abs=1e-12)


def test_log_std_bias_is_shifted_inside_new_bounds_so_its_gradient_is_not_clamped_away():
    _,new=agents();raw=torch.randn(8,7)
    new.parameters_at(raw)[1].exp().sum().backward()
    gradient=new.actor.network[-1].bias.grad
    assert torch.isfinite(gradient).all() and (gradient[21:40]>0).all()
    assert gradient[:21].eq(0).all() and gradient[40:].eq(0).all()


def test_actual_target_Q_sampler_uses_the_reduced_variance_not_only_collection():
    old,new=agents()
    class QuadraticQ(torch.nn.Module):
        def forward(self,x):return -x[:,-21:-2].square().sum(-1,keepdim=True)
    for agent in (old,new):
        agent.target1=QuadraticQ();agent.target2=QuadraticQ()
        with torch.no_grad():
            agent.actor.network[-1].weight[:19].zero_();agent.actor.network[-1].bias[:19].zero_()
    batch=dict(next_actor_obs=torch.zeros(1024,7),next_critic_obs=torch.zeros(1024,10),
        reward=torch.zeros(1024),terminated=torch.zeros(1024,dtype=torch.bool))
    torch.manual_seed(717);a,_=old.critic_target(batch,torch.tensor(0.),torch.tensor(0.))
    torch.manual_seed(717);b,_=new.critic_target(batch,torch.tensor(0.),torch.tensor(0.))
    assert (a<0).all() and (b<0).all()
    assert 15<float(a.mean()/b.mean())<17


def initializer_inputs():
    from test_rl_absorbing_geometry import initializer_inputs as base
    state,experience=base()
    state['config']=asdict(SACConfig(initial_policy_std=.1,min_policy_std=.03,max_policy_std=.12))
    state['model']['actor.network.4.bias']=torch.zeros(42)
    return state,experience


def test_new_initializer_keeps_mean_Q_tensors_and_clears_old_reward_labels_and_exact_dispatch():
    from prepare_gentle_servo_actor import prepare
    from prepare_actual_success_actor_tail import identical
    state,experience=initializer_inputs();before=deepcopy(state);exp=deepcopy(experience)
    new,replay,bias=prepare(state,experience)
    assert identical(state,before) and identical(experience,exp)
    assert new['artifact_type']==GentleServoCriticSACPilot.artifact_type
    assert staged_policy_class(new['artifact_type']) is GentleServoCriticSACPilot
    assert new['config']['min_policy_std']==.0075 and new['config']['max_policy_std']==.03
    assert torch.equal(new['model'][bias][:21],state['model'][bias][:21])
    assert torch.equal(new['model']['q1.weight'],state['model']['q1.weight'])
    assert replay['goal_contract']==new['goal_contract']
    assert not any(new['successful_train_transitions']['episodes'].values())
    assert not any(replay['measured_train_credit_bank']['episodes'].values())
    validate_quarter_policy_state(new)


@pytest.mark.parametrize('bad',['trained_Q','unknown_std','nonfinite_actor','unknown_profile','bad_entropy'])
def test_old_learning_unknown_std_and_unknown_saved_identity_are_rejected(bad):
    from prepare_gentle_servo_actor import prepare
    state,experience=initializer_inputs()
    if bad=='trained_Q':state['critic_updates']=1
    if bad=='unknown_std':state['config']['min_policy_std']=.01
    if bad=='nonfinite_actor':state['model']['actor.network.4.bias'][0]=float('nan')
    if bad in ('unknown_profile','bad_entropy'):
        new,_,_=prepare(state,experience)
        if bad=='unknown_profile':new['goal_contract']['body_policy_spread']={}
        else:new['entropy_contract']['target_per_dim']=-1.
        with pytest.raises(ValueError):validate_quarter_policy_state(new)
    else:
        with pytest.raises(ValueError):prepare(state,experience)
