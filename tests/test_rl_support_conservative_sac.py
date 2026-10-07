"""Critic candidates are bounded queries, never synthetic success experience."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.support_conservative_sac import (
    SupportConservativeHybridSAC,SupportConservativeRegionalSACPilot)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def fixture(n=8):
    agent=SupportConservativeHybridSAC(518,4,21,SACConfig(hidden=16,freeze_actor_normalizer=True,
        entropy_backup=False,min_policy_std=.0075,max_policy_std=.03,initial_policy_std=.01),
        action_projector=AbsoluteGoalJawProjector())
    agent.correction_radius=.3
    def anchor(raw):
        a=raw.new_zeros(len(raw),19);a[:,0]=1
        return a
    agent.executed_body_anchor=anchor
    raw=torch.zeros(n,518);raw[:,144]=1
    return agent,raw


def test_candidate_queries_respect_frozen_coordinates_body_radius_and_far_jaws():
    agent,raw=fixture(2);raw[1,108:144]=1
    before=deepcopy(agent.state_dict());commands=agent.support_candidate_actions(raw)
    assert commands.shape==(2,32,21) and not commands.requires_grad
    assert commands[:,:,0].eq(1).all() and commands[:,:,1:19].abs().le(.3).all()
    assert commands[:,:,19:].abs().eq(1).all() and commands[1,:,19:].eq(-1).all()
    assert commands[0,:,19:].unique(dim=0).shape[0]==4
    assert all(torch.equal(v,agent.state_dict()[k]) for k,v in before.items())


def test_equal_Q_on_candidates_and_actual_commands_has_zero_support_gap_and_no_actor_gradient():
    agent,raw=fixture();critic=torch.zeros(8,4)
    for q in (agent.q1,agent.q2):
        for p in q.parameters():p.data.zero_()
    data_actions=agent.act(raw)
    features=torch.cat((critic,data_actions),-1)
    values=tuple(q(features).squeeze(-1) for q in (agent.q1,agent.q2))
    loss,report=agent.critic_regularization(dict(actor_obs=raw),critic,values)
    assert loss.item()==pytest.approx(0,abs=1e-7)
    loss.backward()
    assert all(p.grad is None for p in agent.actor.parameters())
    assert report['critic_support_synthetic_replay_rows']==0


def test_penalty_lowers_optimistic_unobserved_action_slope_without_fabricating_Q_targets():
    agent,raw=fixture(2);critic=torch.zeros(2,4)
    for name in ('q1','q2'):
        q=nn.Linear(25,1,bias=False);q.weight.data.zero_();q.weight.data[0,5]=2
        setattr(agent,name,q)
    # Actual commands use coordinate1=0. Critic queries cover both sides.
    candidate=torch.zeros(2,2,21);candidate[:,0,1]=-1;candidate[:,1,1]=1
    agent.support_candidate_actions=lambda obs:candidate
    action=torch.zeros(2,21);features=torch.cat((critic,action),-1)
    values=tuple(q(features).squeeze(-1) for q in (agent.q1,agent.q2))
    loss,_=agent.critic_regularization(dict(actor_obs=raw),critic,values);loss.backward()
    assert agent.q1.weight.grad[0,5]>0 and agent.q2.weight.grad[0,5]>0
    assert action.eq(0).all() and raw[:,144].eq(1).all()


def test_actual_update_and_checkpoint_preserve_the_separate_support_contract():
    torch.manual_seed(23);agent,raw=fixture()
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,4),action=agent.act(raw),
        next_actor_obs=raw.clone(),next_critic_obs=torch.zeros(8,4),reward=torch.ones(8),
        terminated=torch.ones(8,dtype=torch.bool))
    saved={k:v.clone() for k,v in batch.items()}
    report=agent.update(batch);restored,_=fixture();restored.restore(agent.checkpoint())
    assert report['actor_updated'] and report['critic_support_actual_TRAIN_rows']==8
    assert all(torch.equal(v,batch[k]) for k,v in saved.items())
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
    torch.testing.assert_close(agent.act(raw,True),restored.act(raw,True),atol=0,rtol=0)
    altered=deepcopy(agent.checkpoint());altered['hybrid_contract']['critic_support_regularization']['weight']=2.
    with pytest.raises(ValueError):restored.restore(altered)
    assert staged_policy_class(SupportConservativeRegionalSACPilot.artifact_type) is SupportConservativeRegionalSACPilot
