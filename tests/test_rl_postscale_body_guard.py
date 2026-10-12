"""Rounding repair must preserve outputs, budgets and rejection semantics."""
from copy import deepcopy

import pytest
import torch

from test_rl_regional_guard_metric_cache import regional_fixture
from test_rl_success_cohort_guard import sample
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.algorithms import success_cohort_guard as cohort
from kuavo_isaaclab_scene.rl.algorithms import success_update_guard as guard


def test_output_correction_preserves_hidden_std_jaw_and_input_state():
    agent=regional_fixture();head=agent.actor.network.heads[0]
    with torch.no_grad():head[-1].bias[1]+=.03
    batch,groups=cohort.cohort_batch(agent)
    params=list(head.parameters());before=[p.detach().clone() for p in params]
    memory=deepcopy(agent.success_guard_memory);optimizer=deepcopy(agent.actor_optimizer.state_dict())
    correction,kinds=cohort.quantized_body_output_correction(agent,batch,groups,params)
    assert correction is not None and kinds['body'] and torch.isfinite(correction).all()
    offset=0
    for p in params:
        part=correction[offset:offset+p.numel()].reshape_as(p);offset+=p.numel()
        if p is head[-1].weight or p is head[-1].bias:assert not part[19:].any()
        else:assert not part.any()
    assert all(torch.equal(p,v) for p,v in zip(params,before))
    assert equal(memory,agent.success_guard_memory) and equal(optimizer,agent.actor_optimizer.state_dict())
    with pytest.raises(ValueError,match='complete regional'):
        cohort.quantized_body_output_correction(agent,batch,groups,params[-2:])


def test_postscale_repairs_actual_candidate_without_scaling_away_correction(monkeypatch):
    agent=regional_fixture();params=list(agent.actor.network.heads[0].parameters())
    previous=[p.detach().clone() for p in params];proposal=[v.clone() for v in previous]
    proposal[-1][1]+=.001;target=float(previous[-1][1]);trust=torch.tensor(.002)
    seen=[]
    def metrics():return {'value':float(params[-1][1].detach())},torch.ones(1,2,dtype=torch.bool)
    monkeypatch.setattr(cohort,'within_ceilings',lambda values,*args:values['value']<=target)
    monkeypatch.setattr(cohort,'restoration_score',lambda values,*args:(values['value']-target,))
    def corrector(*args):
        seen.append(float(params[-1][1].detach()))
        v=torch.zeros(sum(p.numel() for p in params));offset=sum(p.numel() for p in params[:-1])+1
        v[offset]=target-float(params[-1][1].detach())-1e-6
        return v,dict(body=1,jaw_loss=0,jaw_label=0)
    monkeypatch.setattr(cohort,'quantized_body_output_correction',corrector)
    scale,report=guard.repair_scaled_body_candidate(agent,{}, {},params,previous,proposal,trust,metrics)
    assert seen and target<seen[0]<float(proposal[-1][1])
    assert scale==2.**-6 and report['accepted'] and metrics()[0]['value']<=target
    assert report['actual_delta_norm']<=trust and not torch.equal(params[-1],previous[-1])


def test_failed_postscale_restores_parameters_without_advancing_Adam(monkeypatch):
    agent=regional_fixture();params=list(agent.actor.network.heads[0].parameters())
    old=[p.detach().clone() for p in params];proposal=[v+.0001 for v in old]
    optimizer=deepcopy(agent.actor_optimizer.state_dict());memory=deepcopy(agent.success_guard_memory)
    monkeypatch.setattr(cohort,'within_ceilings',lambda *args:False)
    monkeypatch.setattr(cohort,'quantized_body_output_correction',lambda *args:(None,dict(body=1,jaw_loss=1,jaw_label=1)))
    scale,report=guard.repair_scaled_body_candidate(agent,{}, {},params,old,proposal,torch.tensor(.01),lambda:({},torch.ones(1,2,dtype=torch.bool)))
    assert scale==0 and not report['accepted']
    assert all(torch.equal(p,v) for p,v in zip(params,old))
    assert equal(optimizer,agent.actor_optimizer.state_dict()) and equal(memory,agent.success_guard_memory)


def test_accepted_ordinary_candidate_does_not_call_postscale(monkeypatch):
    agent,_=sample(cohort.TUBE_VARIANT)
    def forbidden(*args):raise AssertionError('Ordinary accepted candidate must remain unchanged')
    monkeypatch.setattr(guard,'repair_scaled_body_candidate',forbidden)
    accepted,report=guard.guarded_actor_step(agent,-agent.actor.network[-1].bias[1],None)
    assert accepted and report['actor_success_guard_postscale_body_fallback_regions']=={}
