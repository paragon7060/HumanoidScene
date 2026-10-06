"""Collection coverage cannot alter body sampling or the production jaw gate."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.correlated_goal_exploration import CorrelatedGoalExploration
from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_behavior_exploration import (
    JointJawBehaviorExploration, VARIANT, jaw_behavior_config)


def agent(probability):
    def continuous(normalized, *, noise, body_latent_offset=None, raw=None):
        body = torch.tanh(.2+.05*noise+(0 if body_latent_offset is None else body_latent_offset))
        logits = torch.logit(normalized.new_tensor(probability)).expand(len(normalized), 2)
        return body, normalized.new_zeros(len(normalized)), logits
    def project(obs, body, closed):
        return torch.cat((body,torch.where(obs[:, :2].bool(),closed.to(body)*2-1,-1.)), -1)
    return SimpleNamespace(actor_normalizer=lambda x:x, actor_features=lambda x:x,
        continuous_sample=continuous, projected_command=project)


@pytest.mark.parametrize('probability', [[0.,1.],[.2,.7]])
@pytest.mark.parametrize('variant,epsilon', [('joint-epsilon10',.1),('joint-epsilon30',.3)])
def test_joint_mixture_distribution_and_exact_body_with_no_extra_rng(probability,variant,epsilon):
    torch.manual_seed(47)
    count=200000
    obs=torch.zeros(count,98);obs[:,:2]=1
    region=torch.arange(count)%4;obs[:,94:98].scatter_(1,region[:,None],1)
    noise=torch.randn(count,21)
    state=torch.random.get_rng_state().clone()
    sampler=JointJawBehaviorExploration(jaw_behavior_config(variant))
    action=sampler.act(agent(probability),obs,noise)
    assert torch.equal(state,torch.random.get_rng_state())
    assert torch.equal(action[:,:19],torch.tanh(.2+.05*noise[:,:19]))
    p,q=probability
    expected=torch.tensor([(1-p)*(1-q),(1-p)*q,p*(1-q),p*q])*(1-epsilon)+epsilon/4
    index=(action[:,19]>0).long()*2+(action[:,20]>0).long()
    observed=torch.bincount(index,minlength=4)/count
    torch.testing.assert_close(observed,expected,atol=.0025,rtol=0)
    assert sampler.report()['uniform_joint_rows']/count==pytest.approx(epsilon,abs=.0025)
    assert sampler.report()['rows']==count
    if variant=='joint-epsilon30':
        stats=sampler.report()
        assert stats['projected_branch_counts_by_region']==torch.bincount(region*4+index,minlength=16).reshape(4,4).tolist()
        restored=JointJawBehaviorExploration(jaw_behavior_config(variant),stats)
        assert restored.report()==stats


@pytest.mark.parametrize('variant', ('joint-epsilon10','joint-epsilon30'))
def test_far_jaws_remain_open_and_projected_counts_are_real_commands(variant):
    torch.manual_seed(8)
    obs=torch.zeros(1024,98);obs[512:,1]=1;obs[:,94]=1
    sampler=JointJawBehaviorExploration(jaw_behavior_config(variant))
    action=sampler.act(agent([1.,1.]),obs,torch.randn(1024,21))
    assert action[:512,19:21].eq(-1).all() and action[:,19].eq(-1).all()
    assert sampler.report()['projected_branch_counts'][2:]==[0,0]
    assert sum(sampler.report()['projected_branch_counts'])==1024


def test_default_sampler_and_global_environment_identity_are_unchanged():
    model=agent([.2,.7])
    def ordinary(obs,noise,**options):
        body,_,logits=model.continuous_sample(obs,noise=noise[:,:19],raw=obs,**options)
        u=.5*(1+torch.erf(noise[:,19:]/2**.5))
        return model.projected_command(obs,body,u<logits.sigmoid())
    model.act_with_latent_noise=ordinary
    ids=torch.tensor([3,1]);obs=torch.ones(2,2)
    torch.manual_seed(1);old=CorrelatedGoalExploration(.99,4,21,'cpu');a=old.act(model,obs,ids)
    torch.manual_seed(1);new=CorrelatedGoalExploration(.99,4,21,'cpu');b=new.act(model,obs,ids,jaw_behavior=None)
    assert torch.equal(a,b) and torch.equal(old.noise,new.noise)
    sampler=JointJawBehaviorExploration(jaw_behavior_config(VARIANT))
    saved=new.noise[3].clone()
    new.act(model,obs[:1],torch.tensor([1]),jaw_behavior=sampler)
    assert torch.equal(saved,new.noise[3]) and not new.initialized[0] and not new.initialized[2]


def test_invalid_config_and_statistics_cannot_silently_reset_provenance():
    assert jaw_behavior_config('policy') is None
    with pytest.raises(ValueError):jaw_behavior_config('unknown')
    with pytest.raises(ValueError):JointJawBehaviorExploration({'variant':VARIANT})
    with pytest.raises(ValueError):JointJawBehaviorExploration(jaw_behavior_config(VARIANT),
        dict(rows=2,uniform_joint_rows=3,projected_branch_counts=[1,1,0,0]))
