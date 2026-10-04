"""Behavior correlation never changes deployed means or cross-episode noise."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.correlated_goal_exploration import CorrelatedGoalExploration


def test_stationary_marginal_and_temporal_correlation_keep_noise_scale():
    torch.manual_seed(8)
    noise=CorrelatedGoalExploration(.98,2048,21,'cpu')
    ids=torch.arange(2048)
    first=noise.sample_noise(ids);second=noise.sample_noise(ids)
    assert first.mean().abs()<.015 and second.mean().abs()<.015
    assert first.std()==pytest.approx(1.,abs=.02)
    assert second.std()==pytest.approx(1.,abs=.02)
    correlation=torch.corrcoef(torch.stack((first.flatten(),second.flatten())))[0,1]
    assert correlation==pytest.approx(.98,abs=.002)


def test_active_ids_keep_their_own_history_and_new_wave_has_none():
    torch.manual_seed(3)
    noise=CorrelatedGoalExploration(.98,4,21,'cpu')
    first=noise.sample_noise(torch.tensor([3,1]))
    assert not noise.initialized[0] and not noise.initialized[2]
    saved=noise.noise[3].clone()
    noise.sample_noise(torch.tensor([1]))
    assert torch.equal(noise.noise[3],saved)
    noise.sample_noise(torch.tensor([0,3]))
    assert not noise.initialized[2]
    fresh=CorrelatedGoalExploration(.98,4,21,'cpu')
    assert not fresh.initialized.any() and not fresh.noise.any()
    for ids in (torch.tensor([0,0]),torch.tensor([4]),torch.tensor([-1])):
        with pytest.raises(ValueError,match='Distinct valid'):noise.sample_noise(ids)


def test_behavior_samples_existing_actor_std_and_preserves_goal_projection():
    def network(obs):
        return torch.cat((obs.new_full((len(obs),21),.2),
                          obs.new_full((len(obs),21),-5.3)),-1)
    projected=[]
    def projection(obs,action):
        projected.append(action.clone());result=action.clone();result[:,19:]=-1.;return result
    actor=SimpleNamespace(network=network,log_std_min=-6.,log_std_max=-4.)
    agent=SimpleNamespace(actor=actor,actor_normalizer=lambda x:x,actor_features=lambda x:x,
                          action_projector=projection)
    sampler=CorrelatedGoalExploration(.98,2,21,'cpu')
    obs=torch.zeros(2,480);torch.manual_seed(1)
    action=sampler.act(agent,obs,torch.tensor([0,1]))
    torch.testing.assert_close(projected[0],(.2+torch.exp(torch.tensor(-5.3))*sampler.noise).tanh())
    assert action[:,19:].eq(-1).all()
    torch.testing.assert_close(action[:,:19],projected[0][:,:19])
