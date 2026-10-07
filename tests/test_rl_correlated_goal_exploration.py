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


def test_episode_greedy_rows_keep_projected_body_and_binary_jaws_and_only_selected_noise_history():
    class Hybrid:
        def act(self,observation,deterministic):
            assert deterministic
            result=observation[:,:21].clone();result[:,19:]=torch.tensor([-1.,1.])
            return result
        def act_with_latent_noise(self,observation,noise,body_latent_offset=None):
            result=(observation[:,:21]+.1*noise).tanh()
            if body_latent_offset is not None:result[:,:19]+=body_latent_offset
            result[:,19:]=torch.where(noise[:,19:]>0,1.,-1.)
            return result
    agent=Hybrid();obs=torch.linspace(-.2,.2,4*21).reshape(4,21)
    ids=torch.tensor([8,2,7,1]);greedy=torch.tensor([True,False,True,False])
    offsets=torch.full((4,19),.02)
    mixed=CorrelatedGoalExploration(.99,10,21,'cpu')
    reference=CorrelatedGoalExploration(.99,10,21,'cpu')
    torch.manual_seed(24)
    actual=mixed.act(agent,obs,ids,body_latent_offset=offsets,greedy_mask=greedy)
    torch.manual_seed(24)
    expected=reference.act(agent,obs[~greedy],ids[~greedy],body_latent_offset=offsets[~greedy])
    assert torch.equal(actual[greedy],agent.act(obs[greedy],True))
    assert torch.equal(actual[~greedy],expected)
    assert mixed.initialized.nonzero().flatten().tolist()==[1,2]
    assert torch.equal(mixed.noise,reference.noise)
    # Finishing another environment does not change a retained greedy mode.
    again=mixed.act(agent,obs[[2,1]],ids[[2,1]],greedy_mask=greedy[[2,1]])
    assert torch.equal(again[0],agent.act(obs[2:3],True)[0])
    assert not mixed.initialized[7] and not mixed.initialized[8]


@pytest.mark.parametrize('mask',[torch.tensor([1,0]),torch.tensor([True]),[True,False]])
def test_invalid_mixed_collection_mask_is_rejected_before_action_or_noise(mask):
    sampler=CorrelatedGoalExploration(.99,2,21,'cpu')
    with pytest.raises(ValueError,match='boolean'):
        sampler.act(None,torch.zeros(2,21),torch.tensor([0,1]),greedy_mask=mask)
    assert not sampler.initialized.any()
