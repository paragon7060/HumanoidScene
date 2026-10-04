"""Exact hybrid policy expectations; fixtures are not physical success data."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.hybrid_goal_sac import HybridGoalSAC
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.correlated_goal_exploration import CorrelatedGoalExploration


def agent_and_observations(n=2):
    class Prior:
        def act(self,raw,deterministic=True):return raw.new_zeros(len(raw),24)
    projector=StagedGoalProjector(SimpleNamespace(agent=Prior()),474,True)
    agent=HybridGoalSAC(480,4,21,SACConfig(hidden=16,gamma=.99,entropy_backup=False,
        max_policy_std=.02,min_policy_std=.001,initial_policy_std=.005),action_projector=projector)
    obs=torch.zeros(n,480);obs[:,144]=1;obs[:,-1]=.05
    return agent,obs


def test_four_probabilities_and_far_jaw_mask_are_an_exact_expectation():
    agent,obs=agent_and_observations()
    obs[1,108:144]=1
    logits=torch.tensor([[-1.3862944,1.3862944],[-1.3862944,1.3862944]])
    commands,weights,logs,near=agent.enumerate_jaws(obs,torch.zeros(2,19),logits)
    torch.testing.assert_close(weights[0],torch.tensor([.16,.64,.04,.16]))
    torch.testing.assert_close(weights[1],torch.tensor([1.,0.,0.,0.]))
    assert near.tolist()==[[True,True],[False,False]]
    torch.testing.assert_close(weights.sum(-1),torch.ones(2))
    torch.testing.assert_close(logs[0],weights[0].log())
    assert logs[1].eq(0).all() and commands[1,:,19:21].eq(-1).all()


def test_binary_jaw_policy_has_exact_Q_gradient_without_straight_through_sign():
    agent,obs=agent_and_observations()
    logits=torch.zeros(2,2,requires_grad=True)
    action,p,_,_=agent.enumerate_jaws(obs,torch.zeros(2,19),logits)
    # Q prefers left close/right open. Expected Q has derivatives1,-1.5,
    # although every Q action is exactly -1 or +1.
    q=2*action[:,:,19]-3*action[:,:,20]
    derivative=torch.autograd.grad((p*q).sum(),logits)[0]
    torch.testing.assert_close(derivative,torch.tensor([[1.,-1.5],[1.,-1.5]]))
    obs[:,108:144]=1
    action,p,_,_=agent.enumerate_jaws(obs,torch.zeros(2,19),logits)
    derivative=torch.autograd.grad((p*(2*action[:,:,19]-3*action[:,:,20])).sum(),logits)[0]
    assert derivative.eq(0).all()


def test_behavior_copula_keeps_binary_commands_and_deterministic_playback():
    torch.manual_seed(12)
    agent,obs=agent_and_observations(1024)
    with torch.no_grad():
        agent.actor.network[-1].weight.zero_();agent.actor.network[-1].bias.zero_()
        agent.actor.network[-1].bias[19]=1.3862944
        agent.actor.network[-1].bias[20]=-1.3862944
        agent.actor.network[-1].bias[21:]=torch.log(torch.tensor(.005))
    deterministic=agent.act(obs,True)
    assert deterministic[:,19].eq(1).all() and deterministic[:,20].eq(-1).all()
    sampler=CorrelatedGoalExploration(.98,1024,21,'cpu')
    sampled=sampler.act(agent,obs,torch.arange(1024))
    assert (sampled[:,19:21].abs()==1).all()
    torch.testing.assert_close((sampled[:,19:21]>0).float().mean(0),torch.tensor([.8,.2]),atol=.05,rtol=0)
    assert torch.equal(agent.act(obs,True),deterministic)


def test_learning_and_both_entropy_optimizers_survive_checkpoint():
    torch.manual_seed(8)
    agent,obs=agent_and_observations(64)
    actions=agent.act(obs)
    batch=dict(actor_obs=obs,critic_obs=torch.zeros(64,4),action=actions,
        next_actor_obs=obs.clone(),next_critic_obs=torch.zeros(64,4),
        reward=torch.ones(64),terminated=torch.ones(64,dtype=torch.bool))
    before={k:v.clone() for k,v in agent.actor.state_dict().items()}
    report=agent.update(batch)
    assert report['target_value_mean']==pytest.approx(1.)  # no terminal bootstrap
    assert report['actor_updated'] and report['near_jaw_count_mean']==2
    assert all(torch.isfinite(torch.tensor(v)) for v in report.values())
    assert any(not torch.equal(v,before[k]) for k,v in agent.actor.state_dict().items())
    assert len(agent.optimizers)==4 and agent.discrete_alpha_optimizer.state
    state=agent.checkpoint();restored,_=agent_and_observations(64);restored.restore(state)
    torch.testing.assert_close(agent.act(obs,True),restored.act(obs,True))
    assert len(restored.optimizers)==4
    assert torch.equal(agent.log_alpha_discrete,restored.log_alpha_discrete)
    with pytest.raises(ValueError,match='continuous-only'):restored.restore(state|dict(algorithm='asymmetric_sac'))
    with pytest.raises(ValueError,match='binary gripper'):agent.update(batch|dict(action=torch.zeros(64,21)))


def test_actor_recovery_keeps_hybrid_Q_actual_rows_and_both_temperature_states(tmp_path):
    from copy import deepcopy
    from recover_pose_goal_actor import recover
    agent,obs=agent_and_observations(64)
    initial=deepcopy(agent.checkpoint())
    batch=dict(actor_obs=obs,critic_obs=torch.zeros(64,4),action=agent.act(obs),
        next_actor_obs=obs.clone(),next_critic_obs=torch.zeros(64,4),
        reward=torch.ones(64),terminated=torch.ones(64,dtype=torch.bool))
    agent.update(batch)
    contract=dict(initial_critic_warmup=2048,fade_critic_updates=20000,
        same_physical_unit_fixture=True)
    meta=dict(artifact_type='staged_base_hold_remaining_hybrid_sac_v1',goal_contract=contract,
        actor_updates=1,critic_updates=2050)
    best=initial|meta|dict(actor_updates=0,critic_updates=0)
    latest=agent.checkpoint()|meta
    current=tmp_path/'current';current.mkdir()
    torch.save(latest,current/'checkpoint.pt');torch.save(best,tmp_path/'best.pt')
    (current/'manifest.json').write_text('{}')
    torch.save(dict(goal_contract=contract,executed_goal_transitions=batch),current/'staged_goal_experience.pt')
    path,audit=recover(current/'checkpoint.pt',tmp_path/'best.pt',tmp_path/'recovered',
        normalize_prior_loss_by_radius=True,anchor_prior_to_validated_policy=True,
        replay_capacity=1024,actor_min_replay_rows=128)
    state=torch.load(path,weights_only=True)
    for key,value in state['model'].items():
        assert torch.equal(value,(best if key.startswith('actor.') else latest)['model'][key])
    assert state['algorithm']=='hybrid_goal_sac' and len(state['optimizers'])==4
    assert state['optimizers'][0]['state']=={}
    for actual,expected in zip(state['optimizers'][1:],latest['optimizers'][1:]):
        assert actual['param_groups']==expected['param_groups']
        for parameter,values in actual['state'].items():
            for key,value in values.items():
                assert torch.equal(value,expected['state'][parameter][key])
    rows=torch.load(path.parent/'staged_goal_experience.pt',weights_only=True)['executed_goal_transitions']
    for key,value in batch.items():assert torch.equal(value,rows[key])
    assert audit['actual_rows']==64 and path.name=='checkpoint_00002050.pt'
    assert state['actor_updates']==1 and state['critic_updates']==2050
    invalid=best|dict(algorithm='asymmetric_sac');torch.save(invalid,tmp_path/'wrong.pt')
    with pytest.raises(ValueError,match='binary Q'):
        recover(current/'checkpoint.pt',tmp_path/'wrong.pt',tmp_path/'invalid')
