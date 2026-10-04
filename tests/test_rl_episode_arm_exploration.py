import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.episode_arm_exploration import (
    EpisodeArmExploration,episode_arm_exploration_contract,
    enable_episode_arm_exploration,
)


def observations(n):
    value=torch.zeros(n,480);value[:,94]=1;value[:,-6]=1
    return value


def test_episode_bias_is_lazy_identity_preserving_smooth_and_arm_only():
    torch.manual_seed(91);rng=torch.random.get_rng_state().clone()
    sampler=EpisodeArmExploration(4,'cpu',episode_arm_exploration_contract())
    assert torch.equal(torch.random.get_rng_state(),rng)
    obs=observations(2);obs[1,94]=0;obs[1,96]=1
    ids=torch.tensor([3,1]);zero=sampler.offset(obs,torch.zeros(2),ids)
    assert zero.eq(0).all()
    half=sampler.offset(obs,torch.full((2,),45.),ids)
    full=sampler.offset(obs,torch.full((2,),90.),ids)
    torch.testing.assert_close(half,full*.5)
    torch.testing.assert_close(sampler.offset(obs,torch.full((2,),200.),ids),full)
    assert full[:,[0,15,16,17,18]].eq(0).all()
    assert full[0].abs().max()<=.01 and full[1].abs().max()<=.04
    assert not sampler.initialized[0] and not sampler.initialized[2]
    torch.testing.assert_close(sampler.offset(obs.flip(0),torch.full((2,),90.),ids.flip(0)),full.flip(0))


@pytest.mark.parametrize('invalid', ['duplicate_ids','invalid_region','mixed_region','unheld_phase','negative_clock','clock_shape','bad_contract'])
def test_arm_exploration_rejects_wrong_state_identity_or_contract(invalid):
    cfg=episode_arm_exploration_contract()
    if invalid=='bad_contract':cfg['upper_bias_std']=.5
    if invalid=='bad_contract':
        with pytest.raises(ValueError):EpisodeArmExploration(2,'cpu',cfg)
        return
    sampler=EpisodeArmExploration(2,'cpu',cfg);obs=observations(2);ids=torch.arange(2);clock=torch.ones(2)
    if invalid=='duplicate_ids':ids[:]=0
    elif invalid=='invalid_region':obs[:,94:98]=0
    elif invalid=='mixed_region':obs[:,94:98]=torch.tensor([.5,.5,0,0])
    elif invalid=='unheld_phase':obs[:,-6]=0
    elif invalid=='negative_clock':clock[0]=-1
    elif invalid=='clock_shape':clock=clock[:,None]
    with pytest.raises(ValueError):sampler.offset(obs,clock,ids)


def test_collection_migration_keeps_model_Q_rows_and_physical_contract_verbatim():
    contract=dict(gripper_prior_bound=False,fixed_prior_radius=.05,exploration_correlation=.98,
        collection_noise='independent_per_environment_AR1_pre_tanh_Gaussian',
        physical_contract=dict(solver='PGS',safe_threshold=10.))
    model=dict(weight=torch.randn(3,2));rows=dict(reward=torch.randn(3),action=torch.randn(3,21))
    state=dict(artifact_type='staged_base_hold_remaining_hybrid_sac_v1',goal_contract=contract,
        model=model,optimizers=[{'actual_state':torch.randn(2)}],actor_updates=17,critic_updates=92)
    experience=dict(goal_contract=contract,executed_goal_transitions=rows)
    new,actual,audit=enable_episode_arm_exploration(state,experience)
    assert new['model'] is state['model'] and new['optimizers'] is state['optimizers']
    assert new['actor_updates']==17 and new['critic_updates']==92
    assert actual['executed_goal_transitions'] is rows and 'episode_arm_exploration' not in contract
    assert actual['source_experience_collection_contract']==contract
    assert new['goal_contract']['physical_contract']==contract['physical_contract']
    assert audit['actual_replay_rows']==3 and audit['new_training_updates']==0
    with pytest.raises(ValueError):enable_episode_arm_exploration(new,actual)
    with pytest.raises(ValueError):enable_episode_arm_exploration(state,experience|dict(goal_contract={}))
