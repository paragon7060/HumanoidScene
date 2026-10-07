from types import SimpleNamespace
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.observations.task_timing import (
    CRITIC_HELD_CLOCK_INDEX, CRITIC_REMAINING_INDEX, TASK_TIMING_GROUP, VARIANT, critic_episode_clock_config,
    add_critic_task_time, resolve_critic_episode_clock, task_remaining_fraction)


def test_task_deadline_distinguishes_equal_held_clocks_and_partial_reset():
    # Equal time since manipulation does not imply equal task time: base
    # approach used different amounts of the same episode's budget.
    ready = torch.tensor([800, 850, 900])
    remaining = task_remaining_fraction(ready, 900)
    assert remaining[0] > remaining[1] > remaining[2] and remaining[2] == 0
    reset = ready.clone(); reset[1] = 0
    following = task_remaining_fraction(reset, 900)
    assert following[1] == 1 and torch.equal(following[[0, 2]], remaining[[0, 2]])
    critic = torch.randn(3, 577); critic[:, CRITIC_HELD_CLOCK_INDEX] = .8
    before = critic.clone(); result = add_critic_task_time(critic, remaining)
    assert torch.equal(critic, before)
    assert result.shape==(3,578)
    assert torch.equal(result[:, :CRITIC_REMAINING_INDEX], critic[:, :CRITIC_REMAINING_INDEX])
    assert torch.equal(result[:, CRITIC_REMAINING_INDEX+1:], critic[:, CRITIC_REMAINING_INDEX:])
    assert torch.equal(result[:, CRITIC_REMAINING_INDEX:CRITIC_REMAINING_INDEX+1], remaining)


@pytest.mark.parametrize('remaining', [None, torch.zeros(2), torch.full((2, 1), -1.),
    torch.full((2, 1), 1.1), torch.full((2, 1), float('nan'))])
def test_missing_or_invalid_measured_time_cannot_fall_back_to_held_clock(remaining):
    with pytest.raises(ValueError, match='Measured pre-autoreset'):
        add_critic_task_time(torch.zeros(2, 577), remaining)


def test_clock_contract_requires_prepared_matching_checkpoint():
    assert resolve_critic_episode_clock({}) is None
    clock = critic_episode_clock_config(VARIANT)
    assert resolve_critic_episode_clock(None, VARIANT) == clock
    assert resolve_critic_episode_clock({'critic_episode_clock': clock}) == clock
    with pytest.raises(ValueError, match='fresh Q'):
        resolve_critic_episode_clock({'critic_updates': 5}, VARIANT)
    with pytest.raises(ValueError, match='differs'):
        resolve_critic_episode_clock({'critic_episode_clock': clock | {'critic_feature_index': 0}})


def test_fresh_pilot_keeps_actor_and_rejects_missing_time_and_legacy_resume(tmp_path):
    from test_rl_actual_flap_residual_sac import pilots
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot, actor_anchor_state
    _, old, warm, physical, stage, source = pilots(tmp_path)
    new = ActualFlapResidualSACPilot(warm, physical, tmp_path/'timed', stage,
        body_anchor_state=actor_anchor_state(source), replay_capacity=1024,
        exploration_correlation=.99, train_success_retention=True, critic_episode_clock=VARIANT)
    raw=torch.zeros(2,464);raw[:,144]=1.;critic=torch.zeros(2,530);extra=torch.randn(2,38)
    old.anchor=new.anchor=torch.zeros(2,2)
    old_ao,old_co=old.observations(raw,critic,20,extra)
    ao,co=new.observations(raw,critic,20,extra,critic_episode_remaining=torch.tensor([[.2],[.8]]))
    assert torch.equal(ao,old_ao) and old.contract != new.contract
    torch.testing.assert_close(new.agent.act(ao,True),old.agent.act(old_ao,True),atol=1e-6,rtol=0)
    assert co.shape==(2,578) and new.critic_dim==578
    assert co[:,CRITIC_REMAINING_INDEX].tolist()==pytest.approx([.2,.8])
    assert torch.equal(co[:,CRITIC_HELD_CLOCK_INDEX],old_co[:,CRITIC_HELD_CLOCK_INDEX])
    with pytest.raises(ValueError,match='Measured pre-autoreset'):new.observations(raw,critic,20,extra)
    new.directory.mkdir();new.save(final=True)
    restored=ActualFlapResidualSACPilot(warm,physical,tmp_path/'restored',stage,
        checkpoint=next(new.directory.glob('checkpoint_*.pt')),training=False)
    assert restored.critic_episode_clock==new.critic_episode_clock and restored.contract==new.contract
    old.directory.mkdir();old.save(final=True)
    with pytest.raises(ValueError,match='fresh Q'):
        ActualFlapResidualSACPilot(warm,physical,tmp_path/'bad',stage,
            checkpoint=next(old.directory.glob('checkpoint_*.pt')),critic_episode_clock=VARIANT)


def test_quarantined_rows_keep_matching_pre_and_terminal_remaining_time():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_physics_training import act_measured_held_rows
    from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import observe_measured_held_rows
    class Stages:
        anchors=torch.zeros(4,2)
        def held_context(self,ids):return SimpleNamespace(phase='held_grasp',ids=ids.clone(),
            target_xy=self.anchors[ids].clone(),target_yaw=torch.zeros(len(ids)))
    class Pilot:
        device='cpu';critic_episode_clock=critic_episode_clock_config(VARIANT)
        def act(self,raw,critic,clocks,**options):
            self.before=options['critic_episode_remaining'].clone()
            return torch.zeros(len(raw),24),(raw,critic,raw.clone())
        def observe(self,previous,raw,critic,reward,terminated,clocks,**options):
            self.after=options['critic_episode_remaining'].clone()
    p=Pilot();stages=Stages();ids=torch.tensor([3,1,0]);clocks=torch.tensor([1,2,3])
    obs={'policy':torch.zeros(4,464),'critic':torch.zeros(4,66),TASK_TIMING_GROUP:torch.tensor([[.1],[.2],[.3],[.4]])}
    _,previous=act_measured_held_rows(p,stages,ids,obs,clocks)
    terminal=obs|{TASK_TIMING_GROUP:torch.tensor([[.09],[.19],[.29],[.39]])}
    added=observe_measured_held_rows(p,stages,ids,previous,terminal,torch.zeros(4),
        torch.ones(4,dtype=torch.bool),clocks,torch.tensor([True,False,True,True]))
    assert added==2 and p.before.flatten().tolist()==pytest.approx([.4,.2,.1])
    assert p.after.flatten().tolist()==pytest.approx([.39,.09])
