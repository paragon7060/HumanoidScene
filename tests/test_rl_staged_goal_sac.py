"""Protect held-phase replay and fresh-Q migration across the new controller."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import (
    GOAL_COLUMNS, StagedGoalProjector, StagedGoalSACPilot, copy_remaining_actor, staged_context,
    held_goal_coordinates)


def test_frozen_mean_matches_old_policy_after_removing_base_without_copying_Q():
    torch.manual_seed(42)
    bc=AsymmetricSAC(472,531,24,SACConfig(hidden=16))
    prior=SimpleNamespace(agent=bc)
    source=AsymmetricSAC(474,533,24,SACConfig(hidden=16),
        action_projector=GoalGripperProjector(prior,.05))
    target=AsymmetricSAC(480,539,21,SACConfig(hidden=16,initial_policy_std=.005),
        action_projector=StagedGoalProjector(prior,472))
    before={k:v.clone() for k,v in target.q1.state_dict().items()}
    source.actor_normalizer.mean.copy_(torch.randn(474))
    source.actor_normalizer.var.copy_(torch.rand(474)+.1)
    copy_remaining_actor(source,target)
    raw=torch.randn(8,474)
    raw[:,108:144]=0
    raw[:,144:146]=torch.tensor([1.,0.])
    context=torch.tensor([1.,.1,.7,0.,1.,.05]).expand(8,-1)
    torch.testing.assert_close(target.act(torch.cat((raw,context),-1),True),
                               source.act(raw,True)[:,list(GOAL_COLUMNS)],atol=1e-6,rtol=0)
    for k,v in target.q1.state_dict().items():assert torch.equal(v,before[k])
    assert not target.actor_optimizer.state and not target.q_optimizer.state


def test_staged_context_rejects_the_moving_base_and_records_actual_held_waypoint():
    raw=torch.zeros(1,464)
    stage=SimpleNamespace(phase='approach',manipulation_start=None,
                          target_xy=torch.tensor([[.12,.7]]),target_yaw=.3)
    with pytest.raises(ValueError,match='physically confirmed'):staged_context(raw,stage,.05)
    stage.phase='held_grasp';stage.manipulation_start=68
    result=staged_context(raw,stage,.07)
    assert result.shape==(1,6) and result[0,0]==1
    torch.testing.assert_close(result[0,1:3],stage.target_xy[0])
    assert result[0,-1]==pytest.approx(.07)


def test_remaining_goals_cannot_replace_the_held_base_targets():
    stage=SimpleNamespace(target_xy=torch.tensor([[.13,.72]]),target_yaw=.4)
    class Decoder:
        def decode(self,raw,goal):return goal
    raw=torch.zeros(1,464);remaining=torch.randn(1,21)
    complete=held_goal_coordinates(Decoder(),raw,remaining,stage)
    torch.testing.assert_close(complete[:,list(GOAL_COLUMNS)],remaining)
    torch.testing.assert_close(complete[:,19:21],stage.target_xy)
    assert complete[0,21]==pytest.approx(.4)


def test_independent_jaws_can_reverse_a_prior_close_but_cannot_close_far_away():
    class Prior:
        def act(self,observation,deterministic=True):
            return observation.new_ones(len(observation),24)
    prior=SimpleNamespace(agent=Prior())
    raw=torch.zeros(2,480);raw[:,144]=1;raw[:,-1]=.05
    raw[1,108:144]=1
    requested=torch.full((2,21),-1.)
    locked=StagedGoalProjector(prior,472)(raw,requested)
    free=StagedGoalProjector(prior,472,True)(raw,requested)
    assert locked[0,19:].gt(0).all() and free[:,19:].eq(-1).all()
    requested[:]=1.
    assert StagedGoalProjector(prior,472,True)(raw,requested)[1,19:].eq(-1).all()
    torch.testing.assert_close(free[:,:19],locked[:,:19])


def test_held_phase_critic_warmup_and_actual_replay_survive_a_new_trial(tmp_path):
    """Test fixture transitions exercise learning/storage, not physical success."""
    class Coordinates:
        name='test_remaining_coordinates'
        def box_anchor(self,raw):return raw.new_zeros(len(raw),2)
        def decode(self,raw,goal):return goal
    config=SACConfig(hidden=16,gamma=.999,freeze_actor_normalizer=True)
    bc=AsymmetricSAC(472,531,24,config)
    prior=SimpleNamespace(agent=bc,state={})
    source=AsymmetricSAC(474,533,24,config,action_projector=GoalGripperProjector(prior,.05))
    source.requires_grad_(False);bc.requires_grad_(False)
    def observations(raw,critic,index,anchor):
        return torch.cat((raw,raw.new_zeros(len(raw),10)),-1),torch.cat((critic,raw.new_zeros(len(raw),3)),-1)
    warm=SimpleNamespace(training=False,agent=source,prior=prior,coordinates=Coordinates(),
        center=torch.zeros(24),scale=torch.ones(24),actor_dim=474,
        actor_updates=16910,critic_updates=17410,artifact_type='pose_goal_sac_no_live_reference',
        contract={'source':'unit_test_only'},observations=observations)
    stage=SimpleNamespace(phase='held_grasp',manipulation_start=0,target_xy=torch.tensor([[.12,.7]]),
        target_yaw=.3,name='test_waypoints',templates={'middle':{'source_sha256':'test'}})
    contract={'discount':.999,'reward_profile':{'weights':{'discount':.999}}}
    pilot=StagedGoalSACPilot(warm,contract,tmp_path/'source',stage)
    raw=torch.zeros(1,464);raw[:,144]=1
    critic=torch.zeros(1,530)
    original_actor={k:v.clone() for k,v in pilot.agent.actor.state_dict().items()}
    for index in range(64):
        _,previous=pilot.act(raw,critic,index)
        pilot.observe(previous,raw,critic,torch.ones(1),torch.zeros(1,dtype=torch.bool),index)
    assert pilot.actor_updates==0 and pilot.critic_updates==2
    assert pilot.replay.size==64 and pilot.replay.data['action'].shape[-1]==21
    assert not any(p.requires_grad for p in pilot.agent.target1.parameters())
    for k,v in pilot.agent.actor.state_dict().items():assert torch.equal(v,original_actor[k])
    pilot.directory.mkdir()
    pilot.save(final=True)
    checkpoint=next(pilot.directory.glob('checkpoint_*.pt'))
    restored=StagedGoalSACPilot(warm,contract,tmp_path/'next',stage,checkpoint=checkpoint)
    assert restored.replay.size==64 and restored.critic_updates==2 and restored.actor_updates==0
    for key in pilot.replay.data:
        torch.testing.assert_close(restored.replay.data[key][:64],pilot.replay.data[key][:64])
    for key,value in pilot.agent.q1.state_dict().items():
        torch.testing.assert_close(restored.agent.q1.state_dict()[key],value)
    state=torch.load(checkpoint,weights_only=True)
    assert state['frozen_warm_start']['format_version']==1
    assert state['goal_contract']['old_Q_or_replay_imported'] is False
