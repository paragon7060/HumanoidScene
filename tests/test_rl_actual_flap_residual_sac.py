"""Protect real action coordinates and terminal perception of the correction policy."""
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_hybrid_goal_sac import StagedHybridGoalSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot,actor_anchor_state
from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import actual_flap_relations


def pilots(tmp_path):
    class Coordinates:
        name='unit_test_absolute_remaining_goals'
        def box_anchor(self,raw):return raw.new_zeros(len(raw),2)
        def decode(self,raw,goal):return goal
    torch.manual_seed(8)
    config=SACConfig(hidden=16,gamma=.999,freeze_actor_normalizer=True,entropy_backup=False)
    bc=AsymmetricSAC(472,531,24,config).requires_grad_(False)
    prior=SimpleNamespace(agent=bc,state={})
    warm_agent=AsymmetricSAC(474,533,24,config,action_projector=GoalGripperProjector(prior,.05)).requires_grad_(False)
    def observations(raw,critic,index,anchor):
        return torch.cat((raw,raw.new_zeros(len(raw),10)),-1),torch.cat((critic,raw.new_zeros(len(raw),3)),-1)
    warm=SimpleNamespace(training=False,agent=warm_agent,prior=prior,coordinates=Coordinates(),
        center=torch.zeros(24),scale=torch.ones(24),actor_dim=474,actor_updates=1,critic_updates=1,
        artifact_type='unit_frozen_pose_goal_prior',contract={'unit_fixture_only':True},observations=observations)
    stage=SimpleNamespace(phase='held_grasp',manipulation_start=0,target_xy=torch.tensor([[.12,.7]]),
        target_yaw=.3,name='unit_waypoint',templates={'unit_fixture':True})
    physical={'discount':.999,'reward_profile':{'weights':{'discount':.999}},'terminal_contract':{}}
    old=StagedHybridGoalSACPilot(warm,physical,tmp_path/'old',stage,free_grippers=True,
        normalize_prior_loss_by_radius=True,anchor_prior_to_initial_policy=True,
        fixed_prior_radius=.05,validated_jaw_prior_confidence=.8,jaw_prior_residual_gain=20.,replay_capacity=1024)
    with torch.no_grad():
        old.agent.actor_normalizer.mean.copy_(torch.randn(480)*.03)
        old.agent.actor_normalizer.var.copy_(torch.rand(480)+.1)
        old.agent.actor.network[-1].bias[19].add_(.15)
    old.directory.mkdir();old.save(final=True)
    source=torch.load(next(old.directory.glob('checkpoint_*.pt')),weights_only=True)
    new=ActualFlapResidualSACPilot(warm,physical,tmp_path/'new',stage,
        body_anchor_state=actor_anchor_state(source),replay_capacity=1024,exploration_correlation=.99,
        train_success_retention=True)
    return old,new,warm,physical,stage,source


def test_zero_correction_keeps_executed_source_and_jaws_with_changed_context(tmp_path):
    old,new,*_=pilots(tmp_path)
    raw=torch.randn(12,464)*.03;raw[:,144:146]=torch.tensor([1.,0.]);raw[:,108:144]=0
    critic=torch.zeros(12,530)
    old.anchor=new.anchor=torch.zeros(12,2)
    nominal,_=old.observations(raw,critic,0)
    actual,_=new.observations(raw,critic,0,torch.randn(12,38)*10)
    assert new.actor_dim==518 and new.critic_dim==577
    assert torch.equal(new.nominal_view(actual),nominal)
    torch.testing.assert_close(new.agent.act(actual,True),old.agent.act(nominal,True),atol=1e-6,rtol=0)
    assert new.replay.size==new.actor_updates==new.critic_updates==new.success_bank.size==0
    assert not any(opt.state for opt in new.agent.optimizers)
    assert not new.agent.critic_normalizer.count and new.prior_weight==0
    assert not any(k.startswith(('q','target','optimizers')) for k in new.body_anchor_state['model'])


def test_correction_maps_once_for_collection_branches_targets_and_success_labels(tmp_path):
    _,new,*_=pilots(tmp_path)
    raw=torch.zeros(8,518);raw[:,144]=1;raw[:,-1]=.15
    with torch.no_grad():new.agent.actor.network[-1].bias[:19].fill_(.4)
    normalized=new.agent.actor_normalizer(raw)
    body,logp,logits=new.agent.continuous_sample(normalized,deterministic=True,raw=raw)
    anchor,scale=new.agent.anchor_and_scale(raw)
    expected=anchor+scale*torch.tanh(torch.full_like(anchor,.4))
    torch.testing.assert_close(body,expected)
    assert torch.isfinite(logp).all() and (body.abs()<=1).all()
    branches,*_=new.agent.enumerate_jaws(raw,body,logits)
    torch.testing.assert_close(branches[:,:,:19],expected[:,None].expand(-1,4,-1))
    noise=torch.zeros(8,21)
    torch.testing.assert_close(new.agent.act_with_latent_noise(raw,noise)[:,:19],expected)
    labels=new.agent.act(raw,True)
    assert new.agent.success_body_loss(raw,torch.tanh(torch.full_like(anchor,.4)),labels)==pytest.approx(0.)
    repeated=new.agent.action_projector(raw,new.agent.action_projector(raw,labels))
    torch.testing.assert_close(repeated,labels)
    derivative=torch.autograd.grad(body.sum(),new.agent.actor.network[-1].bias)[0]
    assert (derivative[:19]>0).all()
    batch=dict(actor_obs=raw,critic_obs=torch.zeros(8,577),action=labels,
        next_actor_obs=raw.clone(),next_critic_obs=torch.zeros(8,577),
        reward=torch.ones(8),terminated=torch.zeros(8,dtype=torch.bool))
    target,stats=new.agent.critic_target(batch,torch.tensor(0.),torch.tensor(0.))
    assert torch.isfinite(target).all() and stats['bootstrapped_rows']==8
    report=new.agent.update(batch,successful_train={'actor_obs':raw,'action':labels},success_goal_weight=10.,success_jaw_weight=.05)
    assert report['actor_updated'] and all(torch.isfinite(torch.tensor(v)) for v in report.values())


def test_new_replay_terminal_supplement_and_resume_remain_under_new_contract(tmp_path):
    old,new,warm,physical,stage,source=pilots(tmp_path)
    raw=torch.zeros(64,464);raw[:,144]=1;critic=torch.zeros(64,530)
    current=torch.randn(64,38);terminal=torch.randn(64,38)+4
    previous=new.act(raw,critic,0,supplemental=current)[1]
    new.observe(previous,raw,critic,torch.ones(64),torch.ones(64,dtype=torch.bool),0,supplemental=terminal)
    stored=new.replay.data
    torch.testing.assert_close(stored['actor_obs'][:64,474:512],current)
    torch.testing.assert_close(stored['next_actor_obs'][:64,474:512],terminal)
    torch.testing.assert_close(stored['action'][:64],previous[2])
    with pytest.raises(ValueError,match='supplemental'):new.act(raw,critic,0)
    new.directory.mkdir();new.save(final=True)
    checkpoint=next(new.directory.glob('checkpoint_*.pt'))
    restored=ActualFlapResidualSACPilot(warm,physical,tmp_path/'restored',stage,checkpoint=checkpoint)
    assert restored.replay.size==64 and restored.critic_updates==2
    restored.anchor=new.anchor
    torch.testing.assert_close(restored.agent.act(previous[0],True),new.agent.act(previous[0],True))
    with pytest.raises(ValueError,match='Old observation/control'):
        ActualFlapResidualSACPilot(warm,physical,tmp_path/'bad',stage,checkpoint=next(old.directory.glob('checkpoint_*.pt')))


def test_actual_midpoint_relations_track_panel_rotation_and_mask_invalid_pose():
    panels=torch.zeros(2,2,7);panels[:,:,3]=1
    tcp=panels.clone();tcp[:,:,:3]=.1
    sizes=torch.tensor([[.2,.2,.2],[.2,.2,.2]]);types=torch.zeros(2,dtype=torch.long)
    valid=torch.ones(2,dtype=torch.bool)
    baseline=actual_flap_relations(panels,sizes,types,tcp,valid)
    panels[0,0,3:]=torch.tensor([.7071068,0.,.7071068,0.])
    panels[1,0,3:]=0
    measured=actual_flap_relations(panels,sizes,types,tcp,valid)
    assert measured.shape==(2,38) and torch.isfinite(measured).all()
    assert not torch.equal(measured[0],baseline[0]) and measured[1].eq(0).all()


def test_terminal_mixin_captures_supplement_before_autoreset():
    from kuavo_isaaclab_scene.rl.envs.terminal_observation import TerminalObservationMixin
    class Parent:
        def __init__(self):
            self.values=dict(policy=torch.zeros(2,464),critic=torch.zeros(2,66),actual_flap_relations=torch.zeros(2,38))
            self.observation_manager=SimpleNamespace(compute=lambda **kwargs:self.values)
        def _reset_idx(self,ids):
            for value in self.values.values():value[ids]=-99
        def step(self,action):
            self.values['actual_flap_relations'][:]=torch.arange(38)
            self._reset_idx(torch.tensor([0]))
            return self.values,torch.zeros(2),torch.tensor([True,False]),torch.zeros(2,dtype=torch.bool),{}
    class Env(TerminalObservationMixin,Parent):pass
    env=Env();current,_,_,_,info=env.step(None)
    assert current['actual_flap_relations'][0].eq(-99).all()
    torch.testing.assert_close(info['transition_next_observations']['actual_flap_relations'][0],torch.arange(38).float())


def test_bound_coordinates_have_finite_inactive_entropy_and_no_double_projection(tmp_path):
    _,new,*_=pilots(tmp_path)
    agent=new.agent
    agent.executed_body_anchor=lambda raw:raw.new_full((len(raw),19),1.)
    raw=torch.zeros(2,518);raw[:,144]=1;normalized=agent.actor_normalizer(raw)
    body,logp,logits=agent.continuous_sample(normalized,raw=raw)
    assert body.eq(1).all() and logp.eq(0).all()
    assert agent.continuous_entropy_target(normalized,raw=raw).eq(0).all()
    # At a genuine interior state, affine entropy includes its physical
    # radius while the target shifts by the opposite amount.
    agent.executed_body_anchor=lambda raw:raw.new_zeros(len(raw),19)
    per_dim=raw.new_full((len(raw),19),2.)
    torch.testing.assert_close(agent.body_log_probability(normalized,per_dim,raw),
        (per_dim-torch.log(torch.tensor(.15))).sum(-1))
    torch.testing.assert_close(agent.body_entropy_target(normalized,per_dim,raw),
        (per_dim+torch.log(torch.tensor(.15))).sum(-1))


def test_train_jaw_behavior_activation_preserves_models_replay_and_frozen_eval(tmp_path):
    _,old,warm,physical,stage,_=pilots(tmp_path)
    raw=torch.zeros(64,464);raw[:,144]=1;critic=torch.zeros(64,530);extra=torch.zeros(64,38)
    previous=old.act(raw,critic,0,supplemental=extra)[1]
    old.observe(previous,raw,critic,torch.ones(64),torch.ones(64,dtype=torch.bool),0,supplemental=extra)
    old.directory.mkdir();old.save(final=True)
    checkpoint=next(old.directory.glob('checkpoint_*.pt'))
    state=torch.load(checkpoint,weights_only=True)
    new=ActualFlapResidualSACPilot(warm,physical,tmp_path/'behavior',stage,
        checkpoint=checkpoint,jaw_behavior='joint-epsilon10')
    assert new.contract==old.contract and new.actor_updates==old.actor_updates and new.critic_updates==old.critic_updates
    for key,value in state['model'].items():assert torch.equal(value,new.agent.state_dict()[key])
    for key,value in old.replay.data.items():assert torch.equal(value[:64],new.replay.data[key][:64])
    for before,after in zip(old.agent.optimizers,new.agent.optimizers):
        b,a=before.state_dict(),after.state_dict()
        assert b['param_groups']==a['param_groups'] and b['state'].keys()==a['state'].keys()
        for pid,values in b['state'].items():
            for key,value in values.items():
                assert torch.equal(value,a['state'][pid][key]) if isinstance(value,torch.Tensor) else value==a['state'][pid][key]
    assert new.jaw_behavior_origin['old_replay_rows_at_activation']==64
    assert new.jaw_behavior_sampler.report()['rows']==0
    new.anchor=old.anchor;new.training=old.training=False
    assert torch.equal(new.act(raw,critic,0,supplemental=extra)[0],old.act(raw,critic,0,supplemental=extra)[0])
    assert new.jaw_behavior_sampler.report()['rows']==0
    # Explicit frozen TRAIN behavior is still the learned policy, without the new mixture.
    new.act(raw,critic,0,supplemental=extra,sample_frozen_train_behavior=True)
    assert new.jaw_behavior_sampler.report()['rows']==0
    new.training=True;new.reset_exploration(64)
    new.act(raw,critic,1,supplemental=extra)
    assert new.jaw_behavior_sampler.report()['rows']==64
    new.directory.mkdir();new.save(final=True)
    saved=next(new.directory.glob('checkpoint_*.pt'))
    restored=ActualFlapResidualSACPilot(warm,physical,tmp_path/'restored_behavior',stage,checkpoint=saved)
    assert restored.jaw_behavior==new.jaw_behavior and restored.jaw_behavior_origin==new.jaw_behavior_origin
    assert restored.jaw_behavior_sampler.report()==new.jaw_behavior_sampler.report()
    with pytest.raises(ValueError,match='differs from checkpoint'):
        ActualFlapResidualSACPilot(warm,physical,tmp_path/'bad_behavior',stage,checkpoint=saved,jaw_behavior='policy')
    frozen=ActualFlapResidualSACPilot(warm,physical,tmp_path/'frozen_behavior',stage,checkpoint=saved,training=False)
    assert frozen.jaw_behavior==new.jaw_behavior and frozen.jaw_behavior_sampler.report()==new.jaw_behavior_sampler.report()
