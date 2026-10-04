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
    contract={'discount':.999,'reward_profile':{'weights':{'discount':.999}},'terminal_contract':{}}
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
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import staged_solver_contract
    with pytest.raises(ValueError,match='same phase/waypoint'):
        StagedGoalSACPilot(warm,contract|dict(physics_dynamics=staged_solver_contract('PGS')),
            tmp_path/'wrong_solver',stage,checkpoint=checkpoint)
    from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import grasp_lift_terminal_contract
    with pytest.raises(ValueError,match='same phase/waypoint'):
        StagedGoalSACPilot(warm,contract|dict(terminal_contract=grasp_lift_terminal_contract()),
            tmp_path/'wrong_lift_reference',stage,checkpoint=checkpoint)
    unchanged=checkpoint.read_bytes()
    pilot.save(final=True)
    assert checkpoint.read_bytes()==unchanged
    restored=StagedGoalSACPilot(warm,contract,tmp_path/'next',stage,checkpoint=checkpoint)
    assert restored.replay.size==64 and restored.critic_updates==2 and restored.actor_updates==0
    for key in pilot.replay.data:
        torch.testing.assert_close(restored.replay.data[key][:64],pilot.replay.data[key][:64])
    for key,value in pilot.agent.q1.state_dict().items():
        torch.testing.assert_close(restored.agent.q1.state_dict()[key],value)
    state=torch.load(checkpoint,weights_only=True)
    assert state['frozen_warm_start']['format_version']==1
    assert state['goal_contract']['old_Q_or_replay_imported'] is False
    # The last update in a two-critic tick is critic-only. Do not lose the
    # preceding real actor diagnostic or mistake its zero for actor inactivity.
    restored.warmup=2  # shortened unit-test warmup, never physical training data
    for index in range(64,66):
        _,previous=restored.act(raw,critic,index)
        restored.observe(previous,raw,critic,torch.ones(1),torch.zeros(1,dtype=torch.bool),index)
    assert restored.actor_updates==1 and not restored.latest['actor_updated']
    assert restored.latest_actor['actor_updated'] and restored.latest_actor['critic_update']==5
    assert torch.isfinite(torch.tensor(restored.latest_actor['actor_loss']))
    normalized=StagedGoalSACPilot(warm,contract,tmp_path/'normalized',stage,
        normalize_prior_loss_by_radius=True,anchor_prior_to_initial_policy=True)
    normalized.critic_updates=normalized.warmup  # unit fixture skips2048 gradient calls
    previous=(pilot.replay.data['actor_obs'][:64],pilot.replay.data['critic_obs'][:64],
              pilot.replay.data['action'][:64])
    for index in range(2):
        normalized.observe(previous,raw.expand(64,-1),critic.expand(64,-1),
            torch.ones(64),torch.zeros(64,dtype=torch.bool),index)
        if index==0:assert normalized.latest_actor['teacher_bc_loss']==pytest.approx(0.,abs=1e-12)
    assert normalized.latest_actor['teacher_bc_weight']==pytest.approx(800.)
    assert normalized.contract['normalize_prior_loss_by_radius']
    normalized.directory.mkdir();normalized.save(final=True)
    restored_normalized=StagedGoalSACPilot(warm,contract,tmp_path/'normalized_resume',stage,
        checkpoint=next(normalized.directory.glob('checkpoint_*.pt')))
    assert restored_normalized.normalize_prior_loss_by_radius
    assert restored_normalized.critic_updates==normalized.critic_updates
    assert restored_normalized.contract['actor_prior_source']=='frozen_validated_remaining_goal_actor'
    assert not any(p.requires_grad for p in restored_normalized.frozen_actor_prior.parameters())
    for key,value in normalized.frozen_actor_prior.state_dict().items():
        assert torch.equal(value,restored_normalized.frozen_actor_prior.state_dict()[key])

    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_hybrid_goal_sac import StagedHybridGoalSACPilot
    confident=StagedHybridGoalSACPilot(warm,contract,tmp_path/'confident',stage,
        free_grippers=True,gripper_logit_scale=.005,normalize_prior_loss_by_radius=True,
        anchor_prior_to_initial_policy=True,fixed_prior_radius=.05,
        validated_jaw_prior_confidence=.8,jaw_prior_residual_gain=20.)
    confident.actor_updates=20000
    assert confident.radius==pytest.approx(.05)
    command,observed=confident.act(raw,critic,0)
    confident.directory.mkdir();confident.save(final=True)
    restored_confident=StagedHybridGoalSACPilot(warm,contract,tmp_path/'confident_resume',stage,
        checkpoint=next(confident.directory.glob('checkpoint_*.pt')))
    assert restored_confident.radius==pytest.approx(.05)
    assert not any(p.requires_grad for p in restored_confident.frozen_actor_prior.parameters())
    torch.testing.assert_close(restored_confident.act(raw,critic,0)[0],command)

    # Behavior bias affects actual TRAIN commands only, survives resume as
    # configuration, and never changes greedy evaluation or Q action labels.
    from kuavo_isaaclab_scene.rl.multi_box.experiments.episode_arm_exploration import episode_arm_exploration_contract
    arm=StagedHybridGoalSACPilot(warm,contract,tmp_path/'arm_bias',stage,
        free_grippers=True,fixed_prior_radius=.05,exploration_correlation=.98,
        episode_arm_exploration=episode_arm_exploration_contract())
    measured=raw.clone();measured[:,96]=1
    arm.act(measured,critic,90)  # collecting warmup stays greedy
    assert arm.arm_behavior is None
    hybrid_fixture={k:v[:64].clone() for k,v in pilot.replay.data.items()}
    hybrid_fixture['action'][:,19:21]=-1  # unit fixture uses valid binary jaw labels
    arm.replay.add(**hybrid_fixture);arm.history.append(hybrid_fixture)
    arm.reset_exploration(2)
    physical,previous_arm=arm.act(measured,critic,90,exploration_ids=torch.tensor([1]))
    assert arm.arm_behavior.initialized.tolist()==[False,True]
    assert arm.report()['episode_arm_behavior']['body_bias_rms']>0
    torch.testing.assert_close(physical[:,list(GOAL_COLUMNS)],previous_arm[2])
    arm.observe(previous_arm,measured,critic,torch.ones(1),torch.zeros(1,dtype=torch.bool),90)
    torch.testing.assert_close(arm.replay.data['action'][64],previous_arm[2][0])
    arm.directory.mkdir();arm.save(final=True)
    arm_cp=next(arm.directory.glob('checkpoint_*.pt'))
    arm_resumed=StagedHybridGoalSACPilot(warm,contract,tmp_path/'arm_resume',stage,checkpoint=arm_cp)
    assert arm_resumed.episode_arm_exploration==episode_arm_exploration_contract()
    assert arm_resumed.replay.size==65
    torch.testing.assert_close(arm_resumed.replay.data['action'][:65],arm.replay.data['action'][:65])
    arm_eval=StagedHybridGoalSACPilot(warm,contract,tmp_path/'arm_eval',stage,checkpoint=arm_cp,training=False)
    arm_eval.reset_exploration(2);rng=torch.random.get_rng_state().clone()
    before_eval=(arm_eval.actor_updates,arm_eval.critic_updates,arm_eval.replay.size)
    actual_eval=arm_eval.act(measured,critic,90,exploration_ids=torch.tensor([1]))[1]
    torch.testing.assert_close(actual_eval[2],arm_eval.agent.act(actual_eval[0],True))
    assert torch.equal(torch.random.get_rng_state(),rng) and not arm_eval.arm_behavior.initialized.any()
    arm_eval.observe(actual_eval,measured,critic,torch.ones(1),torch.zeros(1,dtype=torch.bool),90)
    assert (arm_eval.actor_updates,arm_eval.critic_updates,arm_eval.replay.size)==before_eval

    # Actual-success retention is opt-in. Its newest closed bank belongs to
    # the replay snapshot even when an immutable checkpoint counter repeats.
    from test_rl_staged_train_success import episode
    retained=StagedHybridGoalSACPilot(warm,contract,tmp_path/'retained',stage,
        free_grippers=True,train_success_retention=True)
    successful,outcome=episode()
    retained.replay.add(**successful);retained.history.append(successful)
    retained.success_bank.add_episode(successful,outcome,source_run='unit_fixture',split='train')
    retained.directory.mkdir();retained.save(final=True)
    retained_cp=next(retained.directory.glob('checkpoint_*.pt'));unchanged=retained_cp.read_bytes()
    second,second_outcome=episode(env=1)
    retained.success_bank.add_episode(second,second_outcome,source_run='unit_fixture',split='train')
    retained.save(final=True)
    assert retained_cp.read_bytes()==unchanged
    continued=StagedHybridGoalSACPilot(warm,contract,tmp_path/'retained_resume',stage,checkpoint=retained_cp)
    assert continued.replay.size==3 and continued.success_bank.size==6
    assert continued.success_replay_fraction==pytest.approx(.2)
    evaluated=StagedHybridGoalSACPilot(warm,contract,tmp_path/'retained_eval',stage,checkpoint=retained_cp,training=False)
    before=(evaluated.actor_updates,evaluated.critic_updates,evaluated.replay.size,evaluated.success_bank.size)
    evaluated.act(raw,critic,0)
    assert (evaluated.actor_updates,evaluated.critic_updates,evaluated.replay.size,evaluated.success_bank.size)==before

    # A changed proof predicate starts new critics/replay. Only matching
    # behavior transfers, including its actor observation normalization.
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    fresh=StagedGoalSACPilot(warm,contract|dict(terminal_contract=grasp_lift_terminal_contract()),
        tmp_path/'new_proof',stage,normalize_prior_loss_by_radius=True,anchor_prior_to_initial_policy=True)
    q_before={k:v.clone() for k,v in fresh.agent.q1.state_dict().items()}
    prior_state=torch.load(next(normalized.directory.glob('checkpoint_*.pt')),weights_only=True)
    initialize_staged_actor_only(fresh,prior_state)
    assert fresh.actor_updates==fresh.critic_updates==fresh.replay.size==0
    assert all(not opt.state for opt in fresh.agent.optimizers)
    for key,value in q_before.items():assert torch.equal(value,fresh.agent.q1.state_dict()[key])
    for key,value in normalized.agent.actor.state_dict().items():
        assert torch.equal(value,fresh.agent.actor.state_dict()[key])
    changed=prior_state|dict(goal_contract=prior_state['goal_contract']|dict(
        physical_contract=prior_state['goal_contract']['physical_contract']|dict(self_collision={'enabled':True})))
    with pytest.raises(ValueError,match='coordinates, goals and safety'):initialize_staged_actor_only(fresh,changed)
    with pytest.raises(ValueError,match='fresh Q'):initialize_staged_actor_only(normalized,prior_state)

    collecting=StagedGoalSACPilot(warm,contract,tmp_path/'collecting',stage,
        normalize_prior_loss_by_radius=True,actor_min_replay_rows=128)
    collecting.critic_updates=collecting.warmup  # unit fixture, not a runtime migration
    collecting.observe(previous,raw.expand(64,-1),critic.expand(64,-1),
        torch.ones(64),torch.zeros(64,dtype=torch.bool),0)
    assert collecting.actor_updates==0 and collecting.critic_updates==2050
    assert collecting.radius==pytest.approx(.05) and collecting.prior_weight==pytest.approx(800.)
    assert collecting.report()['actor_collection_warmup_remaining']==64
    larger=StagedGoalSACPilot(warm,contract,tmp_path/'larger',stage,replay_capacity=24000)
    assert larger.replay.capacity==24000 and larger.contract['replay_capacity']==24000
    assert larger.replay.size==0 and larger.critic_updates==0
    larger.directory.mkdir();larger.save(final=True)
    empty=torch.load(larger.directory/'staged_goal_experience.pt',weights_only=True)
    assert (larger.directory/'staged_goal_experience.pt').stat().st_size<200000
    assert all(v.untyped_storage().nbytes()==0 for v in empty['executed_goal_transitions'].values())
    capped=StagedGoalSACPilot(warm,contract,tmp_path/'capped',stage,replay_capacity=1024)
    # Storage-only unit fixture; not collected physical training data.
    large_fixture={k:v[:64].repeat((32,*([1]*(v.ndim-1)))) for k,v in pilot.replay.data.items()}
    capped.history=[large_fixture];capped.directory.mkdir();capped.save(final=True)
    retained=torch.load(capped.directory/'staged_goal_experience.pt',weights_only=True)['executed_goal_transitions']
    for key,value in retained.items():
        assert len(value)==1024
        torch.testing.assert_close(value,large_fixture[key][-1024:])
        assert value.untyped_storage().nbytes()==value.numel()*value.element_size()


def test_staged_solver_identity_changes_only_solver_and_rejects_wrong_timing():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import staged_solver_contract,configure_staged_physics
    physics=SimpleNamespace(solver_type=1,min_position_iteration_count=4,min_velocity_iteration_count=2)
    cfg=SimpleNamespace(sim=SimpleNamespace(dt=1/120,physx=physics),decimation=4)
    configure_staged_physics(cfg,{})
    pgs=staged_solver_contract('PGS');configure_staged_physics(cfg,dict(physics_dynamics=pgs))
    assert physics.solver_type==0 and physics.min_position_iteration_count==4 and physics.min_velocity_iteration_count==2
    with pytest.raises(ValueError,match='Legacy'):configure_staged_physics(cfg,{})
    with pytest.raises(ValueError,match='dynamics'):configure_staged_physics(cfg,dict(physics_dynamics=pgs|dict(solver_type=1)))
    cfg.decimation=8
    with pytest.raises(ValueError,match='timestep'):configure_staged_physics(cfg,dict(physics_dynamics=pgs))
