"""Absolute goal labels must reproduce commands and stay separate from Q data."""
import json
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.layout_generalization import matrix6,yaw_matrix


def raw_state():
    raw=torch.zeros(2,464);raw[:,439]=1
    raw[:,71:77]=matrix6(yaw_matrix(.3,raw)).expand(2,-1)
    raw[:,68:71]=torch.tensor([.6,.2,.7])
    raw[:,:20]=torch.linspace(-.2,.2,20)
    raw[:,416:436]=.003
    return raw


def test_pose_prior_rejects_unknown_or_different_physical_travel():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
    from kuavo_isaaclab_scene.rl.multi_box.experiments.executed_replay import PHYSICAL_KEYS
    student=object.__new__(PoseStudent);student.state={}
    original={'action_contract':'s63_upright_torso_xz_fixed_pitch_v1'}
    higher=original|{'action_contract':'s63_upright_torso_xz_fixed_pitch_diagnostic_up_0.0600m'}
    student.validate_physical_contract(original)
    with pytest.raises(ValueError,match='Legacy pose prior'):
        student.validate_physical_contract(higher)
    contract={key:None for key in (*PHYSICAL_KEYS,'flap_pose_source')}|higher
    student.state={'physical_contract':contract.copy()}
    student.validate_physical_contract(contract)
    with pytest.raises(ValueError,match='action_contract'):
        student.validate_physical_contract(contract|original)
    with pytest.raises(ValueError,match='reward_profile'):
        student.validate_physical_contract(contract|{'reward_profile':{'changed':True}})


def test_absolute_goal_inverse_reproduces_executed_commands():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    generator=torch.Generator().manual_seed(5)
    physical=torch.rand(2,24,generator=generator)*2-1
    goals=coordinates.encode_physical(raw,physical)
    assert torch.allclose(coordinates.decode(raw,goals),physical,atol=1e-5)
    current=coordinates.encode_physical(raw,torch.zeros_like(physical))
    assert coordinates.decode(raw,current).abs().max()<1e-5


def test_projected_base_inverse_preserves_actual_tilted_base_commands():
    raw=raw_state()
    yaw=yaw_matrix(.5,raw)
    angle=raw.new_tensor(.12)
    pitch=raw.new_tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    pitch[0,0]=pitch[2,2]=angle.cos();pitch[0,2]=angle.sin();pitch[2,0]=-angle.sin()
    raw[:,71:77]=matrix6(yaw@pitch)
    physical=raw.new_zeros(2,24);physical[:,:3]=raw.new_tensor([.084,.32,-.2])
    legacy=PoseGoalCoordinates();exact=PoseGoalCoordinates(exact_projected_base=True)
    goals=exact.encode_physical(raw,physical)
    assert (legacy.decode(raw,goals)-physical).abs().max()>.001
    torch.testing.assert_close(exact.decode(raw,goals),physical,atol=1e-5,rtol=0)
    assert exact.name!=legacy.name
    assert torch.equal(exact.encode_physical(raw,physical),legacy.encode_physical(raw,physical))


def test_student_input_uses_observations_and_clock_without_reference_goals():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    inputs=coordinates.observations(raw,0)
    assert inputs.shape==(2,439) and inputs[:,-1].eq(0).all()
    assert coordinates.observations(raw,900)[:,-1].eq(1).all()
    other=raw.clone();other[:,86+6*22]=1;other[:,86+6*22+12]=.4
    assert not torch.equal(inputs,coordinates.observations(other,0))


def test_time_encoding_adds_only_clock_features():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    features=coordinates.observations(raw,205,16)
    assert features.shape==(2,471)
    assert torch.allclose(features[:,:439],coordinates.observations(raw,205))
    assert features[:,439:455].abs().max()<1e-5
    assert torch.allclose(features[:,455:471],torch.tensor([(-1.)**i for i in range(1,17)]).expand(2,-1))


def test_longer_demo_clock_distinguishes_late_states_and_preserves_legacy_inputs():
    coordinates=PoseGoalCoordinates();raw=raw_state()
    assert torch.equal(coordinates.observations(raw,450),coordinates.observations(raw,450,clock_horizon=410))
    assert not torch.equal(coordinates.observations(raw,450,clock_horizon=700),
                           coordinates.observations(raw,550,clock_horizon=700))
    assert coordinates.observations(raw,900,clock_horizon=700)[:,-1].eq(1).all()
    with pytest.raises(ValueError,match='clock horizon'):
        coordinates.observations(raw,0,clock_horizon=0)


def test_goal_projection_matches_physical_close_gate_and_entropy_mask():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
    from kuavo_isaaclab_scene.rl.multi_box.experiments.guided_exploration import GraspActionProjector
    coordinates=PoseGoalCoordinates();raw=raw_state()
    raw[:,386]=1 # left assigned flap0; other hand uses flap1
    raw[:,350]=.1;raw[:,377]=.3 # left near; right far
    physical=torch.zeros(2,24);physical[:,20:22]=1
    projected=GraspActionProjector([('body',20),('left_gripper',1),('right_gripper',1),('head',2)])(raw,physical)
    inputs=coordinates.observations(raw,0)
    z=physical.clone();z[:,22:24]=physical[:,20:22]
    goal_projector=GoalGripperProjector()
    assert torch.equal(goal_projector(inputs,z)[:,22:24],projected[:,20:22])
    assert torch.equal(goal_projector.entropy_mask(inputs)[:,22:24],torch.tensor([[1.,0.],[1.,0.]]))


def test_neural_prior_bounds_generated_goals_and_keeps_physical_gripper_gate():
    from types import SimpleNamespace
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import GoalGripperProjector
    coordinates=PoseGoalCoordinates();raw=raw_state()
    raw[:,386]=1;raw[:,350]=.1;raw[:,377]=.3
    obs=torch.cat((coordinates.observations(raw,0),torch.zeros(2,2)),-1)
    reference=torch.zeros(2,24);reference[:,22:]=1
    prior=SimpleNamespace(agent=SimpleNamespace(act=lambda x,deterministic:reference))
    projected=GoalGripperProjector(prior,.05)
    action=torch.full((2,24),.8,requires_grad=True)
    z=projected(obs,action)
    assert torch.equal(z[:,:22],torch.full((2,22),.05))
    assert torch.equal(z[:,22],torch.full((2,),.95))
    assert z[:,23].eq(-1).all() # Distant hand remains physically open.
    z.sum().backward();assert action.grad.eq(0).all()
    # Historical Q labels use only the physical gate, preserving actual goals.
    measured=GoalGripperProjector()(obs,action.detach())
    assert measured[:,:22].eq(.8).all()
    with pytest.raises(ValueError,match='frozen neural prior'):
        GoalGripperProjector(max_prior_deviation=.05)


def test_neural_prior_floor_is_independent_of_demo_replay_fade():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import PoseGoalSACPilot
    pilot=object.__new__(PoseGoalSACPilot)
    pilot.prior_weight=10.;pilot.prior_floor=10.;pilot.fade_updates=20000
    assert pilot.prior_weight_at(40000)==10.
    pilot.prior_floor=0.
    assert pilot.prior_weight_at(10000)==5. and pilot.prior_weight_at(40000)==0.


def test_anchor_context_does_not_change_warm_start_predictions():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import widen_bc_actor,GoalGripperProjector
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    config=SACConfig(hidden=16)
    original=AsymmetricSAC(439,531,24,config)
    original.actor_normalizer.update(torch.randn(10,439)*.4+1.2)
    state=original.checkpoint()|dict(artifact_type=PoseStudent.artifact_type,
        action_coordinates=PoseGoalCoordinates.name,goal_center=torch.zeros(24),goal_scale=torch.ones(24))
    student=PoseStudent(state)
    new=AsymmetricSAC(441,533,24,config,action_projector=GoalGripperProjector())
    widen_bc_actor(student,new)
    raw=PoseGoalCoordinates().observations(raw_state(),51)
    expanded=torch.cat((raw,torch.randn(2,2)),-1)
    a=original.actor(original.actor_normalizer(raw),deterministic=True)[0]
    b=new.actor(new.actor_normalizer(expanded),deterministic=True)[0]
    assert torch.allclose(a,b,atol=1e-7)
    assert torch.equal(new.actor_normalizer.count,original.actor_normalizer.count)


@pytest.mark.parametrize('artifact',['pose_goal_student_BC_diagnostic_NOT_SAC','pose_goal_sac_no_live_reference'])
def test_ordinary_delta_trainer_rejects_pose_goal_artifacts(tmp_path,artifact):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint
    (tmp_path/'manifest.json').write_text(json.dumps({'artifact_type':artifact}))
    with pytest.raises(ValueError,match='separate absolute action coordinates'):
        _compatible_checkpoint(tmp_path/'checkpoint.pt',{},data_only=True)


def test_pose_goal_can_explore_below_the_legacy_gaussian_floor():
    import math
    from kuavo_isaaclab_scene.rl.algorithms.sac import SquashedActor
    low=SquashedActor(3,2,16,max_std=.003,min_std=.0001)
    legacy=SquashedActor(3,2,16)
    with torch.no_grad():
        for actor in [low,legacy]:
            for p in actor.parameters():p.zero_()
            actor.network[-1].bias[2:].fill_(-8.)
        x=torch.zeros(8192,3)
        small=low(x)[0].std(0)
        original=legacy(x)[0].std(0)
    assert torch.allclose(small,torch.full((2,),math.exp(-8)),rtol=.04)
    assert torch.allclose(original,torch.full((2,),math.exp(-5)),rtol=.04)


def test_exploration_fork_preserves_mean_Q_and_actual_experience(tmp_path):
    import importlib.util
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig

    path=Path(__file__).parents[1]/'scripts/rl/fork_pose_goal_sac.py'
    spec=importlib.util.spec_from_file_location('goal_exploration_fork',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'source';source.mkdir()
    agent=AsymmetricSAC(441,533,24,SACConfig(hidden=16,initial_policy_std=.01,max_policy_std=.02))
    state=agent.checkpoint()|dict(artifact_type='pose_goal_sac_no_live_reference',
        actor_updates=2068,critic_updates=2568,format_version=1,
        goal_contract=dict(action_coordinates='same_goals',demo_fade_updates=4000))
    torch.save(state,source/'checkpoint.pt')
    (source/'manifest.json').write_text('{}')
    measured=dict(actor_obs=torch.randn(3,441),action=torch.randn(3,24),reward=torch.randn(3,1))
    torch.save(dict(goal_contract=state['goal_contract'],executed_goal_transitions=measured,
                    collection_policy_config={'gamma':.95}),
               source/'pose_goal_experience.pt')
    destination,audit=module.fork_checkpoint(source/'checkpoint.pt',tmp_path/'fork')
    forked=torch.load(destination,weights_only=True)
    copy=AsymmetricSAC(441,533,24,SACConfig(**forked['config']))
    copy.restore(forked)
    x=torch.randn(10,441)
    assert torch.equal(agent.act(x,deterministic=True),copy.act(x,deterministic=True))
    for key,value in state['model'].items():
        if key=='actor.network.4.weight':value=value[:24];other=forked['model'][key][:24]
        elif key=='actor.network.4.bias':value=value[:24];other=forked['model'][key][:24]
        else:other=forked['model'][key]
        assert torch.equal(value,other),key
    data=torch.load(destination.parent/'pose_goal_experience.pt',weights_only=True)
    assert all(torch.equal(value,data['executed_goal_transitions'][key]) for key,value in measured.items())
    assert data['collection_goal_contract']==state['goal_contract']
    assert data['collection_policy_config']=={'gamma':.95}
    assert data['goal_contract']['demo_fade_updates']==20000
    assert audit['deterministic_mean_unchanged'] and copy.config.max_policy_std==.003


def test_goal_exploration_fork_rejects_another_action_contract(tmp_path):
    import importlib.util
    from pathlib import Path
    spec=importlib.util.spec_from_file_location('goal_exploration_fork',
        Path(__file__).parents[1]/'scripts/rl/fork_pose_goal_sac.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checkpoint=tmp_path/'ordinary.pt';torch.save({'format_version':1},checkpoint)
    with pytest.raises(ValueError,match='native goal-SAC'):
        module.fork_checkpoint(checkpoint,tmp_path/'fork')
    assert not (tmp_path/'fork').exists()


def test_goal_discount_matches_potential_and_explicit_migration_keeps_actor(tmp_path,monkeypatch):
    import importlib.util
    from pathlib import Path
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
    from kuavo_isaaclab_scene.rl.multi_box.experiments import pose_goal_sac as goal

    torch.set_num_threads(1)
    contract=dict(discount=.999,reward_profile=dict(weights=dict(discount=.999)),
                  action_contract='s63_upright_torso_xz_fixed_pitch_v1')
    raw=raw_state()[:1].clone()
    raw[:,86+4*22]=1;raw[:,86+4*22+12:86+4*22+15]=torch.tensor([.8,.2,1.])
    raw[:,388+4]=1;raw[:,400+4]=1
    physical=torch.zeros(1,24);physical[:,20:22]=-1
    coordinates=PoseGoalCoordinates()
    center=coordinates.encode_physical(raw,physical)[0]
    center[19:21]-=coordinates.box_anchor(raw)[0];center[22:24]=0
    prior=AsymmetricSAC(439,531,24,SACConfig(hidden=16))
    state=prior.checkpoint()|dict(artifact_type=PoseStudent.artifact_type,
        action_coordinates=coordinates.name,goal_center=center,goal_scale=torch.ones(24),
        initial_box_relative_goals=True)
    torch.save(state,tmp_path/'bc.pt')
    measured=dict(actor_obs=raw,critic_obs=torch.cat((raw,torch.zeros(1,66)),1),
        action=physical,next_actor_obs=raw,next_critic_obs=torch.cat((raw,torch.zeros(1,66)),1),
        reward=torch.tensor([5.]),terminated=torch.tensor([True]))
    monkeypatch.setattr(goal,'read_executed_successes',lambda *args:(measured,
        dict(successful_episodes=1,source_dataset_sha256='physical-test-fixture')))
    pilot=goal.PoseGoalSACPilot(tmp_path/'bc.pt',None,contract,tmp_path/'initial',training=False)
    assert pilot.agent.config.gamma==.999 and not pilot.report()['discount_mismatch']
    assert pilot.fade_updates==20000 and pilot.agent.config.max_policy_std==.003
    old=tmp_path/'old';old.mkdir()
    pilot.agent.config.gamma=.99
    legacy=pilot.agent.checkpoint()|dict(artifact_type=pilot.artifact_type,
        format_version=1,goal_contract=pilot.contract,bc_prior=state,actor_updates=11,critic_updates=17)
    torch.save(legacy,old/'checkpoint.pt');(old/'manifest.json').write_text(json.dumps(contract))
    actual={key:value.repeat(4,*([1]*(value.ndim-1))) for key,value in pilot.seed.items()}
    torch.save(dict(goal_contract=pilot.contract,executed_goal_transitions=actual),old/'pose_goal_experience.pt')
    spec=importlib.util.spec_from_file_location('discount_fork',Path(__file__).parents[1]/'scripts/rl/fork_pose_goal_sac.py')
    fork=importlib.util.module_from_spec(spec);spec.loader.exec_module(fork)
    manifest=tmp_path/'physical.json';manifest.write_text(json.dumps(contract))
    path,audit=fork.fork_checkpoint(old/'checkpoint.pt',tmp_path/'aligned',
        align_discount_with=manifest,preserve_exploration=True,critic_warmup_updates=5)
    migrated=goal.PoseGoalSACPilot(path,None,contract,tmp_path/'trained',training=True)
    assert migrated.agent.config.gamma==.999 and migrated.discount_warmup_updates==5
    assert migrated.actor_updates==11 and migrated.critic_updates==22
    for key,value in pilot.agent.actor.state_dict().items():
        assert torch.equal(value,migrated.agent.actor.state_dict()[key]),key
    assert torch.equal(migrated.replay.data['reward'][:4],actual['reward'])
    assert audit['discount_alignment']['recorded_rewards_unchanged']
    migrated.directory.mkdir();migrated.save(final=True)
    saved=torch.load(next(migrated.directory.glob('checkpoint_*.pt')),weights_only=True)
    assert 'pending_discount_critic_warmup' not in saved
    reloaded=goal.PoseGoalSACPilot(next(migrated.directory.glob('checkpoint_*.pt')),
        None,contract,tmp_path/'next',training=True)
    assert reloaded.discount_warmup_updates==0 and reloaded.critic_updates==22


def test_goal_discount_rejects_a_different_potential_horizon():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_goal_sac import reward_discount
    with pytest.raises(ValueError,match='discounts differ'):
        reward_discount(dict(discount=.999,reward_profile=dict(weights=dict(discount=.99))))


def test_multiple_actual_seeds_reset_their_clock_and_initial_box_anchor(tmp_path,monkeypatch):
    from kuavo_isaaclab_scene.rl.multi_box.experiments import pose_goal_sac as goal
    from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseStudent
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    raw=raw_state()[:1].repeat(4,1)
    raw[:,86+4*22]=1;raw[:,86+4*22+12:86+4*22+15]=torch.tensor([.8,.2,1.])
    raw[2:,86+4*22+12]+=.15;raw[:,388+4]=1;raw[:,400+4]=1
    physical=torch.zeros(4,24);physical[:,20:22]=-1
    measured=dict(actor_obs=raw,critic_obs=torch.cat((raw,torch.zeros(4,66)),1),
        action=physical,next_actor_obs=raw,next_critic_obs=torch.cat((raw,torch.zeros(4,66)),1),
        reward=torch.tensor([0.,5.,0.,5.]),terminated=torch.tensor([False,True,False,True]))
    monkeypatch.setattr(goal,'merge_executed_successes',lambda *args:(measured,
        dict(successful_episodes=2,source_dataset_sha256='two-actual-episode-test-fixture')))
    prior=AsymmetricSAC(439,531,24,SACConfig(hidden=16))
    state=prior.checkpoint()|dict(artifact_type=PoseStudent.artifact_type,
        action_coordinates=PoseGoalCoordinates.name,goal_center=torch.zeros(24),goal_scale=torch.ones(24),
        initial_box_relative_goals=True)
    torch.save(state,tmp_path/'bc.pt')
    pilot=goal.PoseGoalSACPilot(tmp_path/'bc.pt',['one.hdf5','two.hdf5'],
        dict(discount=.999,action_contract='s63_upright_torso_xz_fixed_pitch_v1'),tmp_path/'run',training=False)
    seed=pilot.seed['actor_obs']
    torch.testing.assert_close(seed[:,438],torch.tensor([0.,1/410,0.,1/410]))
    assert torch.equal(seed[0,-2:],seed[1,-2:]) and torch.equal(seed[2,-2:],seed[3,-2:])
    assert not torch.equal(seed[0,-2:],seed[2,-2:])
    assert pilot.report()['seed_episodes']==2
    assert pilot.contract['seed_episode_context']=='separate_clock_and_initial_box_anchor'
    assert torch.equal(pilot.seed['reward'],measured['reward'])


def test_actor_recovery_preserves_latest_critic_real_replay_and_fade_clock(tmp_path):
    import importlib.util
    from pathlib import Path
    from copy import deepcopy
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    spec=importlib.util.spec_from_file_location('recover_actor',Path(__file__).parents[1]/'scripts/rl/recover_pose_goal_actor.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    agent=AsymmetricSAC(441,533,24,SACConfig(hidden=16))
    state=agent.checkpoint()|dict(artifact_type='pose_goal_sac_no_live_reference',
        goal_contract={'same_physical_contract':True},actor_updates=40000,critic_updates=40500)
    current=tmp_path/'current';current.mkdir();best=tmp_path/'best';best.mkdir()
    torch.save(state,best/'checkpoint.pt');(current/'manifest.json').write_text('{}')
    latest=deepcopy(state)
    for key,value in latest['model'].items():
        if key.startswith(('actor.','q1.','q2.')):value.add_(.5)
    torch.save(latest,current/'checkpoint.pt')
    measured={'action':torch.tensor([[.7]*24]),'reward':torch.tensor([-6.])}
    torch.save(dict(goal_contract=state['goal_contract'],executed_goal_transitions=measured),current/'pose_goal_experience.pt')
    path,audit=module.recover(current/'checkpoint.pt',best/'checkpoint.pt',tmp_path/'recovery')
    recovered=torch.load(path,weights_only=True)
    for key,value in recovered['model'].items():
        expected=state['model'][key] if key.startswith('actor.') else latest['model'][key]
        assert torch.equal(value,expected),key
    rows=torch.load(path.parent/'pose_goal_experience.pt',weights_only=True)['executed_goal_transitions']
    assert all(torch.equal(value,rows[key]) for key,value in measured.items())
    assert recovered['actor_updates']==40000 and recovered['critic_updates']==40500
    assert audit['latest_critic_optimizer_and_replay_preserved']
