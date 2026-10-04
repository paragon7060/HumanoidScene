"""Synthetic control/learning fixtures; none is evidence of physical grasp success."""
from copy import deepcopy
from types import SimpleNamespace
import json

import h5py

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
from kuavo_isaaclab_scene.rl.algorithms.hybrid_goal_sac import HybridGoalSAC
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.pose_student import PoseGoalCoordinates,pose_clock
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_actions import (
    PHYSICAL_COLUMNS,PhysicalBodyProjector,actor_only_snapshot,servo_raw_actor,body_command,full_command,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_body_sac import PhysicalBodySACPilot,physical_signature,PhysicalTrainSuccessBank
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import StagedGoalProjector
from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import grasp_lift_terminal_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.physical_native_seed import seed_physical_training_successes
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class,staged_policy_metadata


def setup_fixture(tmp_path,*,gain=.5):
    config=SACConfig(hidden=16,gamma=.999,freeze_actor_normalizer=True,
                     initial_policy_std=.005,min_policy_std=.001,max_policy_std=.005)
    coordinates=PoseGoalCoordinates()
    raw=torch.zeros(1,464);raw[:,71:77]=torch.tensor([1.,0.,0.,0.,1.,0.])
    raw[:,86]=1;raw[:,96]=1;raw[:,386]=1;raw[:,388]=1;raw[:,400]=1;raw[:,439]=1
    prior=SimpleNamespace(agent=AsymmetricSAC(472,531,24,config),state={})
    def observations(raw,critic,index,anchor):
        features=coordinates.observations(raw,index,16,410,condition_on_shelf=True)
        clock=pose_clock(raw,index,410)
        return torch.cat((features,anchor.expand(len(raw),-1)),-1),torch.cat((critic,clock,anchor.expand(len(raw),-1)),-1)
    warm=SimpleNamespace(training=False,agent=AsymmetricSAC(474,533,24,config),
        prior=prior,coordinates=coordinates,observations=observations)
    physical=dict(discount=.999,reward_profile=dict(weights=dict(discount=.999)),
                  terminal_contract=grasp_lift_terminal_contract())
    stage=SimpleNamespace(phase='held_grasp',manipulation_start=0,target_xy=torch.tensor([[.1,.7]]),
        target_yaw=.2,name='synthetic_waypoints',templates={'upper':'synthetic_not_a_success'})
    source=HybridGoalSAC(480,539,21,config,action_projector=StagedGoalProjector(prior,472,True,.05))
    state=source.checkpoint()|dict(artifact_type='staged_base_hold_remaining_hybrid_sac_v1',
        actor_updates=42,critic_updates=99,frozen_warm_start=dict(synthetic_fixture=True),
        goal_contract=dict(actor_dim=480,critic_dim=539,fixed_prior_radius=.05,
            goal_center=[0.]*21,goal_scale=[1.]*21,physical_contract=physical_signature(physical),
            shelf_templates=stage.templates,waypoint_format=stage.name))
    pilot=PhysicalBodySACPilot(warm,physical,tmp_path/'source',stage,frozen_goal_state=state,residual_gain=gain)
    pilot.anchor=coordinates.box_anchor(raw).clone()
    return pilot,warm,physical,stage,raw,state


def test_actual_physical_projection_has_gradients_and_independent_close_gate():
    def baseline(obs):return obs.new_full((len(obs),21),.9)
    projector=PhysicalBodyProjector(baseline)
    obs=torch.zeros(2,480);obs[:,144]=1;obs[1,108:144]=1
    requested=torch.full((2,21),-.2,requires_grad=True)
    actual=projector(obs,requested)
    torch.testing.assert_close(actual[:,:19],torch.full((2,19),.8))
    assert actual[1,19:].eq(-1).all()
    actual[:,:19].sum().backward()
    torch.testing.assert_close(requested.grad[:,:19],torch.full((2,19),.5))


@pytest.mark.parametrize('gain',[.5,2.])
def test_frozen_replay_dispatch_restores_physical_commands_without_goal_labels(tmp_path,gain):
    pilot,warm,physical,stage,raw,_=setup_fixture(tmp_path,gain=gain)
    pilot.directory.mkdir(parents=True);pilot.save(final=True)
    checkpoint=next(pilot.directory.glob('checkpoint_*.pt'))
    cls=staged_policy_class(pilot.artifact_type)
    replay=cls(warm,physical,tmp_path/'frozen-video',stage,checkpoint=checkpoint,training=False)
    before=(replay.actor_updates,replay.critic_updates,replay.replay.size)
    action,previous=replay.act(raw,torch.zeros(1,530),0)
    torch.testing.assert_close(action[:,list(PHYSICAL_COLUMNS)],previous[2],atol=0,rtol=0)
    replay.observe(previous,raw,torch.zeros(1,530),torch.tensor([0.]),torch.tensor([False]),0)
    assert (replay.actor_updates,replay.critic_updates,replay.replay.size)==before
    assert list(staged_policy_metadata(replay))==['physical_body_contract']
    assert staged_policy_class('pose_goal_sac_no_live_reference') is None


def test_full_gain_can_represent_opposite_servo_commands_without_larger_local_noise(tmp_path):
    # Full endpoints are commands, not inverse requested-goal labels.
    def baseline(obs):
        result=obs.new_zeros(len(obs),21)
        result[:,:19]=torch.linspace(-1,1,19).to(obs)
        return result
    obs=torch.zeros(1,480);obs[:,144]=1
    projector=PhysicalBodyProjector(baseline,2.)
    for target in [-1.,1.]:
        requested=torch.zeros(1,21);requested[:,:19]=(target-baseline(obs)[:,:19])/2
        torch.testing.assert_close(projector(obs,requested)[:,:19],torch.full((1,19),target))
    half,_,_,_,_,_=setup_fixture(tmp_path/'half')
    full,warm,physical,stage,raw,_=setup_fixture(tmp_path/'full',gain=2.)
    assert full.radius==2. and full.agent.config.actor_lr==pytest.approx(half.agent.config.actor_lr/4)
    for name in ['initial_policy_std','min_policy_std','max_policy_std']:
        assert 2*getattr(full.agent.config,name)==pytest.approx(.5*getattr(half.agent.config,name))
    ao,_=full.observations(raw,torch.zeros(1,530),0)
    old=ao.clone();old[:,-1]=.05
    torch.testing.assert_close(full.agent.actor_normalizer(ao),full.command_prior.normalizer(old),atol=1e-6,rtol=0)
    full.directory.mkdir(parents=True);full.save(final=True)
    cp=next(full.directory.glob('checkpoint_*.pt'))
    resumed=PhysicalBodySACPilot(warm,physical,tmp_path/'resumed',stage,checkpoint=cp)
    assert resumed.radius==2. and resumed.contract==full.contract
    with pytest.raises(ValueError,match='gain differs'):
        PhysicalBodySACPilot(warm,physical,tmp_path/'wrong',stage,checkpoint=cp,residual_gain=.5)


@pytest.mark.parametrize('all_terminal',[False,True])
def test_terminal_controller_placeholders_are_never_decoded_for_Q_bootstrap(tmp_path,all_terminal):
    pilot,_,_,_,raw,_=setup_fixture(tmp_path)
    _,previous=pilot.act(raw,torch.zeros(1,530),0)
    ao,co,action=[v.expand(64,-1).clone() for v in previous]
    terminal=torch.ones(64,dtype=torch.bool) if all_terminal else torch.arange(64)%2==0
    na=ao.clone();na[terminal]=0  # These finite placeholders cannot be decoded.
    nc=co.clone();nc[terminal]=0
    report=pilot.agent.update(dict(actor_obs=ao,critic_obs=co,action=action,
        next_actor_obs=na,next_critic_obs=nc,reward=torch.ones(64),terminated=terminal))
    assert report['bootstrapped_rows']==int((~terminal).sum())
    assert report['actor_updated'] and all(torch.isfinite(torch.tensor(v)) for v in report.values())


def test_body_prior_does_not_solve_unused_singular_base_projection(tmp_path):
    pilot,_,_,_,raw,_=setup_fixture(tmp_path)
    pilot.coordinates.exact_projected_base=True
    raw[:,71:77]=0
    ao,_=pilot.observations(raw,torch.zeros(1,530),0)
    body=pilot.command_prior(ao)
    assert body.shape==(1,21) and body.isfinite().all()
    # The live full command retains the original unsafe-base rejection.
    with pytest.raises(ValueError,match='singular'):
        full_command(pilot.coordinates,ao,body)


def test_actor_only_prior_reconstructs_servo_and_never_copies_source_Q(tmp_path):
    pilot,_,_,_,raw,source=setup_fixture(tmp_path)
    ao,co=pilot.observations(raw,torch.zeros(1,530),0)
    copied=servo_raw_actor(ao)
    assert torch.equal(copied[:,:86],raw[:,:86]) and torch.equal(copied[:,412:440],raw[:,412:440])
    snapshot=actor_only_snapshot(source)
    assert set(snapshot['actor'])==set(pilot.agent.actor.state_dict())
    assert not any(k.startswith('q') for k in snapshot['actor'])
    assert not pilot.agent.q_optimizer.state and pilot.critic_updates==pilot.actor_updates==0
    assert any(not torch.equal(v,source['model']['q1.'+k]) for k,v in pilot.agent.q1.state_dict().items())
    old_obs=ao.clone();old_obs[:,-1]=.05
    torch.testing.assert_close(pilot.agent.actor_normalizer(ao),pilot.command_prior.normalizer(old_obs),atol=1e-6,rtol=0)
    command,previous=pilot.act(raw,torch.zeros(1,530),0)
    assert command.shape==(1,24) and command.isfinite().all()
    assert torch.equal(body_command(command),previous[2])
    with pytest.raises(ValueError,match='held phase'):servo_raw_actor(ao*0)
    bad=ao.clone();bad[:,173]=0
    with pytest.raises(ValueError,match='telemetry'):servo_raw_actor(bad)


def test_physical_replay_update_resume_and_frozen_eval_keep_literal_commands(tmp_path):
    pilot,warm,physical,stage,raw,_=setup_fixture(tmp_path)
    command,previous=pilot.act(raw,torch.zeros(1,530),0)
    prior_weights={k:v.clone() for k,v in pilot.command_prior.state_dict().items()}
    pilot.warmup=0  # Only this synthetic fixture shortens the1024-update warmup.
    pilot.observe(tuple(v.expand(64,-1) for v in previous),raw.expand(64,-1),torch.zeros(64,530),
        torch.ones(64),torch.zeros(64,dtype=torch.bool),0)
    assert pilot.replay.size==64 and pilot.critic_updates==2 and pilot.actor_updates==1
    torch.testing.assert_close(pilot.replay.data['action'][:64],body_command(command).expand(64,-1))
    assert pilot.latest_actor['actor_updated'] and torch.isfinite(torch.tensor(pilot.latest_actor['actor_loss']))
    for k,v in pilot.command_prior.state_dict().items():assert torch.equal(v,prior_weights[k])
    pilot.reset_exploration(64)
    sampled,previous=pilot.act(raw.expand(64,-1),torch.zeros(64,530),0)
    assert sampled.isfinite().all() and (previous[2][:,19:].abs()==1).all()
    assert torch.equal(body_command(sampled),previous[2])
    assert bool((previous[2][:,:19].std(0)>0).any())
    pilot.warmup=1024;pilot.directory.mkdir();pilot.save(final=True)
    checkpoint=next(pilot.directory.glob('checkpoint_*.pt'))
    saved=torch.load(checkpoint,weights_only=True)
    assert saved['algorithm']=='hybrid_physical_body_sac' and 'goal_contract' not in saved
    restored=PhysicalBodySACPilot(warm,physical,tmp_path/'restored',stage,checkpoint=checkpoint)
    assert restored.replay.size==64 and restored.critic_updates==2 and restored.actor_updates==1
    for k in pilot.replay.data:torch.testing.assert_close(restored.replay.data[k][:64],pilot.replay.data[k][:64])
    evaluated=PhysicalBodySACPilot(warm,physical,tmp_path/'eval',stage,checkpoint=checkpoint,training=False)
    rng=torch.random.get_rng_state().clone();before=(evaluated.actor_updates,evaluated.critic_updates,evaluated.replay.size)
    _,prev=evaluated.act(raw,torch.zeros(1,530),0)
    assert torch.equal(torch.random.get_rng_state(),rng)
    evaluated.observe(prev,raw,torch.zeros(1,530),torch.ones(1),torch.zeros(1,dtype=torch.bool),0)
    assert before==(evaluated.actor_updates,evaluated.critic_updates,evaluated.replay.size)
    wrong=deepcopy(saved);wrong['algorithm']='hybrid_goal_sac'
    with pytest.raises(ValueError,match='goal-action'):evaluated.agent.restore(wrong)


def test_physical_success_labels_and_projected_actor_loss_cannot_be_confused_with_goals(tmp_path):
    pilot,_,_,_,raw,_=setup_fixture(tmp_path)
    ao,_=pilot.observations(raw,torch.zeros(1,530),0)
    labels=pilot.command_prior(ao)
    requested=torch.zeros(1,19,requires_grad=True)
    loss=pilot.agent.success_body_loss(ao,requested,labels)
    assert loss.item()==pytest.approx(0.,abs=1e-12)
    goal_style_loss=torch.nn.functional.mse_loss(requested,labels[:,:19])
    assert goal_style_loss>.01  # Applying the old residual/goal loss would be wrong.
    bank=PhysicalTrainSuccessBank(480,539)
    with pytest.raises(ValueError,match='goal labels'):
        bank.restore(dict(format='actual_train_success_held_goal_transitions_v1'))
    broken=labels.clone();broken[:,19]=.4
    with pytest.raises(ValueError,match='binary jaws'):full_command(pilot.coordinates,ao,broken)


def native_fixture(tmp_path,mutation=None):
    """Synthetic recorded states exercise import guards, not physical success."""
    pilot,_,physical,_,raw,_=setup_fixture(tmp_path)
    raw=raw.expand(2,-1).clone()
    critic=torch.zeros(2,530);critic[:,:464]=raw
    next_critic=critic.clone()
    for offset in [35,36,41,42,43,44,49,50,51,52,53,54]:next_critic[-1,464+offset]=1
    ao,_=pilot.observations(raw,critic,torch.arange(2))
    actions=full_command(pilot.coordinates,ao,pilot.command_prior(ao))
    transitions=dict(actor_obs=raw,critic_obs=critic,next_actor_obs=raw.clone(),
        next_critic_obs=next_critic,action=actions,reward=torch.tensor([-1.,1.]),
        terminated=torch.tensor([False,True]),success=torch.tensor([False,True]),unsafe=torch.zeros(2,dtype=torch.bool))
    layout=dict(seed=123,target_region='shelf_3_right',split='train')
    outcome=dict(wave=1,environment=0,split='train',layout=layout,complete=True,
        initial_layout_valid=True,executed_transition_rows=2,
        result=dict(success=True,unsafe=False,pinching=[True,True],stable_hands=[True,True],
            opposing_flaps=True,proof_lift=True,hold_time_s=.2667,rack_clearance_m=.01,
            staged_base=dict(phase='held_grasp',manipulation_start=0,base_target_xy_rack_m=[.1,.7],
                base_target_yaw_rack_rad=.2)))
    meta=dict(training_contract=physical,collection_source='staged_base_hold_remaining_hybrid_sac_v1',
        episode_layouts=[dict(wave=1,environment=0,layout=deepcopy(layout))])
    if mutation:mutation(meta,outcome,transitions)
    dataset=tmp_path/'synthetic_native.hdf5'
    with h5py.File(dataset,'w') as f:
        f.attrs['manifest_json']=json.dumps(meta)
        episode=f.create_group('episodes/episode_0')
        episode.attrs.update(wave=1,environment=0,success=True,initial_layout_guard_valid=True,
            layout_json=json.dumps(outcome['layout']))
        for key,value in transitions.items():episode.create_dataset('transitions/'+key,data=value.numpy())
    return pilot,dataset,outcome,transitions


def test_native_import_keeps_literal_commands_and_rejects_duplicate_before_mutation(tmp_path):
    pilot,dataset,outcome,transitions=native_fixture(tmp_path)
    provenance=seed_physical_training_successes(pilot,dataset,[outcome],source_run='synthetic_fixture')
    assert pilot.replay.size==pilot.success_bank.size==2
    assert pilot.actor_updates==pilot.critic_updates==0 and not provenance['requested_goals_inverted']
    torch.testing.assert_close(pilot.replay.data['action'][:2],body_command(transitions['action']),rtol=0,atol=0)
    with pytest.raises(ValueError,match='Duplicate'):
        seed_physical_training_successes(pilot,dataset,[outcome],source_run='synthetic_fixture')
    assert pilot.replay.size==pilot.success_bank.size==2


@pytest.mark.parametrize('mutation',[
    lambda m,o,t:o.update(split='validation'),
    lambda m,o,t:m['episode_layouts'][0]['layout'].update(split='holdout'),
    lambda m,o,t:m.update(centered_world_probe=True),
    lambda m,o,t:t['action'].__setitem__((0,0),.7),
    lambda m,o,t:t['next_critic_obs'].__setitem__((-1,507),0),
    lambda m,o,t:t['next_critic_obs'].__setitem__((0,523),1),
])
def test_native_invalid_paths_never_mutate_physical_replay_or_stage(tmp_path,mutation):
    pilot,dataset,outcome,_=native_fixture(tmp_path,mutation)
    stage,anchor=pilot.stage,pilot.anchor
    with pytest.raises(ValueError):
        seed_physical_training_successes(pilot,dataset,[outcome],source_run='synthetic_fixture')
    assert pilot.replay.size==pilot.success_bank.size==0 and pilot.stage is stage and pilot.anchor is anchor
