"""Separate dynamics identity and measured row transfer for CPU PhysX SAC."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest
import torch

from batched_staged_goal_with_drive import validate_managed_physics_device
from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_physics_training import (
    validate_cpu_physics_training,act_measured_held_rows)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
    CPU_PHYSICS_BACKEND,staged_solver_contract,configure_staged_physics)


def inputs():
    def wave(split,start):
        return dict(split=split,layouts=[dict(episode_index=i%2,layout=dict(seed=start+i,
            split='train' if split=='train' else 'holdout',target_region=REGIONS[i%4])) for i in range(128)])
    dev=wave('validation',900000)
    return [dev,wave('train',1000000),deepcopy(dev),wave('train',1000200)],dict(
        physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))


def validate(waves,contract,**kwargs):
    return validate_cpu_physics_training(waves,contract,enabled=True,**(
        dict(physics_device='cpu',learner_device='cuda:0',training=True,steps=900)|kwargs))


def test_explicit_CPU_MDP_preserves_original_requests_and_disjoint_TRAIN():
    waves,contract=inputs();before=deepcopy((waves,contract));audit=validate(waves,contract)
    assert audit['fresh_TRAIN_requested']==256 and audit['requested_per_region']==32
    assert not audit['source_GPU_Q_replay_import_eligible'] and not audit['independent_FINAL_used']
    assert (waves,contract)==before
    assert validate_cpu_physics_training([],{},enabled=False,physics_device='cuda:0',
        learner_device='cuda:0',training=True,steps=10) is None


@pytest.mark.parametrize('kwargs',[
    {'physics_device':'cuda:0'},{'learner_device':'cpu'},{'training':False},
    {'steps':1},{'other_probe':True},
])
def test_CPU_training_refuses_implicit_frozen_or_other_probe(kwargs):
    with pytest.raises(ValueError,match='explicit CPU simulation'):validate(*inputs(),**kwargs)


@pytest.mark.parametrize('mutation',[
    lambda w:w[1].update(split='holdout'),
    lambda w:w[1].update(background_variant='packed'),
    lambda w:w[1]['layouts'].pop(),
    lambda w:w[1]['layouts'][0]['layout'].update(seed=True),
    lambda w:w[1]['layouts'][0]['layout'].update(split='holdout'),
    lambda w:w[1]['layouts'][0]['layout'].update(target_region=REGIONS[1]),
    lambda w:w[3].update(layouts=deepcopy(w[1]['layouts'])),
    lambda w:w[1]['layouts'][0]['layout'].update(seed=w[0]['layouts'][0]['layout']['seed']),
    lambda w:w[2]['layouts'][0]['layout'].update(seed=9000999),
])
def test_CPU_training_refuses_changed_denominators_FINAL_and_seed_leakage(mutation):
    waves,contract=inputs();mutation(waves)
    with pytest.raises(ValueError):validate(waves,contract)


def test_GPU_Q_contract_cannot_be_used_for_CPU_learning_and_CPU_cannot_run_on_GPU():
    waves,_=inputs()
    with pytest.raises(ValueError,match='own PGS'):validate(waves,dict(physics_dynamics=staged_solver_contract('PGS')))
    contract=inputs()[1]
    cfg=SimpleNamespace(sim=SimpleNamespace(device='cpu',dt=1/120,physx=SimpleNamespace(solver_type=1)),decimation=4)
    configure_staged_physics(cfg,contract);assert cfg.sim.physx.solver_type==0
    cfg.sim.device='cuda:0'
    with pytest.raises(ValueError,match='actual CPU simulation'):configure_staged_physics(cfg,contract)
    assert staged_solver_contract('PGS')==dict(solver='PGS',solver_type=0,physics_dt_s=1/120,control_dt_s=1/30)


def test_only_reviewed_CPU_backend_is_ignored_for_frozen_actor_matching():
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import frozen_cpu_actor_contract
    source=dict(physics_dynamics=staged_solver_contract('PGS'),actions={'unchanged':24},terminal_contract={'rack':10})
    CPU=source|dict(physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))
    before=deepcopy(CPU);assert frozen_cpu_actor_contract(CPU)==source and CPU==before
    for dynamics in (CPU['physics_dynamics']|dict(physics_backend='unknown'),
                     CPU['physics_dynamics']|dict(physics_dt_s=.01),
                     staged_solver_contract('TGS',physics_backend=CPU_PHYSICS_BACKEND)):
        with pytest.raises(ValueError,match='cannot bypass'):frozen_cpu_actor_contract(source|dict(physics_dynamics=dynamics))


def test_manager_validates_CPU_contract_and_refuses_probes_before_runtime(tmp_path):
    waves,contract=inputs();wp=tmp_path/'waves.json';mp=tmp_path/'manifest.json'
    wp.write_text(json.dumps(waves));mp.write_text(json.dumps(contract))
    flags=['--cpu-physics-training','--training','--waves-json',str(wp),'--training-manifest',str(mp)]
    assert validate_managed_physics_device('cpu',flags,learner_device='cuda:0')['training']
    with pytest.raises(ValueError):validate_managed_physics_device('cpu',flags,learner_device='cpu')
    for flag in ('--reset-world-frame-probe=x.json','--frozen-physics-backend-eval','--base-waypoint-probe'):
        with pytest.raises(ValueError):validate_managed_physics_device('cpu',flags+[flag],learner_device='cuda:0')


@pytest.mark.parametrize('learner_device',['cpu','cuda:0'])
def test_measured_transfer_retains_ids_terminal_context_actions_and_quarantine(learner_device):
    if learner_device=='cuda:0' and not torch.cuda.is_available():pytest.skip('GPU integration checked separately with isolation')
    from kuavo_isaaclab_scene.rl.multi_box.experiments.batched_staged_goal import observe_measured_held_rows
    class Stages:
        anchors=torch.tensor([[10.,11.],[20.,21.],[30.,31.],[40.,41.]])
        def held_context(self,ids):
            return SimpleNamespace(phase='held_grasp',manipulation_start=True,
                target_xy=self.anchors[ids]+1,target_yaw=ids.float()+.1)
    class Pilot:
        device=learner_device
        supplemental_observation_dim=38
        def act(self,raw,critic,clocks,*,exploration_ids,supplemental):
            assert all(str(v.device)==self.device for v in
                (raw,critic,clocks,exploration_ids,supplemental,self.anchor,self.stage.target_xy,self.stage.target_yaw))
            self.acted=(raw.clone(),critic.clone(),clocks.clone(),exploration_ids.clone())
            command=torch.cat((raw[:,:1],supplemental[:,:1]),-1)
            return command,(raw,critic,command)
        def observe(self,previous,raw,critic,reward,terminated,clock,*,supplemental):
            assert all(str(v.device)==self.device for v in
                (*previous,raw,critic,reward,terminated,clock,supplemental,self.anchor,self.stage.target_xy))
            self.observed=(previous,raw,critic,reward,terminated,clock,supplemental)
    p=Pilot();stages=Stages();ids=torch.tensor([0,2,3]);clocks=torch.tensor([4,8,12])
    observation=dict(policy=torch.arange(8.).reshape(4,2),critic=torch.arange(4.)[:,None],
        actual_flap_relations=torch.arange(152.).reshape(4,38))
    command,previous=act_measured_held_rows(p,stages,ids,observation,clocks,supplemental_group='actual_flap_relations')
    assert command.device.type=='cpu'
    torch.testing.assert_close(command[:,0],observation['policy'][ids,0])
    terminal={k:v+100 for k,v in observation.items()}
    from kuavo_isaaclab_scene.rl.multi_box.observations.flap_supplement import SUPPLEMENTAL_GROUP
    terminal[SUPPLEMENTAL_GROUP]=terminal.pop('actual_flap_relations')
    added=observe_measured_held_rows(p,stages,ids,previous,terminal,torch.arange(4.),
        torch.tensor([False,True,True,False]),clocks,torch.tensor([True,False,False,True]))
    assert added==2
    prev,raw,critic,reward,terminated,clock,supplemental=p.observed
    torch.testing.assert_close(prev[0].cpu(),observation['policy'][[0,3]])
    torch.testing.assert_close(raw.cpu(),terminal['policy'][[0,3]])
    assert reward.cpu().tolist()==[0.,3.] and clock.cpu().tolist()==[4,12]
    torch.testing.assert_close(p.anchor.cpu(),stages.anchors[[0,3]])
    torch.testing.assert_close(p.stage.target_xy.cpu(),stages.anchors[[0,3]]+1)


def test_actor_only_CPU_migration_keeps_actual_actor_and_refuses_GPU_Q_resume(tmp_path):
    from test_rl_actual_flap_residual_sac import pilots
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    _,old,warm,physical,stage,_=pilots(tmp_path)
    old.physical_contract['physics_dynamics']=staged_solver_contract('PGS')
    old.body_anchor_state['source_goal_contract']['physical_contract']['physics_dynamics']=staged_solver_contract('PGS')
    old.directory.mkdir();old.save(final=True);cp=next(old.directory.glob('checkpoint_*.pt'))
    source=torch.load(cp,weights_only=True)
    current=physical|dict(physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))
    new=ActualFlapResidualSACPilot(warm,current,tmp_path/'CPU_fresh',stage,
        body_anchor_state=source['body_anchor_state'],replay_capacity=1024,exploration_correlation=.99,
        train_success_retention=True)
    before={k:v.clone() for k,v in new.agent.state_dict().items() if k.startswith(('q','target','critic_normalizer'))}
    report=initialize_staged_actor_only(new,source)
    assert report['actor_only'] and not report['optimizer_states_imported']
    for k,v in before.items():assert torch.equal(v,new.agent.state_dict()[k])
    for k,v in source['model'].items():
        if k.startswith(('actor.','actor_normalizer.')):assert torch.equal(v,new.agent.state_dict()[k])
    assert new.actor_updates==new.critic_updates==new.replay.size==new.success_bank.size==0
    assert not any(opt.state for opt in new.agent.optimizers) and not new.agent.critic_normalizer.count
    with pytest.raises(ValueError,match='same phase/waypoint/remaining-goal contract'):
        ActualFlapResidualSACPilot(warm,current,tmp_path/'bad_CPU_resume',stage,checkpoint=cp,training=False)
    new.directory.mkdir();new.save(final=True);CPU_cp=next(new.directory.glob('checkpoint_*.pt'))
    restored=ActualFlapResidualSACPilot(warm,current,tmp_path/'CPU_resume',stage,checkpoint=CPU_cp,training=True)
    assert restored.replay.size==0 and restored.physical_contract['physics_dynamics']['physics_backend']==CPU_PHYSICS_BACKEND
    with pytest.raises(ValueError,match='same phase/waypoint/remaining-goal contract'):
        ActualFlapResidualSACPilot(warm,physical|dict(physics_dynamics=staged_solver_contract('PGS')),
            tmp_path/'CPU_Q_on_GPU',stage,checkpoint=CPU_cp,training=False)
