"""Retention fixtures test data separation, not simulated physical success."""
from copy import deepcopy
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import (
    TrainSuccessBank,REGIONS,add_completed_training_wave,retention_config,
    match_measured_success_paths,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.success_registration import CompletedTrainSuccessRegistration


def episode(region=REGIONS[0],n=3,env=0,wave=1):
    rows={k:torch.zeros(n,d) for k,d in [('actor_obs',480),('next_actor_obs',480),('critic_obs',539),('next_critic_obs',539),('action',21)]}
    for k in ['actor_obs','next_actor_obs','critic_obs','next_critic_obs']:rows[k][:,-6]=1
    rows['actor_obs'][:,94+REGIONS.index(region)]=1
    rows['action'][:,19:21]=-1
    rows['reward']=torch.arange(n,dtype=torch.float32);rows['terminated']=torch.zeros(n,dtype=torch.bool);rows['terminated'][-1]=True
    outcome=dict(wave=wave,split='train',environment=env,layout=dict(seed=100+env,target_region=region),
        complete=True,initial_layout_valid=True,result=dict(success=True,unsafe=False,invalid_reset=False,
            pinching=[True,True],stable_hands=[True,True],opposing_flaps=True,proof_lift=True,
            hold_time_s=.2667,rack_clearance_m=.01,unsafe_causes=dict(robot_rack_collision=False),
            staged_base=dict(phase='held_grasp',manipulation_start=100)))
    return rows,outcome


@pytest.mark.parametrize('mutation',[
    lambda o:o.update(split='validation'),lambda o:o.update(initial_layout_valid=False),
    lambda o:o['result'].update(unsafe=True),lambda o:o['result'].update(numerical_failure=True),
    lambda o:o['result'].update(proof_lift=False),lambda o:o['result'].update(hold_time_s=.1),
    lambda o:o['result'].update(pinching=[True,False]),
    lambda o:o['result']['staged_base'].update(waypoint_probe=dict(name='changed')),
])
def test_evaluation_invalid_probe_and_incomplete_physical_success_rejected(mutation):
    bank=TrainSuccessBank(480,539);rows,outcome=episode();mutation(outcome)
    with pytest.raises(ValueError):bank.add_episode(rows,outcome,source_run='unit_fixture',split='train')
    assert bank.size==0


def test_labels_are_exact_executed_goals_and_whole_paths_survive_round_trip():
    bank=TrainSuccessBank(480,539);rows,outcome=episode();bank.add_episode(rows,outcome,source_run='fixture',split='train')
    restored=TrainSuccessBank(480,539);restored.restore(bank.state())
    saved=restored.episodes[REGIONS[0]][0]['rows']
    for key in rows:torch.testing.assert_close(rows[key],saved[key],rtol=0,atol=0)
    with pytest.raises(ValueError,match='Duplicate'):bank.add_episode(rows,outcome,source_run='fixture',split='train')
    rows['action'][:,19]=0
    with pytest.raises(ValueError,match='binary'):TrainSuccessBank(480,539).add_episode(rows,outcome,source_run='fixture',split='train')


def test_region_balance_does_not_let_frequent_region_swamp_scarce_success():
    bank=TrainSuccessBank(480,539)
    for env in range(5):
        rows,outcome=episode(REGIONS[0],env=env);rows['reward'].fill_(10);bank.add_episode(rows,outcome,source_run='fixture',split='train')
    rows,outcome=episode(REGIONS[2],env=10);rows['reward'].fill_(20);bank.add_episode(rows,outcome,source_run='fixture',split='train')
    batch=bank.sample(64,'cpu');assert int((batch['reward']==10).sum())==32;assert int((batch['reward']==20).sum())==32
    normal={k:v.clone() for k,v in batch.items()};normal['reward'].fill_(-10)
    mixed,count=bank.mix(normal,.2,'cpu');assert count==13 and len(mixed['reward'])==64
    assert int((mixed['reward']>=0).sum())==13
    empty=TrainSuccessBank(480,539);same,count=empty.mix(normal,.2,'cpu');assert same is normal and count==0


def test_ended_paths_only_use_their_kept_environment_ids():
    bank=TrainSuccessBank(480,539);left,outcome=episode(env=7)
    other={k:v.clone() for k,v in left.items()};other['reward'].fill_(-100)
    batches=[]
    for i in range(3):
        batches.append((torch.tensor([4,7]),{k:torch.stack((other[k][i],left[k][i])) for k in left}))
    failed=deepcopy(outcome);failed['environment']=4;failed['result']['success']=False
    added=add_completed_training_wave(bank,dict(split='train'),[failed,outcome],batches,source_run='fixture')
    assert added==1;torch.testing.assert_close(bank.episodes[REGIONS[0]][0]['rows']['reward'],torch.arange(3,dtype=torch.float32))
    with pytest.raises(ValueError,match='Evaluation'):add_completed_training_wave(bank,dict(split='validation'),[outcome],batches,source_run='fixture')


def test_capacity_removes_whole_episodes_and_never_truncates_a_path():
    bank=TrainSuccessBank(480,539)
    for i in range(5):
        rows,outcome=episode(n=900,env=i);bank.add_episode(rows,outcome,source_run='fixture',split='train')
    assert bank.size==3600
    assert all(len(e['rows']['reward'])==900 for e in bank.episodes[REGIONS[0]])


def test_success_hook_sees_first_path_before_later_same_wave_eviction():
    bank=TrainSuccessBank(480,539); outcomes=[]; batches=[]; first_observed=[]
    for i in range(5):
        rows,outcome=episode(n=900,env=i); outcomes.append(outcome)
        batches.append((torch.full((900,),i),rows))
    def keep_first():
        if not first_observed:
            first_observed.append(deepcopy(bank.episodes[REGIONS[0]][0]))
    assert add_completed_training_wave(bank,dict(split='train'),outcomes,batches,
        source_run='fixture',after_success_added=keep_first)==5
    assert first_observed[0]['outcome']['environment']==0
    assert bank.episodes[REGIONS[0]][0]['outcome']['environment']==1


def test_approach_region_mismatch_or_nonterminal_rows_are_rejected():
    for kind in ['approach','region','terminal']:
        bank=TrainSuccessBank(480,539);rows,outcome=episode()
        if kind=='approach':rows['actor_obs'][0,-6]=0
        if kind=='region':outcome['layout']['target_region']=REGIONS[1]
        if kind=='terminal':rows['terminated'][0]=True
        with pytest.raises(ValueError):bank.add_episode(rows,outcome,source_run='fixture',split='train')


def test_matching_uses_exact_real_critic_rewards_terminal_and_clock_not_inverse_actions():
    rows,outcome=episode();rows['critic_obs'][:,0]=torch.arange(3);rows['next_critic_obs'][:,0]=torch.arange(3)+1
    rows['critic_obs'][:,530]=torch.arange(3)/410
    path={k:rows[k].clone() for k in ['critic_obs','next_critic_obs','reward','terminated']}
    path['critic_obs']=path['critic_obs'][:,:530];path['next_critic_obs']=path['next_critic_obs'][:,:530];path['first_held_step']=0
    ids=match_measured_success_paths(rows,[path],raw_critic_dim=530,clock_horizon=410)
    assert ids[0].tolist()==[0,1,2]
    changed=deepcopy(path);changed['next_critic_obs'][1,400]=1
    with pytest.raises(ValueError,match='0 matching'):match_measured_success_paths(rows,[changed],raw_critic_dim=530,clock_horizon=410)
    duplicated={k:torch.cat((v,v)) for k,v in rows.items()}
    with pytest.raises(ValueError,match='2 matching'):match_measured_success_paths(duplicated,[path],raw_critic_dim=530,clock_horizon=410)


def test_completed_episode_registration_seeds_bank_before_remaining_environment_finishes():
    bank=TrainSuccessBank(480,539);rows,done=episode(env=7);done['result']['steps']=3
    pending=deepcopy(done);pending.update(environment=4,complete=False);pending['result']['success']=False
    calls=[]
    tracker=CompletedTrainSuccessRegistration(bank,source_run='fixture',
        after_success_added=lambda:calls.append(bank.size))
    batches=[(torch.full((3,),7),rows)]
    assert tracker.register(dict(split='train'),[pending,done],batches,
        control_step=3,actor_updates=10,critic_updates=40)==1
    assert bank.size==3 and calls==[3] and not pending['complete']
    for k in rows:assert torch.equal(bank.episodes[REGIONS[0]][0]['rows'][k],rows[k])
    report=tracker.report();assert report['events'][0]['registration_control_step']==3
    assert report['events'][0]['terminal_control_step']==3
    assert tracker.register(dict(split='train'),[done],batches,
        control_step=9,actor_updates=13,critic_updates=52,at_wave_end=True)==0
    assert calls==[3] and bank.size==3


def test_wave_registration_mode_preserves_delayed_legacy_timing():
    bank=TrainSuccessBank(480,539);rows,done=episode()
    tracker=CompletedTrainSuccessRegistration(bank,source_run='fixture',mode='wave')
    batches=[(torch.zeros(3,dtype=torch.long),rows)]
    assert tracker.register(dict(split='train'),[done],batches,
        control_step=3,actor_updates=10,critic_updates=40)==0
    assert bank.size==0
    assert tracker.register(dict(split='train'),[done],batches,
        control_step=9,actor_updates=13,critic_updates=52,at_wave_end=True)==1
    assert tracker.report()['events'][0]['boundary']=='wave_end'


@pytest.mark.parametrize('bad',['incomplete','unsafe','hold','one_hand'])
def test_early_registration_cannot_bypass_completed_physical_success(bad):
    bank=TrainSuccessBank(480,539);rows,done=episode()
    if bad=='incomplete':done['complete']=False
    if bad=='unsafe':done['result']['unsafe']=True
    if bad=='hold':done['result']['hold_time_s']=.1
    if bad=='one_hand':done['result']['pinching']=[True,False]
    tracker=CompletedTrainSuccessRegistration(bank,source_run='fixture')
    with pytest.raises(ValueError):tracker.register(dict(split='train'),[done],
        [(torch.zeros(3,dtype=torch.long),rows)],control_step=3,actor_updates=10,critic_updates=40)
    assert bank.size==0 and tracker.report()['registered_completed_TRAIN_episodes']==0


def test_registration_never_imports_evaluation_or_another_environments_rows():
    bank=TrainSuccessBank(480,539);rows,done=episode(env=7)
    tracker=CompletedTrainSuccessRegistration(bank,source_run='fixture')
    for split in ('validation','test'):
        assert tracker.register(dict(split=split),[done],[],
            control_step=3,actor_updates=10,critic_updates=40)==0
        with pytest.raises(ValueError,match='Evaluation'):tracker.register(dict(split=split),[done],
            [(torch.full((3,),7),rows)],control_step=3,actor_updates=10,critic_updates=40)
    with pytest.raises(ValueError,match='missing'):tracker.register(dict(split='train'),[done],
        [(torch.full((3,),4),rows)],control_step=3,actor_updates=10,critic_updates=40)
    assert bank.size==0


def test_registration_remembers_completed_identity_after_bank_eviction():
    bank=TrainSuccessBank(480,539);tracker=CompletedTrainSuccessRegistration(bank,source_run='fixture')
    outcomes=[];batches=[]
    for i in range(5):
        rows,done=episode(n=900,env=i);outcomes.append(done);batches.append((torch.full((900,),i),rows))
        assert tracker.register(dict(split='train'),[done],batches,
            control_step=900,actor_updates=i,critic_updates=4*i)==1
    assert bank.size==3600
    assert tracker.register(dict(split='train'),outcomes,batches,
        control_step=900,actor_updates=5,critic_updates=20,at_wave_end=True)==0
    assert bank.size==3600 and tracker.report()['registered_completed_TRAIN_episodes']==5
    altered=deepcopy(outcomes[0]);altered['layout']['seed']+=1
    with pytest.raises(ValueError,match='identity changed'):tracker.register(dict(split='train'),[altered],batches,
        control_step=900,actor_updates=5,critic_updates=20,at_wave_end=True)
    contradicted=deepcopy(outcomes[0]);contradicted['result']['success']=False
    with pytest.raises(ValueError,match='successful attempt'):tracker.register(dict(split='train'),[contradicted],batches,
        control_step=900,actor_updates=5,critic_updates=20,at_wave_end=True)
