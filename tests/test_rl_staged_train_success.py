"""Retention fixtures test data separation, not simulated physical success."""
from copy import deepcopy
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import (
    TrainSuccessBank,REGIONS,add_completed_training_wave,retention_config,
    match_measured_success_paths,
)


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
