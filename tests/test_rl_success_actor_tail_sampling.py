"""Synthetic retention fixtures verify data selection, never physics success."""
import pytest
import torch

from test_rl_staged_train_success import episode
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import (
    KEYS, REGIONS, TrainSuccessBank, retention_config,
)


def bank_with_paths(config=None):
    bank=TrainSuccessBank(480,539,config=config)
    for i,region in enumerate(REGIONS):
        rows,outcome=episode(region,n=200,env=i)
        rows['actor_obs'][:,10]=torch.arange(200)
        rows['critic_obs'][:,10]=torch.arange(200)+1000
        bank.add_episode(rows,outcome,source_run='fixture',split='train')
    return bank


def test_actor_half_tail_preserves_regions_and_exact_measured_row_tuples():
    bank=bank_with_paths(retention_config('tail64-half'))
    torch.manual_seed(631)
    batch=bank.sample_actor(64,'cpu')
    for i,region in enumerate(REGIONS):
        mask=batch['actor_obs'][:,94:98].argmax(-1)==i
        assert int(mask.sum())==16
        assert int((batch['reward'][mask]>=136).sum())>=8
        original=bank.episodes[region][0]['rows']
        for row in mask.nonzero().flatten():
            index=int(batch['reward'][row])
            for key in KEYS:assert torch.equal(batch[key][row],original[key][index])


def test_opt_in_does_not_change_Q_sampling_and_default_actor_rng():
    ordinary=bank_with_paths()
    tail=bank_with_paths(retention_config('tail64-half'))
    torch.manual_seed(7);expected=ordinary.sample(64,'cpu')
    torch.manual_seed(7);actual=tail.sample(64,'cpu')
    torch.manual_seed(7);default_actor=ordinary.sample_actor(64,'cpu')
    for key in KEYS:
        assert torch.equal(expected[key],actual[key])
        assert torch.equal(expected[key],default_actor[key])
    torch.manual_seed(9);a,ca=ordinary.mix(expected,.2,'cpu')
    torch.manual_seed(9);b,cb=tail.mix(expected,.2,'cpu')
    assert ca==cb==13
    for key in KEYS:assert torch.equal(a[key],b[key])


def test_short_success_path_and_checkpoint_sampling_round_trip():
    config=retention_config('tail64-half')
    bank=TrainSuccessBank(480,539,config=config)
    rows,outcome=episode(n=3)
    bank.add_episode(rows,outcome,source_run='fixture',split='train')
    restored=TrainSuccessBank(480,539,config=config);restored.restore(bank.state())
    torch.manual_seed(43);a=bank.sample_actor(64,'cpu')
    torch.manual_seed(43);b=restored.sample_actor(64,'cpu')
    for key in KEYS:assert torch.equal(a[key],b[key])
    assert set(a['reward'].tolist())<={0.,1.,2.}
    with pytest.raises(ValueError,match='contract'):
        TrainSuccessBank(480,539).restore(bank.state())


@pytest.mark.parametrize('mutation',[
    lambda c:c['actor_sampling'].update(tail_fraction=.75),
    lambda c:c['actor_sampling'].update(tail_steps=0),
    lambda c:c['actor_sampling'].update(scope='Q_and_actor'),
])
def test_unknown_sampling_contract_rejected(mutation):
    config=retention_config('tail64-half');mutation(config)
    with pytest.raises(ValueError,match='contract'):TrainSuccessBank(480,539,config=config)


def test_reanchored_tail_checkpoint_restores_models_replay_and_four_optimizers(tmp_path):
    from test_rl_actual_flap_reanchored_sac import learned_source
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
    _,pilot,warm,physical,stage,_=learned_source(tmp_path)
    pilot.success_bank.config=retention_config('tail64-half')
    raw=torch.zeros(64,464);raw[:,144]=1.
    critic=torch.zeros(64,530);extra=torch.zeros(64,38)
    previous=pilot.act(raw,critic,0,supplemental=extra)[1]
    pilot.observe(previous,raw,critic,torch.ones(64),torch.ones(64,dtype=torch.bool),0,supplemental=extra)
    pilot.agent.update(pilot.replay.sample(64,'cpu'))
    pilot.directory.mkdir();pilot.save(final=True)
    checkpoint=next(pilot.directory.glob('checkpoint_*.pt'))
    restored=ReanchoredActualFlapSACPilot(warm,physical,tmp_path/'restore_tail',stage,checkpoint=checkpoint)
    assert restored.success_bank.config==retention_config('tail64-half')
    assert restored.contract==pilot.contract and restored.replay.size==pilot.replay.size==64
    for key,value in pilot.agent.state_dict().items():assert torch.equal(value,restored.agent.state_dict()[key])
    for before,after in zip(pilot.agent.optimizers,restored.agent.optimizers):
        a,b=before.state_dict(),after.state_dict()
        assert a['param_groups']==b['param_groups'] and a['state'].keys()==b['state'].keys()
        for key in a['state']:
            for name,value in a['state'][key].items():
                if isinstance(value,torch.Tensor):assert torch.equal(value,b['state'][key][name])
                else:assert value==b['state'][key][name]
    assert torch.equal(pilot.agent.act(previous[0],True),restored.agent.act(previous[0],True))


def actual_checkpoint_seed_fixture(tmp_path):
    from copy import deepcopy
    from test_rl_actual_flap_reanchored_sac import learned_source
    from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import MeasuredTrainCreditBank
    _,pilot,*_=learned_source(tmp_path)
    pilot.directory.mkdir();pilot.save(final=True)
    initial=torch.load(next(pilot.directory.glob('checkpoint_*.pt')),weights_only=True)
    source=deepcopy(initial);source['actor_updates']=692;source['critic_updates']=4816
    success=TrainSuccessBank(518,577)
    credit=MeasuredTrainCreditBank(518,577,initial['config']['gamma'],initial['measured_train_credit'])
    for i,region in enumerate(REGIONS):
        _,outcome=episode(region,n=3,env=i)
        rows={k:torch.zeros(3,d) for k,d in (('actor_obs',518),('next_actor_obs',518),
              ('critic_obs',577),('next_critic_obs',577),('action',21))}
        for k in ('actor_obs','next_actor_obs','critic_obs','next_critic_obs'):rows[k][:,-6]=1
        for k in ('actor_obs','next_actor_obs'):rows[k][:,94+i]=1
        rows['action'][:,19:]=-1
        rows['reward']=torch.arange(3,dtype=torch.float32)
        rows['terminated']=torch.tensor([False,False,True])
        success.add_episode(rows,outcome,source_run='fixture',split='train')
        credit.add_episode(rows,outcome,source_run='fixture')
    source['successful_train_transitions']=success.state()
    source['measured_train_credit_bank']=credit.state()
    return initial,source


def test_bank_seeding_keeps_initial_models_and_never_imports_source_Q_or_actor(tmp_path):
    from prepare_actual_success_actor_tail import identical,prepare
    initial,source=actual_checkpoint_seed_fixture(tmp_path)
    for key,value in source['model'].items():
        if key.startswith(('actor.','q1.','q2.','target1.','target2.')):value.add_(10)
    output,experience,proof=prepare(initial,source)
    assert identical(output['model'],initial['model']) and identical(output['optimizers'],initial['optimizers'])
    assert output['actor_updates']==output['critic_updates']==0
    assert len(experience['executed_goal_transitions']['reward'])==0
    assert proof['actual_completed_TRAIN_success_bank']['rows']==12
    assert proof['actual_completed_TRAIN_success_and_failure_credit_bank']['rows']==12
    assert output['goal_contract']==experience['goal_contract']
    assert output['successful_train_transitions']['config']==retention_config('tail64-half')
    assert initial['successful_train_transitions']['config']==retention_config()
    for region in REGIONS:
        assert identical(output['successful_train_transitions']['episodes'][region],
                         source['successful_train_transitions']['episodes'][region])


def test_bank_seed_explicitly_activates_matching_known_credit_from_fresh_input(tmp_path):
    from prepare_actual_success_actor_tail import prepare
    initial,source=actual_checkpoint_seed_fixture(tmp_path)
    initial.pop('measured_train_credit')
    initial.pop('measured_train_credit_bank_report',None)
    output,experience,_=prepare(initial,source)
    assert output['measured_train_credit']==source['measured_train_credit']==experience['measured_train_credit']
    assert output['measured_train_credit_bank_report']['rows']==12


def test_checkpoint_without_writer_open_credit_data_seeds_literal_success_paths(tmp_path):
    from prepare_actual_success_actor_tail import identical,prepare
    initial,source=actual_checkpoint_seed_fixture(tmp_path)
    source.pop('measured_train_credit_bank')
    output,experience,proof=prepare(initial,source)
    assert proof['initial_measured_credit_seed_failed_episodes']==0
    assert proof['initial_measured_credit_seed_source']=='completed_safe_successful_TRAIN_paths_in_immutable_checkpoint'
    assert proof['new_online_TRAIN_failures_remain_collected_and_added']
    for region in REGIONS:
        a=source['successful_train_transitions']['episodes'][region][0]
        b=experience['measured_train_credit_bank']['episodes'][region][0]
        assert a['identity']==b['identity'] and identical(a['rows'],b['rows'])
    assert output['measured_train_credit_bank_report']['rows']==12


@pytest.mark.parametrize('bad',('goal','initial_Q','DEV','broken_path','anchor'))
def test_seed_rejects_changed_context_nonfresh_Q_and_nonTRAIN_data(tmp_path,bad):
    from prepare_actual_success_actor_tail import prepare
    initial,source=actual_checkpoint_seed_fixture(tmp_path)
    if bad=='goal':source['goal_contract']['body_correction_radius']=.15
    if bad=='initial_Q':initial['model']['critic_normalizer.count'].fill_(1)
    if bad=='DEV':source['successful_train_transitions']['episodes'][REGIONS[0]][0]['outcome']['split']='validation'
    if bad=='broken_path':source['measured_train_credit_bank']['episodes'][REGIONS[0]][0]['rows']['next_actor_obs'][0,0]=1
    if bad=='anchor':source['body_anchor_state']['source_actor_updates']+=1
    with pytest.raises(ValueError):prepare(initial,source)
