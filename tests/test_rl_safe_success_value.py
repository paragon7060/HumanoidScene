"""Fresh reward identity must not change safety or silently reuse old Q labels."""
from copy import deepcopy
from dataclasses import asdict,replace
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.rewards.success_value import (
    with_success_value_profile,configured_success_value,success_value_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
    configured_reward_weights,frozen_actor_reward_contract,contact_reward_weights,learning_termination_mask,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.precision_capture import (
    with_precision_capture_profile,configured_capture_geometry,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.absorbing_geometry import (
    with_absorbing_geometry_profile,configured_geometry_shaping,absorbing_geometry_contract,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.model import MultiBoxRewardModel
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import ReanchoredActualFlapSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import TrainSuccessBank,retention_config,KEYS
from kuavo_isaaclab_scene.rl.multi_box.experiments.measured_train_credit import measured_credit_config,VARIANT,MeasuredTrainCreditBank
from prepare_success_value_actor import prepare
from test_rl_precision_capture import source_contract
from test_rl_multi_box_rewards import grasp


@pytest.mark.parametrize('absorbing',[False,True])
def test_success64_only_changes_reviewed_reward_identity(absorbing):
    old=with_precision_capture_profile(source_contract())
    if absorbing:old=with_absorbing_geometry_profile(old)
    saved=deepcopy(old);new=with_success_value_profile(old)
    assert old==saved and configured_reward_weights(old['reward_profile']).grasp.success_event==8
    assert asdict(configured_reward_weights(new['reward_profile']))==asdict(replace(contact_reward_weights(),grasp=replace(contact_reward_weights().grasp,success_event=64.)))
    assert frozen_actor_reward_contract(new)==frozen_actor_reward_contract(old)
    assert configured_capture_geometry(new['reward_profile'])==configured_capture_geometry(old['reward_profile'])
    assert configured_geometry_shaping(new['reward_profile'])==configured_geometry_shaping(old['reward_profile'])
    assert configured_success_value(new['reward_profile'])==success_value_contract()
    assert learning_termination_mask(new['reward_profile'],torch.tensor([False]),torch.tensor([True])).all()
    new['terminal_contract']['rack_force']=99
    assert frozen_actor_reward_contract(new)!=frozen_actor_reward_contract(old)


@pytest.mark.parametrize('tamper',['marker','weight','missing_marker','other_weight'])
def test_unknown_success_reward_cannot_pass_frozen_compatibility(tamper):
    new=with_success_value_profile(with_precision_capture_profile(source_contract()))
    profile=new['reward_profile']
    if tamper=='marker':profile['success_value']['success_event_weight']=63
    elif tamper=='weight':profile['weights']['grasp']['success_event']=63
    elif tamper=='missing_marker':profile.pop('success_value')
    else:profile['weights']['common']['robot_rack_collision']=0
    with pytest.raises(ValueError):frozen_actor_reward_contract(new)
    with pytest.raises(ValueError):configured_reward_weights(profile)


def test_unsafe_success_never_receives_large_bonus():
    profile=with_success_value_profile(source_contract())['reward_profile'];model=MultiBoxRewardModel(configured_reward_weights(profile))
    assert model.grasp(grasp()).terms['success_event'].eq(64).all()
    unsafe=model.grasp(grasp(drop=True))
    assert unsafe.terms['success_event'].eq(0).all() and unsafe.total.lt(0).all()
    assert unsafe.terms['box_drop'].eq(-12).all()


def test_general_weight_validator_keeps_its_original_ratio():
    ordinary=replace(contact_reward_weights(),grasp=replace(contact_reward_weights().grasp,success_event=64.))
    with pytest.raises(ValueError):ordinary.validate()
    special=configured_reward_weights(with_success_value_profile(source_contract())['reward_profile'])
    special.validate()
    with pytest.raises(ValueError):replace(special,common=replace(special.common,robot_rack_collision=0)).validate()


@pytest.mark.parametrize('event,term',[
    ('robot_rack_collision_event','robot_rack_collision'),
    ('self_collision_event','self_collision'),('obstacle_collision_event','obstacle_collision'),
    ('box_drop_event','box_drop'),('workspace_limit_event','workspace_limit')])
def test_each_unsafe_event_keeps_identical_penalties_and_suppresses_success(event,term):
    value=grasp(n=1);value=replace(value,common=replace(value.common,**{event:torch.ones(1,dtype=torch.bool)}))
    source=MultiBoxRewardModel(contact_reward_weights()).grasp(value)
    changed=MultiBoxRewardModel(configured_reward_weights(with_success_value_profile(source_contract())['reward_profile'])).grasp(value)
    assert changed.terms['success_event'].eq(0).all()
    assert changed.terms['one_hand_pinch_event'].eq(0).all() and changed.terms['bilateral_pinch_event'].eq(0).all()
    assert changed.terms[term].lt(0).all()
    torch.testing.assert_close(changed.total,source.total,atol=0,rtol=0)


def inputs():
    credit=measured_credit_config(VARIANT);retention=retention_config('tail64-half')
    goal=dict(actor_dim=518,critic_dim=577,physical_contract=with_precision_capture_profile(source_contract()),train_success_retention=retention)
    bank=TrainSuccessBank(518,577,retention);measured=MeasuredTrainCreditBank(518,577,.999,credit)
    initial=dict(artifact_type=ReanchoredActualFlapSACPilot.artifact_type,action_dim=21,actor_updates=0,critic_updates=0,optimizers=[dict(state={}) for _ in range(4)],goal_contract=goal,config=dict(gamma=.999),model={'actor.weight':torch.tensor([1.]),'jaw_actor.weight':torch.tensor([2.]),'actor_normalizer.mean':torch.tensor([3.]),'critic_normalizer.count':torch.tensor(0.),'q1.weight':torch.tensor([4.]),'q2.weight':torch.tensor([5.])},body_anchor_state={},frozen_warm_start={},frozen_actor_prior={},successful_train_transitions=bank.state(),measured_train_credit=credit,measured_train_credit_bank_report=measured.report())
    actor=deepcopy(initial);actor.update(actor_updates=4337,critic_updates=19396)
    for k in actor['model']:actor['model'][k]=actor['model'][k]+10
    replay=dict(goal_contract=deepcopy(goal),executed_goal_transitions={k:torch.empty(0) for k in KEYS},measured_train_credit=deepcopy(credit),successful_train_transitions=bank.state(),measured_train_credit_bank=measured.state())
    return initial,actor,replay


def test_fresh_q_fork_keeps_measured_actor_but_excludes_old_critics_and_labels():
    initial,actor,replay=inputs();state,experience=prepare(initial,actor,replay)
    for k in ('actor.weight','jaw_actor.weight','actor_normalizer.mean'):assert torch.equal(state['model'][k],actor['model'][k])
    for k in ('q1.weight','q2.weight','critic_normalizer.count'):assert torch.equal(state['model'][k],initial['model'][k])
    assert all(not o['state'] for o in state['optimizers']) and state['actor_updates']==state['critic_updates']==0
    assert not any(state['successful_train_transitions']['episodes'].values())
    assert not any(experience['measured_train_credit_bank']['episodes'].values())
    assert not any(len(v) for v in experience['executed_goal_transitions'].values())
    assert initial['goal_contract']==actor['goal_contract']==replay['goal_contract']
    assert state['goal_contract']['physical_contract']['reward_profile']['success_value']==success_value_contract()


@pytest.mark.parametrize('tamper',['trained_Q','optimizer','controller','old_online_rows'])
def test_fresh_q_fork_refuses_unmatched_or_nonfresh_sources(tamper):
    initial,actor,replay=inputs()
    if tamper=='trained_Q':initial['critic_updates']=1
    elif tamper=='optimizer':initial['optimizers'][0]['state']={'trained':True}
    elif tamper=='controller':actor['body_anchor_state']={'different':True}
    else:replay['executed_goal_transitions']['reward']=torch.ones(1)
    with pytest.raises(ValueError):prepare(initial,actor,replay)
