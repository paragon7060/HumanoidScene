from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.rewards.contact_profile import (
    ContactProgressReward,contact_shaping_contract,contact_reward_weights,
    configured_reward_weights,frozen_actor_reward_contract,opposing_pad_contact_quality,
    with_contact_reward_profile,learning_termination_mask,
)
from kuavo_isaaclab_scene.rl.multi_box.rewards.weights import MultiBoxRewardWeights
from kuavo_isaaclab_scene.rl.multi_box.success import FingerFlapContacts


def contacts(n=1):
    return FingerFlapContacts(torch.zeros(n,2,2,2),torch.ones(n,2,2,2,dtype=torch.bool),
        torch.ones(n,2,2,dtype=torch.bool),torch.ones(n,dtype=torch.bool))


def test_score_uses_actual_qualified_pads_and_opposing_flaps_without_rewarding_excess_force():
    c=contacts(5)
    c.force_n[0,0,0,0]=5.
    c.force_n[1,0,0]=5.
    c.force_n[2,0,0]=5.;c.force_n[2,1,1]=5.
    c.force_n[3,0,0]=50.;c.force_n[3,1,1]=50.
    c.force_n[4,:,0]=5.  # Both hands on the same flap are not bilateral proof.
    assert opposing_pad_contact_quality(c).tolist()==[.0625,.25,1.,1.,.25]
    c.in_region[2,1,1]=False;c.opposed[3]=False
    assert opposing_pad_contact_quality(c)[2:4].tolist()==[.25,0.]


def test_no_contact_unavailable_ambiguous_or_nonfinite_cannot_earn_contact_quality():
    c=contacts(4)
    c.force_n[1,0]=5.;c.force_n[1,1,1]=5.  # Ambiguous left hand.
    c.force_n[2,0,0]=5.;c.force_n[2,1,1]=5.;c.available[2]=False
    c.force_n[3,0,0]=5.;c.force_n[3,1,1]=5.;c.force_n[3,0,0,0]=float('nan')
    assert opposing_pad_contact_quality(c).tolist()==[0.,0.,0.,0.]


def test_contact_acquire_lose_cycle_and_timeout_cannot_farm_progress():
    c=contacts();tracker=ContactProgressReward(1,'cpu',discount=.999,config=contact_shaping_contract())
    mask=torch.ones(1,dtype=torch.bool);off=~mask
    assert tracker.step(c,trainable=mask,terminated=off).item()==0
    c.force_n[0,0,0]=5.;c.force_n[0,1,1]=5.
    acquire=tracker.step(c,trainable=mask,terminated=off).item()
    held=tracker.step(c,trainable=mask,terminated=off).item()
    end=tracker.step(c,trainable=mask,terminated=mask).item()
    assert acquire==pytest.approx(.999) and held<0 and end==-1
    assert acquire+.999*held+.999**2*end==pytest.approx(0,abs=1e-7)


def test_partial_reset_and_settling_rebase_only_the_requested_environment():
    c=contacts(2);c.force_n[:,0,0]=5.;c.force_n[:,1,1]=5.
    tracker=ContactProgressReward(2,'cpu',discount=.999,config=contact_shaping_contract())
    mask=torch.ones(2,dtype=torch.bool)
    assert not tracker.step(c,trainable=mask,terminated=~mask).any()
    tracker.reset(torch.tensor([1]))
    reward=tracker.step(c,trainable=mask,terminated=~mask)
    assert reward[0]<0 and reward[1]==0
    assert not tracker.step(c,trainable=~mask,terminated=~mask).any()


def base_contract():
    return dict(reward_profile=dict(weights=asdict(MultiBoxRewardWeights()),geometry_profile='unchanged'),
        observations=dict(policy=[464],critic=[66]),terminal_contract=dict(safety='unchanged'))


def test_contact_profile_is_explicit_and_only_frozen_actor_compatibility_can_strip_it():
    old=base_contract();original=deepcopy(old);new=with_contact_reward_profile(old)
    assert old==original and new['terminal_contract']==old['terminal_contract']
    assert frozen_actor_reward_contract(new)==old
    assert configured_reward_weights(new['reward_profile'])==contact_reward_weights()
    changed=deepcopy(new);changed['reward_profile']['weights']['grasp']['success_event']=99.
    with pytest.raises(ValueError):frozen_actor_reward_contract(changed)
    with pytest.raises(ValueError):with_contact_reward_profile(new)


def test_absorbing_timeout_contact_profile_stops_Q_bootstrap_without_rewriting_simulator_facts():
    terminated=torch.tensor([False,True,False]);truncated=torch.tensor([True,False,False])
    old=base_contract();new=with_contact_reward_profile(old)
    assert learning_termination_mask(old['reward_profile'],terminated,truncated) is terminated
    assert learning_termination_mask(new['reward_profile'],terminated,truncated).tolist()==[True,True,False]
    assert terminated.tolist()==[False,True,False] and truncated.tolist()==[True,False,False]


def test_actor_only_contact_migration_keeps_Q_optimizers_and_replay_fresh():
    from kuavo_isaaclab_scene.rl.algorithms.asymmetric_sac import AsymmetricSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    torch.manual_seed(3)
    source=AsymmetricSAC(4,8,2,SACConfig(hidden=8))
    agent=AsymmetricSAC(4,8,2,SACConfig(hidden=8))
    old=base_contract();new=with_contact_reward_profile(old)
    source_contract=dict(physical_contract=old,action_coordinates='unchanged')
    target_contract=dict(physical_contract=new,action_coordinates='unchanged')
    state=dict(artifact_type='test_actor',goal_contract=source_contract,model=source.state_dict(),actor_updates=30,critic_updates=80)
    pilot=SimpleNamespace(agent=agent,critic_updates=0,actor_updates=0,replay=SimpleNamespace(size=0),
        artifact_type='test_actor',contract=target_contract,frozen_actor_prior=None)
    q={k:v.clone() for k,v in agent.q1.state_dict().items()}
    with pytest.raises(ValueError):initialize_staged_actor_only(pilot,state)
    receipt=initialize_staged_actor_only(pilot,state,allow_contact_reward_change=True)
    assert receipt['actual_replay_rows']==0 and receipt['contact_reward_change']
    assert all(torch.equal(v,agent.q1.state_dict()[k]) for k,v in q.items())
    assert all(not opt.state for opt in agent.optimizers)
    for k,v in source.actor.state_dict().items():assert torch.equal(v,agent.actor.state_dict()[k])
    bad=deepcopy(state);bad['goal_contract']['physical_contract']['terminal_contract']['safety']='different'
    with pytest.raises(ValueError):initialize_staged_actor_only(pilot,bad,allow_contact_reward_change=True)


def test_confident_jaw_migration_preserves_executed_signs_and_original_controller_network():
    from torch import nn
    from kuavo_isaaclab_scene.rl.algorithms.hybrid_goal_sac import HybridGoalSAC
    from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only
    class Projection:
        free_grippers=True
        def __call__(self,obs,action):return action
    cfg=SACConfig(hidden=8,actor_feature_mode='flat')
    source=HybridGoalSAC(8,10,21,cfg,action_projector=Projection(),validated_jaw_prior_confidence=.8,jaw_prior_residual_gain=20.)
    with torch.no_grad():source.actor.network[-1].weight[19:21]=0.;source.actor.network[-1].bias[19:21]=.2
    prior=nn.ModuleDict(dict(actor=deepcopy(source.actor),actor_normalizer=deepcopy(source.actor_normalizer)))
    source.validated_jaw_prior=lambda x:prior['actor'].network(x).chunk(2,-1)[0][:,19:21]
    with torch.no_grad():source.actor.network[-1].bias[19:21]=.1
    agent=HybridGoalSAC(8,10,21,cfg,action_projector=Projection(),validated_jaw_prior_confidence=.8,jaw_prior_residual_gain=20.)
    destination_prior=nn.ModuleDict(dict(actor=deepcopy(agent.actor),actor_normalizer=deepcopy(agent.actor_normalizer)))
    agent.validated_jaw_prior=lambda x:destination_prior['actor'].network(x).chunk(2,-1)[0][:,19:21]
    contract=dict(physical_contract=base_contract())
    pilot=SimpleNamespace(agent=agent,actor_updates=0,critic_updates=0,replay=SimpleNamespace(size=0),
        artifact_type='test',contract=contract,frozen_actor_prior=destination_prior,validated_jaw_prior_confidence=.8)
    saved=dict(artifact_type='test',goal_contract=contract,model=source.state_dict(),frozen_actor_prior=prior.state_dict(),actor_updates=1,critic_updates=4)
    initialize_staged_actor_only(pilot,saved)
    x=torch.zeros(2,8)
    assert torch.equal(agent.act(x,True),source.act(x,True))
    assert agent.act(x,True)[:,19:21].eq(-1.).all()
    # Re-anchoring to the learned network would falsely close both jaws.
    destination_prior.load_state_dict({k:v for k,v in agent.state_dict().items() if k.startswith(('actor.','actor_normalizer.'))})
    assert agent.act(x,True)[:,19:21].eq(1.).all()
