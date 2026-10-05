"""Opt-in tactile shaping with explicit fresh-Q reward identity."""
from copy import deepcopy
from dataclasses import asdict,replace

import torch

from .weights import MultiBoxRewardWeights
from ..success import classify_pinches


NAME='opposing_pad_contact_progress_v1'


def contact_shaping_contract():
    return dict(name=NAME,min_jaw_force_n=5.,contact_progress_weight=1.,
        pad_force_saturates_at_success_threshold=True,
        score='weaker_pad_then_weaker_hand_opposing_flaps',
        geometry='filtered_contact_in_panel_region_and_opposed_jaws',
        terminal_potential_zero=True,time_limit_is_absorbing_for_Q=True,
        closing_without_contact_reward=False)


def learning_termination_mask(profile,terminated,truncated):
    """Keep simulator facts intact while matching Q to absorbing task timeout."""
    if 'contact_shaping' not in profile:return terminated
    frozen_actor_reward_contract(dict(reward_profile=profile))
    if terminated.shape!=truncated.shape or terminated.dtype!=torch.bool or truncated.dtype!=torch.bool:
        raise ValueError('Q terminal and timeout masks must match')
    return terminated | truncated


def contact_reward_weights():
    original=MultiBoxRewardWeights()
    weights=replace(original,
        common=replace(original.common,box_drop=12.,workspace_limit=12.),
        grasp=replace(original.grasp,one_hand_pinch_event=.5,bilateral_pinch_event=2.,success_event=8.),
        high_level=replace(original.high_level,failure_event=12.))
    weights.validate()
    dense=sum(getattr(weights.grasp,k) for k in ('approach_progress','front_staging_progress',
        'alignment_progress','capture_progress','jaw_gap_progress','proof_lift_progress'))
    if weights.grasp.success_event<=dense+contact_shaping_contract()['contact_progress_weight']:
        raise ValueError('Success must dominate contact and geometric shaping')
    return weights


def with_contact_reward_profile(contract):
    result=deepcopy(contract)
    profile=result['reward_profile']
    if 'contact_shaping' in profile or profile.get('weights')!=asdict(MultiBoxRewardWeights()):
        raise ValueError('Fresh contact reward requires the unchanged recognized source reward')
    profile.update(weights=asdict(contact_reward_weights()),contact_shaping=contact_shaping_contract())
    return result


def frozen_actor_reward_contract(contract):
    """Restore the legacy reward ONLY for reading a frozen actor's inputs.

    Never use this derived contract to label, seed, or resume new-reward Q.
    Every non-reward observation/action/safety field remains untouched.
    """
    profile=contract.get('reward_profile',{})
    if 'contact_shaping' not in profile:return contract
    if profile['contact_shaping']!=contact_shaping_contract() \
            or profile.get('weights')!=asdict(contact_reward_weights()):
        raise ValueError('Unknown or modified contact reward cannot bypass compatibility')
    result=deepcopy(contract)
    result['reward_profile'].pop('contact_shaping')
    result['reward_profile']['weights']=asdict(MultiBoxRewardWeights())
    return result


def configured_reward_weights(profile):
    if profile.get('contact_shaping') is None:
        if profile.get('weights')!=asdict(MultiBoxRewardWeights()):
            raise ValueError('Unknown legacy reward weights')
        return MultiBoxRewardWeights()
    frozen_actor_reward_contract(dict(reward_profile=profile))
    return contact_reward_weights()


def opposing_pad_contact_quality(contacts):
    """Bounded physical contact score; neither jaw commands nor distances enter."""
    contacts.validate()
    finite=torch.isfinite(contacts.force_n).flatten(1).all(-1)
    force=torch.where(torch.isfinite(contacts.force_n),contacts.force_n,0.).clamp_min(0)
    pad=(force/5.).clamp_max(1.)*contacts.in_region
    hand=(.25*pad.sum(-1)+.5*pad.amin(-1))*contacts.opposed
    def pair(left,right):return .25*(left+right)+.5*torch.minimum(left,right)
    score=torch.maximum(pair(hand[:,0,0],hand[:,1,1]),pair(hand[:,0,1],hand[:,1,0]))
    pinch=classify_pinches(contacts,min_jaw_force_n=5.)
    valid=finite & contacts.available & ~pinch.ambiguous_hands.any(-1)
    return torch.where(valid,score,0.)


class ContactProgressReward:
    """Potential difference with reset isolation and absorbing terminal states."""
    def __init__(self,count,device,*,discount,config):
        if config!=contact_shaping_contract():raise ValueError('Unknown contact shaping identity')
        if not 0<discount<=1:raise ValueError('Expected contact discount in(0,1]')
        self.discount,self.weight=discount,config['contact_progress_weight']
        self.previous=torch.zeros(count,device=device)
        self.initialized=torch.zeros(count,dtype=torch.bool,device=device)
        self.last_quality=torch.zeros(count,device=device)

    def reset(self,ids=None):
        ids=slice(None) if ids is None else ids
        self.previous[ids]=0.;self.initialized[ids]=False;self.last_quality[ids]=0.

    def step(self,contacts,*,trainable,terminated):
        quality=opposing_pad_contact_quality(contacts)
        if quality.shape!=self.previous.shape or trainable.shape!=quality.shape or terminated.shape!=quality.shape \
                or trainable.dtype!=torch.bool or terminated.dtype!=torch.bool:
            raise ValueError('Contact reward masks must match environments')
        current=torch.where(trainable & ~terminated,quality,0.)
        previous=torch.where(self.initialized,self.previous,self.discount*current)
        reward=self.weight*(self.discount*current-previous)
        self.previous.copy_(current);self.initialized.copy_(trainable);self.last_quality.copy_(quality)
        return torch.where(trainable,reward,0.)
