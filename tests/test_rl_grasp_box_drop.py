"""Actual floor-height regression, reset-relative limits and replay identity."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
from kuavo_isaaclab_scene.rl.multi_box.geometry.box_drop import (
    grasp_box_drop, grasp_drop_limit, grasp_drop_terminal_contract,
    grasp_drop_safety_thresholds, configured_drop_limit, configure_grasp_drop,
    frozen_drop_actor_contract, with_reset_drop_profile)


def contract(spec=None):
    spec=spec or replace(MultiBoxSpec(),skill='grasp')
    return dict(terminal_contract=dict(success='exact_grasp_success',
        safety_thresholds=dict(rack_contact_force_n=10.,obstacle_contact_force_n=5.,
            **grasp_drop_safety_thresholds(spec)),**grasp_drop_terminal_contract(spec)))


def test_recorded_side_resting_floor_box_is_a_drop_despite_center_above_12cm():
    # Terminal25 DEV4 env122 was labeled safe timeout on the old predicate.
    height=torch.tensor([.1229799986,1.665+.008,1.665-.03])
    delta=height-1.665
    finite=torch.ones(3,dtype=torch.bool)
    assert grasp_box_drop(height,delta,finite,None).tolist()==[False,False,False]
    assert grasp_box_drop(height,delta,finite,.10).tolist()==[True,False,False]


def test_failure_is_translation_invariant_and_allows_small_settling_or_proof_lift():
    origins=torch.tensor([0.,10.,-8.])
    initial=origins+1.665
    absolute=initial+torch.tensor([-.11,.008,-.03])
    assert grasp_box_drop(absolute-origins,absolute-initial,
                          torch.ones(3,dtype=torch.bool),.10).tolist()==[True,False,False]


def test_nonfinite_baseline_or_state_fails_and_legacy_floor_guard_remains():
    height=torch.tensor([.11,1.6,1.6,1.6])
    lift=torch.tensor([0.,0.,float('nan'),float('inf')])
    assert grasp_box_drop(height,lift,torch.tensor([True,False,True,True]),.10).all()


@pytest.mark.parametrize('skill',['carry','place','full'])
def test_carry_and_place_may_intentionally_lower_to_the_conveyor(skill):
    spec=replace(MultiBoxSpec(),skill=skill)
    assert grasp_drop_limit(spec) is None
    assert grasp_drop_terminal_contract(spec)==grasp_drop_safety_thresholds(spec)=={}
    assert not grasp_box_drop(torch.tensor([.8]),torch.tensor([-.8]),torch.tensor([True]),grasp_drop_limit(spec)).item()


def test_runtime_matches_recorded_new_and_legacy_contracts_without_mutating_sources():
    cfg=SimpleNamespace(multi_box=replace(MultiBoxSpec(),skill='grasp'))
    old=dict(terminal_contract=dict(safety_thresholds={'rack_contact_force_n':10.}))
    before=deepcopy(old)
    configure_grasp_drop(cfg,old)
    assert cfg.multi_box.max_box_drop_height is None and old==before
    new=contract();before=deepcopy(new)
    configure_grasp_drop(cfg,new)
    assert cfg.multi_box.max_box_drop_height==.10 and new==before
    assert grasp_drop_safety_thresholds(cfg.multi_box)['max_box_drop_height_m']==.10


@pytest.mark.parametrize('field,value',[
    ('box_drop_reference','unknown'),('box_drop_height_m',True),
    ('box_drop_height_m',float('nan')),('box_drop_height_m',.2)])
def test_unknown_or_inconsistent_recorded_guard_is_rejected(field,value):
    c=contract();c['terminal_contract'][field]=value
    with pytest.raises(ValueError,match='box-drop'):configured_drop_limit(c)


def test_only_known_actor_compatibility_path_can_strip_the_new_guard():
    c=contract();before=deepcopy(c)
    old=frozen_drop_actor_contract(c)
    assert c==before and configured_drop_limit(old) is None
    assert old['terminal_contract']['safety_thresholds']=={'rack_contact_force_n':10.,'obstacle_contact_force_n':5.}
    changed=contract(replace(MultiBoxSpec(),skill='grasp',max_box_drop_height=.2))
    with pytest.raises(ValueError,match='reviewed'):frozen_drop_actor_contract(changed)


def test_new_reward_mdp_keeps_every_existing_field_and_cannot_migrate_twice():
    old=frozen_drop_actor_contract(contract()) | {'skill':'grasp'}
    old['reward_profile']={'success_event':64.}
    before=deepcopy(old)
    new=with_reset_drop_profile(old)
    assert old==before and frozen_drop_actor_contract(new)==old
    assert configured_drop_limit(new)==.10
    with pytest.raises(ValueError,match='historical'):with_reset_drop_profile(new)
    with pytest.raises(ValueError,match='historical'):with_reset_drop_profile(old | {'skill':'carry'})


def test_standard_sac_data_and_ppo_checkpoint_refuse_old_drop_labels(tmp_path):
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2_sac import _compatible_checkpoint as sac
    from kuavo_isaaclab_scene.rl.multi_box.experiments.train_grasp_v2 import _compatible_checkpoint as ppo
    checkpoint=tmp_path/'checkpoint.pt'
    old=frozen_drop_actor_contract(contract())
    (tmp_path/'manifest.json').write_text(json.dumps(old))
    for data_only in (False,True):
        with pytest.raises(ValueError,match='terminal_contract'):sac(checkpoint,contract(),data_only=data_only)
    with pytest.raises(ValueError,match='box_drop_contract'):
        ppo(checkpoint,dict(box_drop_contract=grasp_drop_terminal_contract(replace(MultiBoxSpec(),skill='grasp'))))


@pytest.mark.parametrize('value',[True,0.,-.1,float('nan'),float('inf'),.51])
def test_invalid_spec_limit_is_rejected(value):
    with pytest.raises(ValueError,match='box-drop'):
        replace(MultiBoxSpec(),max_box_drop_height=value).validate()
