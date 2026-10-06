from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from test_rl_workplace_results import fixture
from test_rl_staged_base_hold import Coordinates,scene
from test_rl_actual_flap_residual_sac import pilots
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import (
    build_region_workplaces,validate_region_workplaces)
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot,actor_anchor_state
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_goal_sac import initialize_staged_actor_only


def workplaces():
    manifest,metrics=fixture();summary=summarize_workplace_results(manifest,metrics)
    source=json.loads((Path(__file__).parents[1]/'docs/assets/rl_v2_staged_base_hold_candidates_20261004.json').read_text())
    selections=dict.fromkeys(summary['regions'],'candidate0');selections['shelf_3_right']='candidate1'
    return source,build_region_workplaces(source,summary,selections,results_SHA256='a'*64),summary,selections


def test_safe_candidate_does_not_become_a_success_or_drop_invalid_denominators():
    source,changed,*_=workplaces();regional=changed['region_workplaces']
    assert source['shelves']==regional['source_shelf_templates']
    right=regional['regions']['shelf_3_right']
    assert right['unproven_grasp_candidate'] and not right['template']['measured_success']
    assert right['TRAIN_evidence']['requested']==4 and right['TRAIN_evidence']['time_out']==4
    raw,_=scene();raw[:,94:98]=0.;raw[:,97]=1.
    stage=StagedBaseHoldDiagnostic(Coordinates(),changed,raw)
    expected=source['shelves']['upper']['base_minus_initial_box_xy_rack_m'][0]+.01
    assert stage.target_xy[0,0]==pytest.approx(expected)
    command=stage.action(raw)
    assert torch.equal(command[:,3:20],torch.zeros(1,17)) and command[:,20:22].tolist()==[[-1.,-1.]]
    assert stage.report()['region_workplace_candidate']['unproven_grasp_candidate']


@pytest.mark.parametrize('kind',['fake_success','wrong_geometry','unsafe_candidate','lost_denominator','heldout_source'])
def test_regional_evidence_cannot_silently_change_geometry_scope_or_success(kind):
    _,changed,*_=workplaces();regional=changed['region_workplaces'];right=regional['regions']['shelf_3_right']
    if kind=='fake_success':right['template']['measured_success']=True
    elif kind=='wrong_geometry':right['template']['base_minus_initial_box_xy_rack_m'][0]+=.1
    elif kind=='unsafe_candidate':right['TRAIN_evidence'].update(unsafe=1,time_out=3)
    elif kind=='lost_denominator':right['TRAIN_evidence']['requested']=3
    else:regional['source_shelf_templates']['upper']['source_split']='holdout'
    with pytest.raises(ValueError):validate_region_workplaces(regional)


def test_only_explicit_actor_migration_accepts_region_targets_and_old_Q_cannot_resume(tmp_path):
    _,_,warm,physical,old_stage,nominal=pilots(tmp_path)
    source_waypoints,changed,*_=workplaces();source_templates=source_waypoints['shelves']
    old_stage=SimpleNamespace(**vars(old_stage));old_stage.templates=source_templates
    nominal=deepcopy(nominal);nominal['goal_contract']['shelf_templates']=source_templates
    anchor=actor_anchor_state(nominal)
    old=ActualFlapResidualSACPilot(warm,physical,tmp_path/'old_region_source',old_stage,
        body_anchor_state=anchor,replay_capacity=1024,exploration_correlation=.99,train_success_retention=True)
    old.directory.mkdir();old.save(final=True);old_checkpoint=next(old.directory.glob('checkpoint_*.pt'))
    source=torch.load(old_checkpoint,weights_only=True)
    stage=SimpleNamespace(**vars(old_stage));stage.templates=changed['region_workplaces']
    new=ActualFlapResidualSACPilot(warm,physical,tmp_path/'new_region',stage,
        body_anchor_state=anchor,replay_capacity=1024,exploration_correlation=.99,train_success_retention=True)
    with pytest.raises(ValueError,match='matching coordinates'):initialize_staged_actor_only(new,source)
    proof=initialize_staged_actor_only(new,source,allow_workplace_change=True)
    assert proof['workplace_change'] and new.replay.size==new.actor_updates==new.critic_updates==new.success_bank.size==0
    assert not any(opt.state for opt in new.agent.optimizers)
    assert all(torch.equal(v,source['model'][k]) for k,v in new.agent.state_dict().items()
        if k.startswith(('actor.','actor_normalizer.')))
    with pytest.raises(ValueError,match='same phase/waypoint'):
        ActualFlapResidualSACPilot(warm,physical,tmp_path/'bad_region_resume',stage,checkpoint=old_checkpoint,training=False)
    new.directory.mkdir();new.save(final=True);saved=next(new.directory.glob('checkpoint_*.pt'))
    restored=ActualFlapResidualSACPilot(warm,physical,tmp_path/'restored_region',stage,checkpoint=saved,training=False)
    assert restored.contract==new.contract and all(torch.equal(v,restored.agent.state_dict()[k])
        for k,v in new.agent.state_dict().items())
