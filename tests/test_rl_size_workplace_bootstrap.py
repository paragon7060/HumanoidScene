"""Failed calibration can initialize learning; it cannot become success evidence."""
from copy import deepcopy

import pytest

from test_rl_size_workplaces import search
from test_rl_region_workplaces import workplaces
from test_rl_size_workplace_actor import source
from prepare_size_workplace_actor import prepare
from prepare_actual_success_actor_tail import identical
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplaces import (
    BOOTSTRAP_FORMAT,KINDS,_evidence,build_size_workplaces,validate_size_workplaces)
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import compatibility_contract
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import (
    frozen_anchor_templates, frozen_actual_anchor_templates)


def failed_search(shift=0):
    result=search(shift)
    for case in result['cases']:
        if (case['region'],case['box_type'],case['candidate'])==('shelf_2_left','medium','candidate0'):
            case.update(success=False,unsafe=True,time_out=False)
            case['actual_terminal'].update(success=False,unsafe=True,time_out=False,
                unsafe_causes=dict(robot_rack_collision=True))
    return result


def bootstrap():
    _,original,*_=workplaces()
    selections={r:{k:'candidate0' for k in ks} for r,ks in KINDS.items()}
    return build_size_workplaces(original,failed_search(),failed_search(10000),selections,
        results_SHA256=dict(discovery='a'*64,confirmation='b'*64),checkpoint_SHA256='c'*64,
        bootstrap_failed_attempts=True)


def test_default_safety_qualification_stays_strict_and_bootstrap_is_explicit():
    _,original,*_=workplaces();selections={r:{k:'candidate0' for k in ks} for r,ks in KINDS.items()}
    with pytest.raises(ValueError,match='confirmed success or measured safe'):
        build_size_workplaces(original,failed_search(),failed_search(10000),selections,
            results_SHA256=dict(discovery='a'*64,confirmation='b'*64),checkpoint_SHA256='c'*64)
    typed=bootstrap()['size_workplaces'];entry=typed['regions']['shelf_2_left']['medium']
    assert typed['name']==BOOTSTRAP_FORMAT
    assert entry['TRAIN_evidence']['requested']==entry['TRAIN_evidence']['unsafe']==4
    assert entry['TRAIN_evidence']['success']==0
    assert entry['unproven_grasp_candidate'] and not entry['template']['measured_success']
    assert all(c['unsafe'] and not c['success'] for rows in entry['TRAIN_cases'].values() for c in rows)
    assert sum(len(v) for v in typed['regions'].values())==6
    assert frozen_anchor_templates(typed)==typed['source_shelf_templates']
    assert frozen_actual_anchor_templates(typed)==typed['source_region_workplaces']
    assert frozen_actual_anchor_templates(typed['source_region_workplaces'])==typed['source_region_workplaces']


def test_nested_actual_anchor_mapping_still_rejects_corrupted_size_provenance():
    typed=bootstrap()['size_workplaces']
    typed['regions']['shelf_2_left']['medium']['template']['base_yaw_rack_rad']+=.01
    with pytest.raises(ValueError,match='geometry'):
        frozen_actual_anchor_templates(typed)


@pytest.mark.parametrize('wrong',['lost_size','lost_failed_request','fake_success','numerical_failure',
    'invalid_fresh_only','wrong_pool','missing_explicit_contract','same_seeds'])
def test_bootstrap_keeps_full_size_scope_failure_truth_and_real_asset_guards(wrong):
    typed=deepcopy(bootstrap()['size_workplaces']);entry=typed['regions']['shelf_2_left']['medium']
    case=entry['TRAIN_cases']['confirmation'][0]
    if wrong=='lost_size':del typed['regions']['shelf_2_right']['medium']
    elif wrong=='lost_failed_request':entry['TRAIN_cases']['confirmation'].pop()
    elif wrong=='fake_success':entry['template']['measured_success']=True
    elif wrong=='numerical_failure':
        case['numerical_failure']=True;case['actual_terminal']['numerical_failure']=True
    elif wrong=='invalid_fresh_only':
        for row in entry['TRAIN_cases']['confirmation']:row['initial_invalid']=True
    elif wrong=='wrong_pool':case['actual_terminal']['target_pool_id']=case['actual_terminal']['target_logical_id']
    elif wrong=='missing_explicit_contract':del typed['failed_candidate_bootstrap']
    else:typed['frozen_TRAIN_searches']['confirmation']=deepcopy(typed['frozen_TRAIN_searches']['discovery'])
    # Counts alone cannot disguise numerical failure or deleted evidence.
    if wrong in ('numerical_failure','invalid_fresh_only'):
        entry['TRAIN_evidence']=_evidence(entry['TRAIN_cases']['discovery']+entry['TRAIN_cases']['confirmation'])
        entry['fresh_confirmation_evidence']=_evidence(entry['TRAIN_cases']['confirmation'])
    with pytest.raises(ValueError):validate_size_workplaces(typed)


def test_actor_initialization_needs_second_opt_in_and_imports_no_failed_Q_rows():
    initial,replay,_=source();waypoints=bootstrap();old=deepcopy(initial)
    with pytest.raises(ValueError,match='allow-failed-bootstrap'):
        prepare(initial,replay,waypoints,source_checkpoint_SHA256='c'*64)
    changed,experience=prepare(initial,replay,waypoints,source_checkpoint_SHA256='c'*64,allow_failed_bootstrap=True)
    assert identical(initial,old)
    for key in ('model','optimizers','actor_training_memory','body_anchor_state','frozen_actor_prior'):
        assert identical(changed[key],old[key])
    assert changed['actor_updates']==changed['critic_updates']==0
    assert compatibility_contract(changed['goal_contract'])==compatibility_contract(old['goal_contract'])
    assert not any(len(v) for v in experience['executed_goal_transitions'].values())
    assert changed['goal_contract']==experience['goal_contract']
    assert changed['goal_contract']['shelf_templates']['name']==BOOTSTRAP_FORMAT
