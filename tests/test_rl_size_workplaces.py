from copy import deepcopy

import pytest
import torch

from test_rl_region_workplaces import workplaces
from test_rl_size_workplace_probe import sized_request
from test_rl_workplace_results import fixture
from test_rl_staged_base_hold import scene,Coordinates
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplace_probe import validate_size_workplace_request
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplaces import (
    KINDS,build_size_workplaces,validate_size_workplaces)
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import (
    frozen_anchor_templates,validate_requested_region_stages)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M


def search(seed_shift=0):
    manifest,metrics=fixture();waves=sized_request()
    _,template,*_=workplaces()
    for row in waves[0]['layouts']:row['layout']['seed']+=seed_shift
    manifest['layout_waves']=waves
    metrics['CPU_workplace_probe']['unmeasured_size_workplace_probe']=validate_size_workplace_request(waves)
    for outcome,row in zip(metrics['outcomes'],waves[0]['layouts']):
        outcome['layout']=row['layout'];layout=row['layout'];kind=layout['target_box_type'];region=layout['target_region']
        raw,_=scene();raw[:,89:91]=0;raw[:,89+int(kind=='medium')]=1
        raw[:,91:94]=torch.tensor(BOX_DIMENSIONS_M[kind]);raw[:,94:98]=0
        ids={'shelf_2_right':0,'shelf_2_left':1,'shelf_3_right':2,'shelf_3_left':3}
        raw[:,94+ids[region]]=1
        stage=StagedBaseHoldDiagnostic(Coordinates(),template,raw,unmeasured_size_probe=True)
        offset=row['waypoint_probe']['offset_xy_yaw']
        stage.target_xy+=stage.target_xy.new_tensor(offset[:2]);stage.target_yaw+=offset[2]
        report=stage.report();report['waypoint_probe']=row['waypoint_probe']
        logical={'shelf_2_right':1,'shelf_2_left':4,'shelf_3_right':6,'shelf_3_left':9}[region]
        pool=(logical+int(kind=='medium')*6) if logical<6 else logical+6
        outcome['result'].update(staged_base=report,target_logical_id=logical,target_pool_id=pool)
    return summarize_workplace_results(manifest,metrics)


def prepared():
    _,original,*_=workplaces()
    first=search();second=search(10000)
    selected={region:{kind:'candidate0' for kind in kinds} for region,kinds in KINDS.items()}
    return original,build_size_workplaces(original,first,second,selected,
        results_SHA256=dict(discovery='a'*64,confirmation='b'*64),checkpoint_SHA256='c'*64)


def test_all_six_supported_sizes_keep_disjoint_TRAIN_evidence_and_source_anchor():
    original,changed=prepared();contract=changed['size_workplaces']
    assert 'region_workplaces' not in changed
    assert frozen_anchor_templates(contract)==original['shelves']
    assert contract['source_region_workplaces']==original['region_workplaces']
    assert sum(len(v) for v in contract['regions'].values())==6
    for region,variants in contract['regions'].items():
        for kind,entry in variants.items():
            assert entry['TRAIN_evidence']['requested']==(4 if region.startswith('shelf_2') else 8)
            assert entry['template']['box_size_m']==list(BOX_DIMENSIONS_M[kind])
            if kind=='medium':
                assert entry['unproven_grasp_candidate'] and not entry['template']['measured_success']
                assert entry['TRAIN_evidence']['success']==0


@pytest.mark.parametrize('region,kind',[(r,k) for r,ks in KINDS.items() for k in ks])
def test_perceived_region_and_actual_asset_choose_only_the_measured_typed_stage(region,kind):
    _,changed=prepared();raw,_=scene();raw[:,89:91]=0;raw[:,89+int(kind=='medium')]=1
    raw[:,91:94]=torch.tensor(BOX_DIMENSIONS_M[kind]);raw[:,94:98]=0
    names=changed['size_workplaces']['perceived_region_names_by_id'];raw[:,94+names.index(region)]=1
    stage=StagedBaseHoldDiagnostic(Coordinates(),changed,raw)
    assert stage.templates==changed['size_workplaces'] and stage.size_workplace['box_type']==kind
    assert stage.template==changed['size_workplaces']['regions'][region][kind]['template']
    assert stage.unmeasured_size_workplace_probe is None
    validate_requested_region_stages([stage],[dict(layout=dict(target_region=region,target_box_type=kind))])
    if region.startswith('shelf_2'):
        with pytest.raises(ValueError,match='original requested layout'):
            validate_requested_region_stages([stage],[dict(layout=dict(target_region=region,target_box_type='medium' if kind=='small' else 'small'))])


@pytest.mark.parametrize('wrong',['same_seeds','lost_size','fake_success','wrong_pool','wrong_offset','heldout',
    'lost_denominator','numerical_success','unsafe_unproven','missing_confirmation'])
def test_incomplete_or_relabelled_evidence_cannot_become_a_matching_size_contract(wrong):
    _,changed=prepared();contract=deepcopy(changed['size_workplaces'])
    entry=contract['regions']['shelf_2_left']['medium'];case=entry['TRAIN_cases']['confirmation'][0]
    if wrong=='same_seeds':contract['frozen_TRAIN_searches']['confirmation']=deepcopy(contract['frozen_TRAIN_searches']['discovery'])
    elif wrong=='lost_size':del contract['regions']['shelf_2_left']['medium']
    elif wrong=='fake_success':entry['template']['measured_success']=True
    elif wrong=='wrong_pool':case['actual_terminal']['target_pool_id']=case['actual_terminal']['target_logical_id']
    elif wrong=='wrong_offset':entry['template']['base_minus_initial_box_xy_rack_m'][0]+=.01
    elif wrong=='heldout':entry['template']['source_split']='holdout'
    elif wrong=='lost_denominator':entry['TRAIN_cases']['confirmation'].pop()
    elif wrong=='numerical_success':case.update(success=True,numerical_failure=True)
    elif wrong=='unsafe_unproven':case.update(unsafe=True);case['actual_terminal']['unsafe']=True
    else:del contract['frozen_TRAIN_searches']['confirmation']
    with pytest.raises(ValueError):validate_size_workplaces(contract)
