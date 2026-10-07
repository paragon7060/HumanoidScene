from copy import deepcopy
import json

import pytest
import torch

from test_rl_cpu_workplace_probe import request, check
from test_rl_staged_base_hold import scene, Coordinates
from test_rl_workplace_results import fixture
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_base_hold import StagedBaseHoldDiagnostic
from kuavo_isaaclab_scene.rl.multi_box.experiments.size_workplace_probe import validate_size_workplace_request, FORMAT
from kuavo_isaaclab_scene.rl.multi_box.experiments.workplace_results import summarize_workplace_results
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_workplaces import build_region_workplaces
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import regional_actor_contract
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M
from batched_staged_goal_with_drive import validate_managed_physics_device


def sized_request():
    waves=request()
    for row in waves[0]['layouts']:
        layout=row['layout'];middle=layout['target_region'].startswith('shelf_2')
        layout['lateral_m']=0.
        layout['target_box_type']='medium' if middle and layout['seed']%2 else 'small'
    return waves


def test_recipe_matrix_preparer_keeps_each_TRAIN_state_and_excludes_evaluation_rows():
    from prepare_size_workplace_probe import prepare
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import staged_solver_contract,CPU_PHYSICS_BACKEND
    originals={r['layout']['seed']:r for r in sized_request()[0]['layouts']}
    recipe=dict(records=[dict(group='train',reference_episode_index=r['episode_index'],layout=r['layout'])
        for r in originals.values()])
    recipe['records'].append(dict(group='development',reference_episode_index=0,
        layout=dict(seed=999999,split='holdout',lateral_m=0.,target_region='shelf_2_left',target_box_type='medium')))
    before=deepcopy(recipe)
    physical=dict(physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))
    waves,audit=prepare(recipe,physical)
    assert recipe==before and audit['unique_TRAIN_layouts']==16
    assert len(waves[0]['layouts'])==128 and 999999 not in {r['layout']['seed'] for r in waves[0]['layouts']}
    assert all(r['layout']==originals[r['layout']['seed']]['layout'] for r in waves[0]['layouts'])
    assert sum(r['layout']['target_box_type']=='medium' for r in waves[0]['layouts'])==32
    with pytest.raises(ValueError):prepare(recipe,physical,[])


def test_medium_geometry_is_explicitly_unmeasured_and_source_contract_never_relabelled():
    raw,templates=scene();raw[:,90]=1;raw[:,91:94]=torch.tensor(BOX_DIMENSIONS_M['medium'])
    original=deepcopy(templates)
    with pytest.raises(ValueError,match='box size'):
        StagedBaseHoldDiagnostic(Coordinates(),templates,raw)
    stage=StagedBaseHoldDiagnostic(Coordinates(),templates,raw,unmeasured_size_probe=True)
    assert templates==original and stage.templates==original['shelves']
    assert not stage.template['measured_success']
    assert stage.template['box_size_m']==list(BOX_DIMENSIONS_M['medium'])
    audit=stage.report()['unmeasured_size_workplace_probe']
    assert audit['name']==FORMAT and not audit['Q_import_eligible']
    assert audit['measured_source_template']==original['shelves']['middle']
    assert stage.target_xy[0].tolist()==[0.,pytest.approx(.7)]


@pytest.mark.parametrize('wrong',['unknown_dimensions','upper_medium','heldout_source'])
def test_diagnostic_flag_cannot_invent_assets_or_use_heldout_waypoints(wrong):
    raw,templates=scene();raw[:,90]=1;raw[:,91:94]=torch.tensor(BOX_DIMENSIONS_M['medium'])
    if wrong=='unknown_dimensions':raw[:,91]=.4
    elif wrong=='upper_medium':
        raw[:,94]=0;raw[:,96]=1;templates['shelves']['upper']=deepcopy(templates['shelves']['middle'])
    else:templates['shelves']['middle']['source_split']='holdout'
    with pytest.raises(ValueError):StagedBaseHoldDiagnostic(Coordinates(),templates,raw,unmeasured_size_probe=True)


def test_supported_new_size_search_is_frozen_TRAIN_only_and_never_matching_replay(tmp_path):
    waves=sized_request();before=deepcopy(waves)
    audit=check(waves,unmeasured_size_probe=True)
    assert waves==before and audit['unmeasured_size_workplace_probe']['frozen_only']
    assert audit['unmeasured_size_workplace_probe']['unique_TRAIN_cases_by_region_and_type']['shelf_2_left']=={'small':2,'medium':2}
    with pytest.raises(ValueError,match='explicit unmeasured'):check(waves)
    for options in (dict(training=True),dict(explicit_frozen=False),dict(waypoint_enabled=False),dict(device='cuda:0')):
        with pytest.raises(ValueError):check(waves,unmeasured_size_probe=True,**options)
    path=tmp_path/'waves.json';path.write_text(json.dumps(waves))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import staged_solver_contract,CPU_PHYSICS_BACKEND
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))))
    child=['--cpu-workplace-probe','--base-waypoint-probe','--unmeasured-size-workplace-probe','--no-training',
        '--waves-json',str(path),'--training-manifest',str(manifest)]
    assert not validate_managed_physics_device('cpu',child)['Q_import_eligible']
    for bad in (child+['--training'],[x for x in child if x!='--cpu-workplace-probe']):
        with pytest.raises(ValueError):validate_managed_physics_device('cpu',bad)


@pytest.mark.parametrize('wrong',['unknown','medium_upper','no_middle_medium','missing_size','wrong_reference'])
def test_size_scope_and_reference_identity_are_not_silently_changed(wrong):
    waves=sized_request();row=waves[0]['layouts'][0]
    if wrong=='unknown':row['layout']['target_box_type']='large'
    elif wrong=='medium_upper':
        row=next(x for x in waves[0]['layouts'] if x['layout']['target_region'].startswith('shelf_3'))
        row['layout']['target_box_type']='medium'
    elif wrong=='no_middle_medium':
        for row in waves[0]['layouts']:row['layout']['target_box_type']='small'
    elif wrong=='missing_size':row['layout'].pop('target_box_type')
    else:row['episode_index']=1
    with pytest.raises(ValueError):validate_size_workplace_request(waves)


def test_regional_probe_requires_all198_tensors_not_a_relaxed_minimum():
    manifest,metrics=fixture();metrics['learner']['regional_actor']=regional_actor_contract()
    with pytest.raises(ValueError,match='every expected'):summarize_workplace_results(manifest,metrics)
    metrics['CPU_workplace_probe']['frozen_network_integrity']['compared_tensor_count']=198
    result=summarize_workplace_results(manifest,metrics)
    assert result['all_expected_model_tensors_and_initial_counters_frozen']
    assert result['frozen_tensor_count']==198 and not result['all178_tensors_and_initial_counters_frozen']
    for count in (197,199):
        metrics['CPU_workplace_probe']['frozen_network_integrity']['compared_tensor_count']=count
        with pytest.raises(ValueError):summarize_workplace_results(manifest,metrics)


def test_mixed_size_counts_remain_separate_and_cannot_create_region_only_training_waypoints():
    manifest,metrics=fixture();manifest['layout_waves']=sized_request()
    metrics['CPU_workplace_probe']['unmeasured_size_workplace_probe']=validate_size_workplace_request(manifest['layout_waves'])
    for outcome,row in zip(metrics['outcomes'],manifest['layout_waves'][0]['layouts']):
        outcome['layout']=row['layout']
        if row['layout']['target_box_type']=='medium':
            raw,template=scene();raw[:,90]=1;raw[:,91:94]=torch.tensor(BOX_DIMENSIONS_M['medium'])
            if row['layout']['target_region']=='shelf_2_left':raw[:,94]=0;raw[:,95]=1
            stage=StagedBaseHoldDiagnostic(Coordinates(),template,raw,unmeasured_size_probe=True)
            report=stage.report();report['waypoint_probe']=row['waypoint_probe']
            logical=4 if row['layout']['target_region'].endswith('left') else 1
            outcome['result'].update(staged_base=report,target_logical_id=logical,target_pool_id=logical+6)
    result=summarize_workplace_results(manifest,metrics)
    assert result['mixed_size_counts_NOT_region_only_training_evidence']
    assert all(x is None for x in result['promising_candidates_for_fresh_TRAIN_recheck'].values())
    for entry in result['regions']['shelf_2_left']:
        assert entry['requested']==4
        assert entry['by_box_type']['small']['requested']==entry['by_box_type']['medium']['requested']==2
    with pytest.raises(ValueError,match='size-specific'):
        build_region_workplaces({},result,{},results_SHA256='a'*64)
    medium=next(x for x in metrics['outcomes'] if x['layout']['target_box_type']=='medium')
    medium['result']['target_pool_id']=medium['result']['target_logical_id']
    with pytest.raises(ValueError,match='actual supported asset'):
        summarize_workplace_results(manifest,metrics)
