from copy import deepcopy
import json
import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_workplace_probe import (
    INCOMPATIBLE_FLAGS, validate_cpu_workplace_probe)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
    CPU_PHYSICS_BACKEND, staged_solver_contract)
from batched_staged_goal_with_drive import validate_managed_physics_device


def request():
    rows=[]
    for region_index,region in enumerate(REGIONS):
        for case in range(4):
            for candidate in range(8):
                rows.append(dict(episode_index=region_index//2,layout=dict(
                    seed=700000+region_index*100+case,split='train',target_region=region,
                    base_lateral_m=.07*case),waypoint_probe=dict(
                    name=f'candidate{candidate}',offset_xy_yaw=[.01*candidate,0.,0.])))
    return [dict(split='train',layouts=rows)]


def check(waves, **changes):
    options=dict(enabled=True,device='cpu',training=False,steps=900,
        waypoint_enabled=True,explicit_frozen=True,other_probe=False)
    options.update(changes)
    return validate_cpu_workplace_probe(waves,dict(physics_dynamics=staged_solver_contract(
        'PGS',physics_backend=CPU_PHYSICS_BACKEND)),**options)


def test_frozen_train_search_preserves_all_requested_cases_and_blocks_replay():
    waves=request();before=deepcopy(waves);audit=check(waves)
    assert waves==before and audit['unique_TRAIN_layouts']==16
    assert audit['original_candidate_requests']==128 and audit['candidates_per_layout']==8
    assert audit['training'] is False and audit['Q_import_eligible'] is False
    assert audit['diagnostic_not_full_DEV_generalization_score']
    assert audit['flap_randomization_draws_and_contact_history_not_matched_between_candidates']
    assert '--base-waypoint-probe' not in INCOMPATIBLE_FLAGS
    assert '--cpu-workplace-probe' not in INCOMPATIBLE_FLAGS
    assert '--cpu-physics-training' in INCOMPATIBLE_FLAGS


@pytest.mark.parametrize('change',[dict(training=True),dict(device='cuda:0'),dict(explicit_frozen=False),
    dict(steps=1),dict(waypoint_enabled=False),dict(other_probe=True)])
def test_probe_cannot_silently_become_training_backend_eval_or_changed_physics(change):
    with pytest.raises(ValueError):check(request(),**change)


def test_eval_relabeling_changed_initial_states_or_unbalanced_candidate_matrix_rejected():
    for kind in ('DEV','FINAL','initial','candidate','region','offset','missing'):
        waves=request();row=waves[0]['layouts'][1]
        if kind=='DEV':waves[0]['split']='validation'
        elif kind=='FINAL':row['layout']['split']='holdout'
        elif kind=='initial':row['layout']['base_lateral_m']+=.01
        elif kind=='candidate':row['waypoint_probe']['name']='candidate0'
        elif kind=='region':row['layout']['target_region']='shelf_3_right'
        elif kind=='offset':row['waypoint_probe']['offset_xy_yaw'][1]=.151
        else:waves[0]['layouts'].pop()
        with pytest.raises(ValueError):check(waves)


def test_managed_cpu_route_has_the_same_freeze_and_data_split_gates(tmp_path):
    waves=tmp_path/'waves.json';waves.write_text(json.dumps(request()))
    manifest=tmp_path/'training_manifest.json';manifest.write_text(json.dumps(dict(
        physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))))
    child=['--cpu-workplace-probe','--base-waypoint-probe','--no-training',
        '--waves-json',str(waves),'--training-manifest',str(manifest)]
    assert validate_managed_physics_device('cpu',child)['Q_import_eligible'] is False
    for extra in (['--training'],['--cpu-physics-training'],['--frozen-physics-backend-eval']):
        with pytest.raises(ValueError):validate_managed_physics_device('cpu',child+extra)
