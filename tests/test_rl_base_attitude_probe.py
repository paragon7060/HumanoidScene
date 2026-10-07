import json
from types import SimpleNamespace as NS

import pytest

from kuavo_isaaclab_scene.rl.multi_box.experiments.base_attitude_probe import (
    ORIGINAL, configure_base_attitude_probe, validate_base_attitude_probe,
    verify_base_attitude_probe)
from batched_staged_goal_with_drive import validate_managed_physics_device
from prepare_size_workplaces import closed_probe
from test_rl_cpu_workplace_probe import check, request


def contract(**changes):
    options=dict(workplace=check(request()), training=False, num_envs=128)
    options.update(changes)
    return validate_base_attitude_probe('soft15_2', **options)


@pytest.mark.parametrize('change', [dict(training=True), dict(num_envs=8),
    dict(workplace=None), dict(workplace={'name':'frozen_DEV128'}),
    dict(workplace={'name':'CPU_PhysX_frozen_TRAIN_workplace_search_v1',
                    'original_candidate_requests':127,'training':False})])
def test_controller_probe_cannot_become_learning_dev_or_a_reduced_search(change):
    with pytest.raises(ValueError):contract(**change)


def test_configure_changes_only_three_declared_fields_and_actual_drive_is_checked():
    gains=NS(**ORIGINAL, yaw_stiffness=40.,height_stiffness=120.)
    cfg=NS(actions=NS(base=NS(dynamic=True,drive=gains)))
    c=contract();configure_base_attitude_probe(cfg,c)
    assert vars(gains)==dict(tilt_stiffness=15.,tilt_damping=2.,max_tilt_acceleration=10.,
                            yaw_stiffness=40.,height_stiffness=120.)
    drive=NS(cfg=gains,_asset=NS(is_fixed_base=False))
    env=NS(action_manager=NS(get_term=lambda name:NS(_drive=drive)))
    actual=verify_base_attitude_probe(env,c)
    assert actual['actual_runtime_gains_verified'] and not actual['Q_import_eligible']
    gains.tilt_damping=22.
    with pytest.raises(ValueError,match='Actual drive'):verify_base_attitude_probe(env,c)


def test_default_is_unchanged_and_other_controllers_cannot_be_combined():
    assert validate_base_attitude_probe(None,workplace=None,training=True,num_envs=8) is None
    configure_base_attitude_probe(None,None)
    assert verify_base_attitude_probe(None,None) is None
    for dynamic,gains in [(False,NS(**ORIGINAL)),(True,NS(**(ORIGINAL|dict(tilt_stiffness=60.))))]:
        with pytest.raises(ValueError):configure_base_attitude_probe(NS(actions=NS(base=NS(dynamic=dynamic,drive=gains))),contract())


def test_manager_rejects_training_and_non_workplace_use(tmp_path):
    waves=tmp_path/'waves.json';waves.write_text(json.dumps(request()))
    from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
        CPU_PHYSICS_BACKEND,staged_solver_contract)
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(
        physics_dynamics=staged_solver_contract('PGS',physics_backend=CPU_PHYSICS_BACKEND))))
    flags=['--base-attitude-gain-probe','soft15_2']
    child=['--cpu-workplace-probe','--base-waypoint-probe','--no-training',
        '--waves-json',str(waves),'--training-manifest',str(manifest),*flags]
    assert not validate_managed_physics_device('cpu',child)['Q_import_eligible']
    for changed in (child+['--training'],flags):
        with pytest.raises(ValueError):validate_managed_physics_device('cpu',changed)
    with pytest.raises(ValueError):validate_managed_physics_device('cuda:0',flags)


def test_altered_gain_results_cannot_supply_original_waypoint_or_replay_contract(tmp_path):
    run=tmp_path/'run';run.mkdir()
    (tmp_path/'launch.json').write_text(json.dumps(dict(run=str(run),command=[
        '--checkpoint',str(tmp_path/'checkpoint.pt'),'--waypoints',str(tmp_path/'waypoints.json')])))
    (tmp_path/'status.json').write_text(json.dumps(dict(training_exit_code=0,training_pid=999999999)))
    (run/'status.json').write_text(json.dumps(dict(status='complete')))
    c=contract()|dict(actual_runtime_gains=ORIGINAL,actual_runtime_gains_verified=True)
    (run/'manifest.json').write_text(json.dumps(dict(training=False,CPU_workplace_probe=check(request()),base_attitude_gain_probe=c)))
    with pytest.raises(ValueError,match='Changed controller'):closed_probe(run)
    with pytest.raises(ValueError,match='verified runtime gains'):closed_probe(run,allow_frozen_base_attitude_probe=True)
