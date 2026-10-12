"""Saved terminal policies must use the same CPU PGS/CUDA frozen DEV route."""
from copy import deepcopy
import json

import pytest

from batched_staged_goal_with_drive import validate_managed_physics_device
from kuavo_isaaclab_scene.rl.multi_box.experiments.cpu_physics_training import (
    validate_cpu_physics_policy_eval, validate_cpu_physics_training)
from kuavo_isaaclab_scene.rl.multi_box.experiments.physics_backend_eval import REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_physics import (
    CPU_PHYSICS_BACKEND, staged_solver_contract)


def inputs():
    waves = [dict(split='validation', layouts=[dict(episode_index=i % 2,
        layout=dict(seed=950000 + i, split='holdout', target_region=REGIONS[i % 4]))
        for i in range(128)])]
    contract = dict(physics_dynamics=staged_solver_contract('PGS',
        physics_backend=CPU_PHYSICS_BACKEND), reward_profile={'unchanged': 1},
        terminal_contract={'unchanged': True})
    return waves, contract


def validate(waves, contract, **options):
    return validate_cpu_physics_policy_eval(waves, contract, **(
        dict(enabled=True, physics_device='cpu', learner_device='cuda:0',
             training=False, steps=900, explicit_frozen=True) | options))


def test_same_physics_cuda_inference_has_full_DEV_and_no_learning_permission():
    waves, contract = inputs()
    before = deepcopy((waves, contract))
    result = validate(waves, contract)
    assert (waves, contract) == before
    assert result['physics_backend'] == CPU_PHYSICS_BACKEND
    assert result['physics_device'] == 'cpu' and result['learner_device'] == 'cuda:0'
    assert result['original_DEV_requested'] == 128 and result['requested_per_region'] == 32
    assert result['same_recorded_physics_contract'] and result['frozen_Q_inference_only']
    assert not any(result[k] for k in ('training', 'Q_import_eligible',
        'optimizer_updates_allowed', 'replay_import_allowed',
        'success_candidate_training_resume_allowed', 'independent_FINAL_used'))
    # Unchanged requests do not assert identical physical reset realizations.
    assert result['constructor_and_contact_solver_history_not_matched']


@pytest.mark.parametrize('options', [
    {'training': True}, {'explicit_frozen': False}, {'steps': 1},
    {'physics_device': 'cuda:0'}, {'learner_device': 'cpu'}, {'other_probe': True},
])
def test_wrong_devices_implicit_training_partial_or_modified_physics_rejected(options):
    with pytest.raises(ValueError):
        validate(*inputs(), **options)


@pytest.mark.parametrize('change', ['TRAIN', 'FINAL', 'short', 'duplicate', 'unbalanced', 'packed'])
def test_training_final_or_changed_full_distribution_is_rejected(change):
    waves, contract = inputs()
    if change == 'TRAIN':
        waves[0]['split'] = 'train'
    elif change == 'FINAL':
        waves[0]['split'] = 'holdout'
    elif change == 'short':
        waves[0]['layouts'].pop()
    elif change == 'duplicate':
        waves[0]['layouts'][1]['layout']['seed'] = 950000
    elif change == 'unbalanced':
        waves[0]['layouts'][0]['layout']['target_region'] = REGIONS[1]
    else:
        waves[0]['background_placement'] = 'packed'
    with pytest.raises(ValueError):
        validate(waves, contract)


@pytest.mark.parametrize('dynamics', [staged_solver_contract('PGS'),
    staged_solver_contract('TGS', physics_backend=CPU_PHYSICS_BACKEND),
    staged_solver_contract('PGS', physics_backend=CPU_PHYSICS_BACKEND) | {'control_dt_s': .01}])
def test_saved_dynamics_identity_cannot_be_retagged_for_evaluation(dynamics):
    waves, contract = inputs()
    with pytest.raises(ValueError, match='unchanged CPU PGS contract'):
        validate(waves, contract | {'physics_dynamics': dynamics})


def test_manager_accepts_instrumented_frozen_eval_and_refuses_learning_controls(tmp_path):
    waves, contract = inputs()
    wp, mp = tmp_path/'waves.json', tmp_path/'manifest.json'
    wp.write_text(json.dumps(waves))
    mp.write_text(json.dumps(contract))
    flags = ['--cpu-physics-policy-eval', '--no-training', '--waves-json', str(wp),
             '--training-manifest', str(mp), '--eval-video-q-values', '--rack-contact-summary']
    result = validate_managed_physics_device('cpu', flags, learner_device='cuda:0')
    assert result['original_DEV_requested'] == 128 and not result['training']
    for flag in ('--cpu-physics-training', '--training', '--frozen-physics-backend-eval',
                 '--cpu-workplace-probe', '--reset-failure-diagnostics',
                 '--body-behavior=arm20-gentle-rest-greedy', '--measured-train-credit=one-step',
                 '--save-train-success-candidates', '--base-attitude-gain-probe=soft15_2'):
        with pytest.raises(ValueError):
            validate_managed_physics_device('cpu', flags + [flag], learner_device='cuda:0')
    with pytest.raises(ValueError):
        validate_managed_physics_device('cuda:0', flags, learner_device='cuda:0')
    with pytest.raises(ValueError):
        validate_cpu_physics_training(waves, contract, enabled=True, physics_device='cpu',
            learner_device='cuda:0', training=False, steps=900)


def test_disabled_option_does_not_change_legacy_routes():
    assert validate_cpu_physics_policy_eval([], {}, enabled=False, physics_device='cuda:0',
        learner_device='cuda:0', training=True, steps=1, explicit_frozen=False) is None
