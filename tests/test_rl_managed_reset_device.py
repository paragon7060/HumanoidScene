"""A CPU physics diagnostic cannot silently become training or FINAL data."""
import json

import pytest

from batched_staged_goal_with_drive import validate_managed_physics_device


def test_default_GPU_training_is_unchanged():
    assert validate_managed_physics_device('cuda:0', ['--training']) is None


def test_explicit_frozen_DEV_CPU_reset_keeps_Q_import_disabled(tmp_path):
    waves=tmp_path/'waves.json'
    waves.write_text(json.dumps([{'split':'validation','layouts':[]}]))
    result=validate_managed_physics_device('cpu',['--no-training','--reset-failure-diagnostics',
        '--steps=1','--waves-json',str(waves)])
    assert result['physics_device']=='cpu' and not result['Q_import_eligible']
    assert result['constructor_and_contact_solver_history_not_matched']


@pytest.mark.parametrize('flags',[
    ['--training','--reset-failure-diagnostics','--steps','1'],
    ['--no-training','--reset-failure-diagnostics','--steps','900'],
    ['--no-training','--steps','1'],
    ['--reset-failure-diagnostics','--steps','1'],
    ['--training','--no-training','--reset-failure-diagnostics','--steps','1'],
])
def test_CPU_refuses_implicit_or_training_and_full_policy_rollout(flags):
    with pytest.raises(ValueError,match='CPU dynamics requires explicit'):
        validate_managed_physics_device('cpu',flags)


@pytest.mark.parametrize('split',['train','holdout'])
def test_CPU_refuses_TRAIN_and_independent_FINAL_before_launch(tmp_path,split):
    waves=tmp_path/'waves.json'
    waves.write_text(json.dumps([{'split':split,'layouts':[]}]))
    with pytest.raises(ValueError,match='cannot collect TRAIN or independent FINAL'):
        validate_managed_physics_device('cpu',['--no-training','--reset-failure-diagnostics',
            '--steps','1','--waves-json',str(waves)])
