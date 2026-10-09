"""Real HDF retention must not drop evaluation paths or mutate TRAIN inputs."""
from copy import deepcopy
import json

import h5py
import numpy as np
import pytest

from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder
from kuavo_isaaclab_scene.rl.multi_box.experiments.trajectory_storage import (
    keep_trajectory, record_completed_trajectory, trajectory_storage_contract)
from test_rl_transition_recorder import _sample


def outcome(split='train', success=False):
    return dict(wave=1, split=split, environment=7,
        layout=dict(seed=45, target_region='shelf_2_right'), initial_layout_valid=True,
        complete=True, result=dict(success=success, unsafe=False, invalid_reset=False,
            pinching=[success, success], stable_hands=[success, success],
            opposing_flaps=success, proof_lift=success, hold_time_s=.267,
            rack_clearance_m=.012, staged_base=dict(phase='held_grasp', manipulation_start=1)))


@pytest.mark.parametrize('mode,expected', [('all', 4), ('successful-train', 3)])
def test_real_hdf_omits_only_failure_copy_and_preserves_all_input_outcomes(tmp_path, mode, expected):
    attempts = [outcome('train', False), outcome('train', True),
                outcome('validation', False), outcome('holdout', False)]
    rows = [_sample(0), _sample(1, terminal=True)]
    original_outcomes, original_rows = deepcopy(attempts), deepcopy(rows)
    path = tmp_path / 'executed.hdf5'
    recorder = RlTransitionRecorder(path, {'action_dim': 1,
        'executed_trajectory_storage': trajectory_storage_contract(mode)})
    counts = [record_completed_trajectory(recorder, rows, attempt, {'q': np.array([.1])},
        mode=mode, collection_mode='actual_fixture') for attempt in attempts]
    recorder.close()
    assert counts == ([2] * 4 if mode == 'all' else [0, 2, 2, 2])
    assert attempts == original_outcomes
    for current, original in zip(rows, original_rows):
        for key in current: np.testing.assert_array_equal(current[key], original[key])
    with h5py.File(path) as file:
        assert len(file['episodes']) == expected
        meta = json.loads(file.attrs['manifest_json'])['executed_trajectory_storage']
        assert meta['full_failed_TRAIN_HDF_history_available'] == (mode == 'all')
        for episode in file['episodes'].values():
            assert episode.attrs['num_transitions'] == 2
            assert episode.attrs['collection_policy_mode'] == 'actual_fixture'
            assert json.loads(episode.attrs['layout_json']) == attempts[0]['layout']
            for key in rows[0]:
                np.testing.assert_array_equal(episode['transitions'][key][:],
                    np.asarray([row[key] for row in rows]))


@pytest.mark.parametrize('field,value', [
    ('unsafe', True), ('pinching', [True, False]), ('hold_time_s', float('nan')),
    ('rack_clearance_m', float('inf'))])
def test_sparse_storage_cannot_promote_incomplete_or_unsafe_success(field, value):
    attempted = outcome('train', True)
    attempted['result'][field] = value
    with pytest.raises(ValueError): keep_trajectory('successful-train', attempted)


def test_empty_invalid_attempt_and_interrupted_development_remain_metadata_only(tmp_path):
    recorder = RlTransitionRecorder(tmp_path / 'partial.hdf5', {})
    invalid = outcome(); invalid['initial_layout_valid'] = False
    assert record_completed_trajectory(recorder, [], invalid, None,
        mode='successful-train') == 0
    partial = outcome('validation'); partial['complete'] = False
    assert record_completed_trajectory(recorder, [_sample(0)], partial, None,
        mode='successful-train') == 1
    recorder.close()
    with h5py.File(tmp_path / 'partial.hdf5') as file:
        assert len(file['episodes']) == 1
        assert file['episodes/episode_000000'].attrs['end_reason'] == 'interrupted_or_step_limit'


def test_unknown_scope_rejected_before_recording():
    with pytest.raises(ValueError): trajectory_storage_contract('drop-all')
    attempted = outcome(); attempted['split'] = 'unknown'
    with pytest.raises(ValueError): keep_trajectory('all', attempted)
