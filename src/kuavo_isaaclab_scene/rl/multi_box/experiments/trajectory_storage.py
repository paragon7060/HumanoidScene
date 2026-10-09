"""Optional HDF retention for future runs; learning and outcomes stay complete."""
import math


def trajectory_storage_contract(mode):
    if mode not in ('all', 'successful-train'):
        raise ValueError('Unknown executed trajectory storage mode')
    return dict(mode=mode, applies_only_to_future_HDF_writes=True,
        all_requested_outcomes_and_actual_row_counts_preserved=True,
        all_evaluation_trajectories_preserved=True,
        failed_TRAIN_transitions_still_enter_unchanged_bounded_replay=True,
        rewards_optimizers_success_banks_and_physics_unchanged=True,
        full_failed_TRAIN_HDF_history_available=mode == 'all',
        original_full_TRAIN_phase_audit_supported=mode == 'all',
        raw_upload_or_existing_file_deletion=False)


def keep_trajectory(mode, outcome):
    trajectory_storage_contract(mode)
    if outcome['split'] not in ('train', 'validation', 'holdout'):
        raise ValueError('Unknown original wave split')
    if mode == 'all' or outcome['split'] != 'train':
        return True
    result = outcome.get('result') or {}
    if not result.get('success'):
        return False
    # Use the original physical TRAIN bank evidence, never a distance proxy.
    from .staged_train_success import validate_success_outcome
    validate_success_outcome('train', outcome)
    for field, minimum in (('hold_time_s', .25), ('rack_clearance_m', .008)):
        value = result[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
            raise ValueError('Successful HDF retention needs finite actual hold/lift evidence')
    return True


def record_completed_trajectory(recorder, samples, outcome, initial_state,
                                *, mode='all', collection_mode=None):
    """Record original rows or omit only a failed TRAIN HDF copy.

    The caller still reports every outcome and sends the original measured
    batches to learning. No sample, outcome or learner state is changed here.
    """
    if not keep_trajectory(mode, outcome) or not samples:
        return 0
    result = outcome.get('result') or {}
    recorder.start_episode(initial_state=initial_state)
    recorder.episode.attrs['wave'] = outcome['wave']
    recorder.episode.attrs['environment'] = outcome['environment']
    import json
    recorder.episode.attrs['layout_json'] = json.dumps(outcome['layout'], sort_keys=True)
    recorder.episode.attrs['initial_layout_guard_valid'] = bool(outcome['initial_layout_valid'])
    if collection_mode is not None:
        recorder.episode.attrs['collection_policy_mode'] = collection_mode
    recorder.append_many(samples)
    recorder.finish_episode(success=bool(result.get('success')),
        reason='numerical_failure_excluded_corrupt_row' if result.get('numerical_failure')
        else 'success' if result.get('success') else 'failure' if outcome['complete']
        else 'interrupted_or_step_limit')
    return len(samples)
