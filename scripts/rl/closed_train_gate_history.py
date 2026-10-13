"""Keep existing scalar TRAIN gate reports after the next wave resets them."""
from copy import deepcopy


def archive_train_gate_summary(history, wave_index, split, summary, outcomes):
    """Copy one existing wave report without querying the policy or simulator.

    Join ``summary.by_environment`` with the matching metrics outcomes using
    (wave, environment). Counts overlap and commands are not physical pinch.
    Incomplete waves retain their explicit completion status. DEV and FINAL
    never contribute to this TRAIN history.
    """
    if split != 'train' or summary is None:
        return
    if summary.get('name') != 'existing_perceived_contact_gates_current_wave_v1':
        raise ValueError('Unknown TRAIN gate report cannot be archived')
    if not outcomes or any(row['wave'] != wave_index or row['split'] != split
                           for row in outcomes):
        raise ValueError('TRAIN gate history requires matching wave outcomes')
    environments = {row['environment'] for row in outcomes}
    if len(environments) != len(outcomes):
        raise ValueError('TRAIN gate history requires distinct environment IDs')
    counts, rows = summary['counts'], summary['by_environment']
    ids = [row['environment'] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= environments:
        raise ValueError('TRAIN gate report has repeated or unknown environment IDs')
    for values in (counts, *(row['counts'] for row in rows)):
        if values.keys() != counts.keys() or any(type(v) is not int or v < 0
                                                for v in values.values()):
            raise ValueError('TRAIN gate counts must be matching nonnegative integers')
    if any(counts[name] != sum(row['counts'][name] for row in rows)
           for name in counts):
        raise ValueError('TRAIN gate aggregate differs from its environment counts')
    record = dict(wave=wave_index, split=split,
        complete=all(row['complete'] for row in outcomes),
        recorded_environment_count=len(outcomes), summary=deepcopy(summary),
        outcome_join_keys=['wave', 'environment'],
        counts_NOT_exclusive_failure_causes_or_physical_pinch=True,
        no_extra_policy_report_sensor_transfer_or_optimizer_input=True)
    existing = [row for row in history if row['wave'] == wave_index]
    if existing:
        if len(existing) != 1 or existing[0] != record:
            raise ValueError('Closed TRAIN gate history is immutable within a run')
        return
    history.append(record)
