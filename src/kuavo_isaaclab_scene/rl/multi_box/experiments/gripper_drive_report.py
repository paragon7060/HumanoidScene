"""Read-only, paired reporting of completed frozen claw-drive waves."""
from collections import Counter
import json
import math


def _case(layout):
    return json.dumps(layout, sort_keys=True, separators=(',', ':'))


def _success(outcome):
    result = outcome.get('result') or {}
    if not result.get('success'):
        return False
    if (not outcome['initial_layout_valid'] or result.get('unsafe')
            or result.get('invalid_reset') or result.get('numerical_failure')
            or any(result.get('unsafe_causes', {}).values())
            or not all(result.get('pinching', [False]))
            or len(result.get('pinching', [])) != 2
            or not all(result.get('stable_hands', [False]))
            or len(result.get('stable_hands', [])) != 2
            or not result.get('opposing_flaps') or not result.get('proof_lift')
            or not math.isfinite(result.get('hold_time_s', math.nan))
            or result['hold_time_s'] < .25
            or not math.isfinite(result.get('rack_clearance_m', math.nan))
            or result['rack_clearance_m'] < .008):
        raise ValueError('Recorded success contradicts physical proof or initial-layout validity')
    return True


def _summary(outcomes):
    valid = [o for o in outcomes if o['initial_layout_valid']]
    safe = [o for o in valid if o.get('result') and not o['result'].get('unsafe')
            and not o['result'].get('invalid_reset') and not o['result'].get('numerical_failure')
            and not any(o['result'].get('unsafe_causes', {}).values())]
    success = [o for o in outcomes if _success(o)]
    distances = [max(o['result']['flap_distances']) for o in safe
                 if len(o['result'].get('flap_distances') or []) == 2
                 and all(math.isfinite(v) for v in o['result']['flap_distances'])]
    causes = Counter(k for o in valid for k, v in
                     (o.get('result') or {}).get('unsafe_causes', {}).items() if v)
    by_region = {}
    for o in outcomes:
        region = o['layout']['target_region']
        count = by_region.setdefault(region, dict(attempts=0, valid=0, successes=0))
        count['attempts'] += 1
        count['valid'] += int(o['initial_layout_valid'])
        count['successes'] += int(_success(o))
    return dict(attempts=len(outcomes), valid=len(valid), initial_invalid=len(outcomes)-len(valid),
        successes=len(success), success_rate=len(success)/len(outcomes), by_region=by_region,
        success_seeds=[o['layout']['seed'] for o in success], unsafe_causes=dict(causes),
        cause_counts_can_overlap=True,
        numerical_failures=sum(bool((o.get('result') or {}).get('numerical_failure')) for o in valid),
        safe_terminal_cases=len(safe), safe_terminal_pinch_counts=dict(Counter(
            str(sum(o['result'].get('pinching', []))) for o in safe)),
        safe_terminal_max_hand_distance_mean_m=(sum(distances)/len(distances) if distances else None),
        distance_cases=len(distances), unsafe_distances_excluded=True,
        full_denominator_includes_initial_invalid=True)


def _paired(reference, candidate):
    ref = {_case(o['layout']): o for o in reference}
    new = {_case(o['layout']): o for o in candidate}
    if ref.keys() != new.keys():
        raise ValueError('Drive comparisons require the identical full initial layout for each case')
    joint_valid = [k for k in ref if ref[k]['initial_layout_valid'] and new[k]['initial_layout_valid']]
    def cause(o, name):
        return bool((o.get('result') or {}).get('unsafe_causes', {}).get(name))
    changes = {}
    for name in ('robot_rack_collision', 'box_speed_limit', 'box_lift_limit', 'box_drop'):
        changes[name] = dict(joint_valid_cases=len(joint_valid),
            resolved=sum(cause(ref[k], name) and not cause(new[k], name) for k in joint_valid),
            introduced=sum(not cause(ref[k], name) and cause(new[k], name) for k in joint_valid))
    return dict(same_initial_cases=len(ref), joint_valid_cases=len(joint_valid),
        changed_initial_validity=sum(ref[k]['initial_layout_valid'] != new[k]['initial_layout_valid'] for k in ref),
        full_denominator_success_gains=sum(not _success(ref[k]) and _success(new[k]) for k in ref),
        full_denominator_success_losses=sum(_success(ref[k]) and not _success(new[k]) for k in ref),
        joint_valid_cause_changes=changes,
        paired_cause_changes_exclude_initial_invalid=True)


def summarize_gripper_drive_probe(manifest, metrics, audits, status):
    """Never treat an unfinished wave, altered actor, or reset replacement as proof.

    Descriptive counts only. A small DEV probe is not independent FINAL
    generalization, and an initial-validity change is a potential confound.
    """
    probe = manifest.get('contact_stability_probe') or {}
    if (manifest.get('training') is not False or probe.get('name') != 'four_claw_motor_drive_comparison'
            or probe.get('frozen_only') is not True or probe.get('Q_import_eligible') is not False):
        raise ValueError('Require the explicit frozen-only claw motor-drive manifest')
    waves = manifest['layout_waves']
    if [w.get('gripper_drive_probe') for w in waves] != ['original', 'soft_2nm', 'original']:
        raise ValueError('Require predeclared original, soft_2nm, original waves')
    if any(w.get('split') != 'validation' for w in waves):
        raise ValueError('This descriptive probe uses DEV only, never TRAIN or FINAL')
    expected = [_case(row['layout']) for row in waves[0]['layouts']]
    expected_sources = [_case(row) for row in waves[0]['layouts']]
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('Distinct initial cases are required')
    if any([_case(row) for row in w['layouts']] != expected_sources for w in waves):
        raise ValueError('All candidates must have the identical ordered initial cases')
    learner = metrics['learner']
    if learner.get('training') is not False or learner.get('online_rows') != 0 or learner.get('replay_size') != 0:
        raise ValueError('Frozen diagnostic must not train or import Q replay')
    completed = status['completed_waves']
    if status.get('interrupted') or not isinstance(completed, int) or not 1 <= completed <= 3:
        raise ValueError('Require at least one fully completed, uninterrupted wave')
    grouped = {w: [] for w in range(completed)}
    for o in metrics['outcomes']:
        if o['wave'] not in grouped or o['split'] != 'validation':
            raise ValueError('Metrics contain an incomplete or undeclared wave')
        grouped[o['wave']].append(o)
    audit_map = {a['wave']: a for a in audits}
    originals = None
    reports = []
    for index, out in grouped.items():
        if sorted(_case(o['layout']) for o in out) != sorted(expected):
            raise ValueError('Each completed wave must retain every requested case, including initial failures')
        audit = audit_map[index]
        name = waves[index]['gripper_drive_probe']
        if audit['name'] != name or audit.get('frozen_only') is not True or audit.get('Q_import_eligible') is not False:
            raise ValueError('Runtime audit differs from the predeclared candidate')
        actual = audit['initialized_PhysX']
        if name == 'soft_2nm':
            if actual != {k: dict(min=v, max=v) for k, v in
                          dict(stiffness=100., damping=5., effort_limit=2.).items()}:
                raise ValueError('Soft drive is not confirmed by initialized PhysX')
        elif originals is None:
            originals = actual
        elif actual != originals:
            raise ValueError('Final original wave did not restore initialized motor drives')
        reports.append(dict(wave=index, candidate=name, initialized_PhysX=actual, **_summary(out)))
    paired = {}
    if completed >= 2:
        paired['soft_vs_first_original'] = _paired(grouped[0], grouped[1])
    if completed == 3:
        paired['original_repeat_vs_first_original'] = _paired(grouped[0], grouped[2])
        paired['soft_vs_original_repeat'] = _paired(grouped[2], grouped[1])
    return dict(artifact_type='frozen_gripper_drive_descriptive_report_v1',
        completed_waves=completed, comparison_complete=completed == 3,
        distinct_DEV_cases=len(expected), waves=reports, paired=paired,
        actor_updates=learner['actor_updates'], critic_updates=learner['critic_updates'],
        training_updates=0, Q_import_eligible=False, independent_FINAL_used=False,
        cause_not_proven_by_small_probe=True,
        limitation='Initial validity and original repeat variation must be considered; no automatic drive change.')
