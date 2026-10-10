#!/usr/bin/env python3
"""Read recorded staged-SAC progress without Isaac, GPU, or Drive access.

Only completed metric snapshots are summarized. This does not infer whether
the writer is alive, import evaluation data, or change a training run.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time


def read_snapshot(path):
    # The live runner currently writes metrics with write_text. A transient
    # partial JSON write is an observation failure, never a stopped job.
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            if attempt == 4:
                raise
            time.sleep(.2)


def distribution(values):
    values = sorted(v for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    if not values:
        return dict(count=0, mean=None, median=None, p90=None, maximum=None)
    def percentile(p):
        x = (len(values) - 1) * p
        lo = int(x)
        return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (x - lo)
    return dict(count=len(values), mean=sum(values) / len(values), median=percentile(.5),
                p90=percentile(.9), maximum=values[-1])


def finite_at_least(value, minimum):
    return type(value) in (int, float) and math.isfinite(value) and value >= minimum


def supported_success(outcome):
    result = outcome.get('result') or {}
    return bool(outcome.get('complete') and outcome.get('initial_layout_valid')
        and result.get('success') and not result.get('unsafe')
        and not result.get('numerical_failure') and not result.get('invalid_reset')
        and not any(result.get('unsafe_causes', {}).values())
        and result.get('pinching') == [True, True]
        and result.get('stable_hands') == [True, True]
        and result.get('opposing_flaps') and result.get('proof_lift')
        and finite_at_least(result.get('hold_time_s'), .25)
        and finite_at_least(result.get('rack_clearance_m'), .008))


def summarize_wave(rows):
    regions = defaultdict(Counter)
    causes = Counter()
    cause_sets = Counter()
    pinch = Counter()
    quality = []
    distances = [[], []]
    distances_by_outcome = {k: [[], []] for k in
        ('supported_success', 'safe_timeout', 'unsafe')}
    unsupported = []
    rack_terminal = defaultdict(lambda: dict(phases=Counter(), bodies=Counter(), forces=[]))
    for outcome in rows:
        result = outcome.get('result') or {}
        region = regions[outcome['layout']['target_region']]
        region['attempts'] += 1
        region['initial_valid'] += int(bool(outcome.get('initial_layout_valid')))
        region['complete'] += int(bool(outcome.get('complete')))
        region['task_success_flags'] += int(bool(result.get('success')))
        region['supported_successes'] += int(supported_success(outcome))
        if result.get('success') and not supported_success(outcome):
            unsupported.append(outcome['environment'])
        if not outcome.get('initial_layout_valid') or not outcome.get('complete'):
            continue
        flagged = sorted(k for k, v in result.get('unsafe_causes', {}).items() if v)
        causes.update(flagged)
        if flagged:
            cause_sets[' + '.join(flagged)] += 1
        region['unsafe'] += int(bool(result.get('unsafe')))
        region['timeout'] += int(bool(result.get('time_out')))
        if result.get('unsafe') and 'robot_rack_collision' in flagged:
            rack = rack_terminal[outcome['layout']['target_region']]
            stage = (result.get('staged_base') or {}).get('phase')
            body = result.get('rack_peak_body')
            rack['phases'][stage if isinstance(stage, str) else 'unknown'] += 1
            rack['bodies'][body if isinstance(body, str) else 'unknown'] += 1
            rack['forces'].append(result.get('rack_peak_force_n'))
        hands = result.get('pinching')
        if isinstance(hands, list) and len(hands) == 2:
            pinch[str(sum(bool(x) for x in hands))] += 1
        if 'contact_quality' in result:
            quality.append(result['contact_quality'])
        pair = result.get('flap_distances')
        if isinstance(pair, list) and len(pair) == 2:
            for i, value in enumerate(pair):
                distances[i].append(value)
                if supported_success(outcome):
                    distances_by_outcome['supported_success'][i].append(value)
                if result.get('time_out') and not result.get('unsafe'):
                    distances_by_outcome['safe_timeout'][i].append(value)
                if result.get('unsafe'):
                    distances_by_outcome['unsafe'][i].append(value)
    return dict(wave=rows[0]['wave'], split=rows[0]['split'],
        by_region={k:dict(v) for k, v in regions.items()},
        supported_successes=sum(v['supported_successes'] for v in regions.values()),
        attempts=len(rows), initial_valid=sum(v['initial_valid'] for v in regions.values()),
        unsupported_success_environments=unsupported,
        safety_causes_on_valid_completed_attempts=dict(causes),
        simultaneous_safety_cause_sets=dict(cause_sets),
        rack_collision_terminal_diagnostics=dict(
            by_region={k:dict(count=sum(v['phases'].values()),
                phase_counts=dict(v['phases']), peak_body_counts=dict(v['bodies']),
                peak_force_n=distribution(v['forces'])) for k,v in rack_terminal.items()},
            valid_completed_unsafe_attempts_only=True,
            failure_step_peak_body_not_first_contact_or_contact_history=True),
        end_pinching_hand_counts=dict(pinch),
        end_contact_quality=distribution(quality),
        end_matched_flap_surface_distance_m=[distribution(v) for v in distances],
        end_flap_surface_distance_by_outcome_m={
            k: [distribution(v) for v in pair] for k, pair in distances_by_outcome.items()},
        unsafe_terminal_distances_are_not_safe_approach_metrics=True,
        end_contact_statistics_are_not_ever_contact_rates=True)


def compare_development(reference, current):
    """Keep the full attempt denominator; common-valid pairs are diagnostic."""
    before = {row['environment']: row for row in reference}
    after = {row['environment']: row for row in current}
    unmatched = sorted(set(before) ^ set(after))
    mismatched = sorted(i for i in before.keys() & after.keys()
        if before[i]['layout'] != after[i]['layout'])
    regions = defaultdict(Counter)
    for i in before.keys() & after.keys():
        if i in mismatched:
            continue
        a, b = before[i], after[i]
        region = regions[a['layout']['target_region']]
        region['same_requested_layout_pairs'] += 1
        if not (a.get('initial_layout_valid') and b.get('initial_layout_valid')
                and a.get('complete') and b.get('complete')):
            continue
        region['common_valid_completed_pairs'] += 1
        sa, sb = supported_success(a), supported_success(b)
        region['baseline_successes'] += int(sa)
        region['current_successes'] += int(sb)
        region['lost_successes'] += int(sa and not sb)
        region['gained_successes'] += int(sb and not sa)
    return dict(reference_wave=reference[0]['wave'], current_wave=current[0]['wave'],
        unmatched_environments=unmatched, different_requested_layout_environments=mismatched,
        by_region={k: dict(v) for k, v in regions.items()},
        common_valid_comparison_is_diagnostic_not_the_primary_success_rate=True,
        repeated_reset_physics_are_not_asserted_identical=True)


def summarize(run):
    manifest = read_snapshot(run / 'manifest.json')
    metrics = read_snapshot(run / 'metrics.json')
    grouped = defaultdict(list)
    for outcome in metrics['outcomes']:
        grouped[outcome['wave']].append(outcome)
    learner = metrics.get('learner', {})
    profile = manifest.get('reward_profile', {})
    absorbing = profile.get('contact_shaping', {}).get('time_limit_is_absorbing_for_Q')
    inherited = manifest.get('terminal_contract', {}).get('timeouts_bootstrap')
    development = [grouped[w] for w in sorted(grouped)
        if grouped[w][0]['split'] == 'validation']
    progress_fields = ('actor_updates', 'critic_updates', 'online_rows', 'replay_size',
        'critic_warmup_remaining', 'actor_collection_warmup_remaining', 'prior_weight',
        'effective_prior_mse_weight', 'successful_train_bank', 'successful_train_replay_fraction',
        'latest', 'latest_actor', 'actor_success_guard_statistics', 'actor_success_guard_cohort')
    return dict(run_dir=str(run), recorded_writer_status=read_snapshot(run / 'status.json'),
        writer_liveness_not_inferred=True, last_recorded_collection_rows_all_splits=metrics.get('actual_rows'),
        collection_rows_include_DEV_and_base_approach=True,
        held_TRAIN_rows_total=learner.get('online_rows'), current_TRAIN_replay_rows=learner.get('replay_size'),
        learner={k:learner[k] for k in progress_fields if k in learner},
        effective_Q_timeouts_bootstrap=False if absorbing else inherited,
        inherited_base_terminal_description_timeouts_bootstrap=inherited,
        profile_rule_overrides_inherited_timeout_description=bool(absorbing),
        recorded_FINAL_waves=sum(w['split'] == 'holdout' for w in
            (rows[0] for rows in grouped.values())),
        waves=[summarize_wave(grouped[w]) for w in sorted(grouped)],
        paired_DEV_against_initial=[compare_development(development[0], rows)
            for rows in development[1:]],
        development_checks=metrics.get('development_checks', []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = summarize(args.run_dir.expanduser().resolve())
    text = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        print(text, end='')


if __name__ == '__main__':
    main()
