"""Summarize a closed TRAIN workplace search without inflating its sample size."""
from collections import Counter, defaultdict
from math import isfinite

from .cpu_workplace_probe import SOURCE, validate_cpu_workplace_probe
from .physics_backend_eval import REGIONS


def summarize_workplace_results(manifest, metrics):
    waves=manifest.get('layout_waves')
    validate_cpu_workplace_probe(waves,manifest,enabled=True,device=manifest.get('sim_device'),
        training=manifest.get('training'),steps=900,waypoint_enabled=True,explicit_frozen=True)
    frozen=metrics.get('CPU_workplace_probe') or {}
    integrity=frozen.get('frozen_network_integrity') or {}
    learner=metrics.get('learner') or {}
    # Legacy probes used an actor-only initialization at counters0. Learned
    # checkpoints retain their real counters; the probe must not change them.
    initial=frozen.get('initial_learner_counters',dict(
        actor_updates=0,critic_updates=0,replay_size=0,online_rows=0))
    if not isinstance(initial,dict) or set(initial)!=set(('actor_updates','critic_updates','replay_size','online_rows')) \
            or any(type(initial.get(k)) is not int or initial[k]<0
            for k in ('actor_updates','critic_updates','replay_size','online_rows')) \
            or initial['replay_size']!=0 or initial['online_rows']!=0:
        raise ValueError('Frozen workplace search requires explicit nonnegative source counters and no imported replay')
    if frozen.get('source') != SOURCE or frozen.get('Q_import_eligible') is not False \
            or frozen.get('replay_rows_imported') != 0 \
            or not integrity.get('all_model_and_normalizer_tensors_bit_identical') \
            or integrity.get('compared_tensor_count')!=178 \
            or frozen.get('actor_critic_updates_and_replay_size_unchanged') is not True \
            or learner.get('training') is not False \
            or any(learner.get(k)!=initial[k] for k in initial):
        raise ValueError('Workplace results need frozen178 tensors and unchanged initial actor/Q/replay counters')
    outcomes=metrics.get('outcomes',[])
    if len(outcomes)!=128 or {r.get('environment') for r in outcomes}!=set(range(128)):
        raise ValueError('All128 original candidate requests must be present exactly once')
    rows=waves[0]['layouts'];groups=defaultdict(list);cases=[]
    for outcome in outcomes:
        i=outcome['environment'];row=rows[i];result=outcome.get('result') or {}
        if outcome.get('wave')!=0 or outcome.get('split')!='train' or outcome.get('layout')!=row['layout'] \
                or outcome.get('complete') is not True \
                or type(outcome.get('initial_layout_valid')) is not bool:
            raise ValueError('Workplace results require completed matching original TRAIN requests')
        success=bool(result.get('success'));invalid=not outcome.get('initial_layout_valid')
        if success and (invalid or result.get('unsafe') or result.get('invalid_reset')
                or any(result.get('unsafe_causes',{}).values()) or result.get('pinching')!=[True,True]
                or result.get('stable_hands')!=[True,True] or not result.get('opposing_flaps')
                or not result.get('proof_lift') or not isfinite(result.get('hold_time_s',0))
                or result.get('hold_time_s',0)<.25 or not isfinite(result.get('rack_clearance_m',0))
                or result.get('rack_clearance_m',0)<.008):
            raise ValueError('A claimed success must have actual safe bilateral contact, hold and proof lift')
        candidate=row['waypoint_probe'];stage=result.get('staged_base',{})
        if not invalid and stage.get('waypoint_probe')!=candidate:
            raise ValueError('Measured base target must retain its original named candidate')
        case=dict(environment=i,seed=row['layout']['seed'],region=row['layout']['target_region'],
            candidate=candidate['name'],offset_xy_yaw=candidate['offset_xy_yaw'],success=success,
            unsafe=bool(result.get('unsafe')),initial_invalid=invalid,time_out=bool(result.get('time_out')),
            numerical_failure=bool(result.get('numerical_failure')),
            other_terminal=not any((success, invalid, result.get('unsafe'), result.get('time_out'),
                result.get('numerical_failure'))),
            actual_terminal=result)
        cases.append(case);groups[(case['region'],case['candidate'])].append(case)
    summaries={};promising={}
    for region in REGIONS:
        summary=[]
        for (r,name),entries in groups.items():
            if r!=region:continue
            if len(entries)!=4:raise ValueError('Every region/candidate must keep its four original cases')
            counts={key:sum(x[key] for x in entries) for key in ('success','unsafe','initial_invalid',
                'time_out','numerical_failure','other_terminal')}
            causes=Counter(cause for x in entries for cause,active in
                x['actual_terminal'].get('unsafe_causes',{}).items() if active)
            summary.append(dict(candidate=name,requested=4,**counts,offset_xy_yaw=entries[0]['offset_xy_yaw'],
                unsafe_causes=dict(causes),unique_TRAIN_seeds=sorted(x['seed'] for x in entries)))
        if len(summary)!=8:raise ValueError('Every region must include all eight candidates')
        summary.sort(key=lambda x:(-x['success'],x['unsafe'],x['initial_invalid'],
            sum(v*v for v in x['offset_xy_yaw']),x['candidate']))
        summaries[region]=summary
        promising[region]=summary[0] if summary[0]['success']>0 else None
    return dict(unique_fresh_TRAIN_cases=16,unique_cases_per_region=4,candidates_per_case=8,
        physical_candidate_attempts=128,regions=summaries,promising_candidates_for_fresh_TRAIN_recheck=promising,
        all_regions_have_a_measured_success_candidate=all(v is not None for v in promising.values()),
        cases=sorted(cases,key=lambda x:x['environment']),all178_tensors_and_initial_counters_frozen=True,
        frozen_source_actor_updates=initial['actor_updates'],frozen_source_critic_updates=initial['critic_updates'],
        failed_and_initial_invalid_requests_retained=True,not128_independent_cases=True,
        not_a_DEV_or_FINAL_generalization_score=True,flap_draws_and_contact_history_not_matched=True,
        selecting_candidates_does_not_train_or_populate_Q_replay=True,
        independent_FINAL_unused=True,goal_not_complete=True)
