"""Summarize a closed TRAIN workplace search without inflating its sample size."""
from collections import Counter, defaultdict
from math import isfinite

from .cpu_workplace_probe import SOURCE, validate_cpu_workplace_probe
from .physics_backend_eval import REGIONS


def summarize_workplace_results(manifest, metrics):
    waves=manifest.get('layout_waves')
    frozen=metrics.get('CPU_workplace_probe') or {}
    size_probe=frozen.get('unmeasured_size_workplace_probe')
    validate_cpu_workplace_probe(waves,manifest,enabled=True,device=manifest.get('sim_device'),
        training=manifest.get('training'),steps=900,waypoint_enabled=True,explicit_frozen=True,
        unmeasured_size_probe=bool(size_probe))
    integrity=frozen.get('frozen_network_integrity') or {}
    learner=metrics.get('learner') or {}
    expected_tensors=178
    if learner.get('regional_actor') is not None:
        from .regional_actor_servo import regional_actor_contract
        if learner['regional_actor']!=regional_actor_contract():
            raise ValueError('Workplace result has an unknown regional actor contract')
        expected_tensors=198
    if size_probe:
        from .size_workplace_probe import validate_size_workplace_request
        if size_probe!=validate_size_workplace_request(waves):
            raise ValueError('Size workplace result differs from its original supported TRAIN cases')
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
            or integrity.get('compared_tensor_count')!=expected_tensors \
            or frozen.get('actor_critic_updates_and_replay_size_unchanged') is not True \
            or learner.get('training') is not False \
            or any(learner.get(k)!=initial[k] for k in initial):
        raise ValueError('Workplace results need every expected model tensor frozen and unchanged initial actor/Q/replay counters')
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
        if success and (invalid or result.get('unsafe') or result.get('invalid_reset') or result.get('numerical_failure')
                or any(result.get('unsafe_causes',{}).values()) or result.get('pinching')!=[True,True]
                or result.get('stable_hands')!=[True,True] or not result.get('opposing_flaps')
                or not result.get('proof_lift') or not isfinite(result.get('hold_time_s',0))
                or result.get('hold_time_s',0)<.25 or not isfinite(result.get('rack_clearance_m',0))
                or result.get('rack_clearance_m',0)<.008):
            raise ValueError('A claimed success must have actual safe bilateral contact, hold and proof lift')
        candidate=row['waypoint_probe'];stage=result.get('staged_base',{})
        if not invalid and stage.get('waypoint_probe')!=candidate:
            raise ValueError('Measured base target must retain its original named candidate')
        kind=row['layout'].get('target_box_type','small')
        if size_probe and not invalid and kind=='medium' and not result.get('numerical_failure'):
            from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
            from ..scene.spawn import logical_cells,physical_pool_id
            from ..spec import MultiBoxSpec
            expected_probe=stage.get('unmeasured_size_workplace_probe') or {}
            cells=logical_cells(MultiBoxSpec())
            logical=result.get('target_logical_id')
            if type(logical) is not int or not 0<=logical<len(cells):
                raise ValueError('Medium measurement is missing its actual target identity')
            expected_cell=cells[logical]
            if (expected_cell.region_name!=row['layout']['target_region'] or
                expected_probe.get('requested_box_type')!=kind or
                expected_probe.get('requested_box_size_m')!=list(BOX_DIMENSIONS_M[kind]) or
                expected_probe.get('Q_import_eligible') is not False or
                stage.get('template',{}).get('measured_success') is not False or
                result.get('target_pool_id')!=physical_pool_id(expected_cell,1)):
                raise ValueError('Medium measurement needs its actual supported asset and explicitly unmeasured waypoint')
        case=dict(environment=i,seed=row['layout']['seed'],region=row['layout']['target_region'],box_type=kind,
            candidate=candidate['name'],offset_xy_yaw=candidate['offset_xy_yaw'],success=success,
            unsafe=bool(result.get('unsafe')),initial_invalid=invalid,time_out=bool(result.get('time_out')),
            numerical_failure=bool(result.get('numerical_failure')),
            other_terminal=not any((success, invalid, result.get('unsafe'), result.get('time_out'),
                result.get('numerical_failure'))),
            actual_terminal=result)
        cases.append(case);groups[(case['region'],case['candidate'])].append(case)
    summaries={};promising={};typed_promising={}
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
                unsafe_causes=dict(causes),unique_TRAIN_seeds=sorted(x['seed'] for x in entries),
                by_box_type={kind:dict(requested=sum(x['box_type']==kind for x in entries),
                    **{key:sum(x[key] for x in entries if x['box_type']==kind) for key in counts},
                    unique_TRAIN_seeds=sorted(x['seed'] for x in entries if x['box_type']==kind))
                    for kind in sorted({x['box_type'] for x in entries})}))
        if len(summary)!=8:raise ValueError('Every region must include all eight candidates')
        summary.sort(key=lambda x:(-x['success'],x['unsafe'],x['initial_invalid'],
            sum(v*v for v in x['offset_xy_yaw']),x['candidate']))
        summaries[region]=summary
        promising[region]=summary[0] if summary[0]['success']>0 else None
        if size_probe:
            typed_promising[region]={}
            for kind in sorted({x['box_type'] for x in cases if x['region']==region}):
                candidates=[dict(candidate=x['candidate'],offset_xy_yaw=x['offset_xy_yaw'],
                    **x['by_box_type'][kind]) for x in summary if kind in x['by_box_type']]
                candidates.sort(key=lambda x:(-x['success'],x['unsafe'],x['initial_invalid'],
                    sum(v*v for v in x['offset_xy_yaw']),x['candidate']))
                typed_promising[region][kind]=candidates[0] if candidates[0]['success']>0 else None
            # Mixed-size totals must not be passed to the old region-only
            # waypoint builder as measured evidence for either box size.
            promising[region]=None
    return dict(unique_fresh_TRAIN_cases=16,unique_cases_per_region=4,candidates_per_case=8,
        physical_candidate_attempts=128,regions=summaries,promising_candidates_for_fresh_TRAIN_recheck=promising,
        all_regions_have_a_measured_success_candidate=all(v is not None for v in promising.values()),
        cases=sorted(cases,key=lambda x:x['environment']),
        all178_tensors_and_initial_counters_frozen=expected_tensors==178,
        all_expected_model_tensors_and_initial_counters_frozen=True,frozen_tensor_count=expected_tensors,
        unmeasured_size_workplace_probe=size_probe,
        promising_by_region_and_box_type_for_fresh_TRAIN_recheck=typed_promising,
        mixed_size_counts_NOT_region_only_training_evidence=bool(size_probe),
        frozen_source_actor_updates=initial['actor_updates'],frozen_source_critic_updates=initial['critic_updates'],
        failed_and_initial_invalid_requests_retained=True,not128_independent_cases=True,
        not_a_DEV_or_FINAL_generalization_score=True,flap_draws_and_contact_history_not_matched=True,
        selecting_candidates_does_not_train_or_populate_Q_replay=True,
        independent_FINAL_unused=True,goal_not_complete=True)
