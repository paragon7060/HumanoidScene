"""Size-specific base targets from discovery and independent frozen TRAIN recheck."""
from collections import Counter
from copy import deepcopy
import math
import re

from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from ..spec import DEFAULT_RACK_REGIONS
from ..scene.spawn import logical_cells,physical_pool_id
from ..spec import MultiBoxSpec
from .region_workplaces import PERCEIVED_REGION_NAMES, validate_region_workplaces

FORMAT='TRAIN_measured_supported_size_workplaces_v1'
KINDS={region.name:tuple(k for k in ('small','medium') if k in region.allowed_box_types)
       for region in DEFAULT_RACK_REGIONS}
TERMINALS=('success','unsafe','initial_invalid','time_out','numerical_failure','other_terminal')
CELLS=logical_cells(MultiBoxSpec())


def _sha(value):
    return isinstance(value,str) and re.fullmatch(r'[0-9a-f]{64}',value) is not None


def _outcome(case):
    result=case.get('actual_terminal',{})
    if (not isinstance(result,dict) or any(type(case.get(k)) is not bool for k in TERMINALS)
        or any(case[k]!=bool(result.get(k)) for k in ('success','unsafe','numerical_failure','time_out'))):
        raise ValueError('Typed TRAIN flags must match the recorded physical terminal')
    success=bool(case.get('success'))
    if success and (case.get('initial_invalid') or result.get('unsafe') or result.get('numerical_failure')
        or result.get('invalid_reset') or any(result.get('unsafe_causes',{}).values())
        or result.get('pinching')!=[True,True] or result.get('stable_hands')!=[True,True]
        or not result.get('opposing_flaps') or not result.get('proof_lift')
        or type(result.get('hold_time_s')) not in (int,float) or not math.isfinite(result['hold_time_s'])
        or result['hold_time_s']<.25 or type(result.get('rack_clearance_m')) not in (int,float)
        or not math.isfinite(result['rack_clearance_m']) or result['rack_clearance_m']<.008):
        raise ValueError('A measured size waypoint needs actual safe bilateral success')
    if not case['initial_invalid'] and not case['numerical_failure']:
        logical=result.get('target_logical_id');kind=case['box_type']
        if (type(logical) is not int or not 0<=logical<len(CELLS)
            or CELLS[logical].region_name!=case['region']
            or result.get('target_pool_id')!=physical_pool_id(CELLS[logical],('small','medium').index(kind))):
            raise ValueError('Typed evidence must use the actual matching supported asset pool')
    # Numerical/invalid failures must remain in the denominator without being
    # counted twice if their low-level safety flags also became true.
    for key in ('initial_invalid','numerical_failure','unsafe','success','time_out'):
        if case.get(key):return key
    return 'other_terminal'


def _evidence(cases):
    counts=Counter(_outcome(case) for case in cases)
    return dict(requested=len(cases),**{k:counts[k] for k in TERMINALS},
                unique_TRAIN_seeds=sorted(case['seed'] for case in cases))


def validate_size_workplaces(contract):
    source=contract.get('source_region_workplaces') or {}
    validate_region_workplaces(source)
    if (contract.get('name')!=FORMAT or contract.get('source_shelf_templates')!=source['source_shelf_templates']
        or contract.get('perceived_region_names_by_id')!=list(PERCEIVED_REGION_NAMES)
        or set(contract.get('regions',{}))!=set(KINDS)
        or contract.get('source_Q_replay_rows_imported')!=0
        or contract.get('fresh_matching_Q_replay_required') is not True
        or contract.get('independent_FINAL_used') is not False
        or not _sha(contract.get('source_checkpoint_SHA256'))):
        raise ValueError('Supported size targets require their original sources and fresh Q/replay')
    searches=contract.get('frozen_TRAIN_searches',{})
    if set(searches)!={'discovery','confirmation'}:
        raise ValueError('Size targets need both discovery and a fresh TRAIN confirmation')
    all_seeds=[]
    for info in searches.values():
        seeds=info.get('unique_TRAIN_seeds',[])
        if (not _sha(info.get('results_SHA256')) or info.get('requested')!=128
            or info.get('frozen_tensor_count') not in (178,198) or len(seeds)!=16
            or any(type(seed) is not int for seed in seeds) or len(set(seeds))!=16
            or info.get('all_model_and_normalizer_tensors_frozen') is not True):
            raise ValueError('Both searches must retain complete frozen TRAIN16 x8 evidence')
        all_seeds.extend(seeds)
    if len(set(all_seeds))!=32:
        raise ValueError('Discovery and confirmation TRAIN seeds must be disjoint')
    for region,kinds in KINDS.items():
        variants=contract['regions'][region]
        if set(variants)!=set(kinds):raise ValueError('All six supported region/size targets are required')
        original=source['regions'][region]['template']
        for kind,entry in variants.items():
            cases=entry.get('TRAIN_cases',{});offset=entry.get('offset_xy_yaw',[])
            if (set(cases)!=set(searches) or len(offset)!=3
                or any(type(v) not in (int,float) or not math.isfinite(v) for v in offset)
                or max(abs(offset[0]),abs(offset[1]))>.15 or abs(offset[2])>math.pi/12):
                raise ValueError('A size candidate needs its original bounded measured offset')
            combined=[]
            for name,rows in cases.items():
                count=2 if region.startswith('shelf_2') else 4
                if (not isinstance(rows,list) or len(rows)!=count
                    or len({r.get('seed') for r in rows})!=count
                    or any(r.get('region')!=region or r.get('box_type')!=kind
                        or r.get('candidate')!=entry.get('candidate') or r.get('offset_xy_yaw')!=offset
                        or r.get('seed') not in searches[name]['unique_TRAIN_seeds'] for r in rows)):
                    raise ValueError('Size evidence must retain every original typed TRAIN request')
                for case in rows:
                    if case['initial_invalid'] or case['numerical_failure']:continue
                    stage=case['actual_terminal'].get('staged_base',{})
                    measured=stage.get('template',{})
                    if (measured.get('base_minus_initial_box_xy_rack_m')!=original['base_minus_initial_box_xy_rack_m']
                        or measured.get('base_yaw_rack_rad')!=original['base_yaw_rack_rad']
                        or any(abs(a-b)>1e-5 for a,b in zip(measured.get('box_size_m',[]),BOX_DIMENSIONS_M[kind]))
                        or len(measured.get('box_size_m',[]))!=3
                        or stage.get('waypoint_probe')!=dict(name=entry['candidate'],offset_xy_yaw=offset)
                        or not math.isclose(stage.get('base_target_yaw_rack_rad',float('nan')),
                            original['base_yaw_rack_rad']+offset[2],abs_tol=1e-6)):
                        raise ValueError('Measured targets must retain their actual original waypoint and candidate')
                    if kind=='medium' and (measured.get('measured_success') is not False
                        or stage.get('unmeasured_size_workplace_probe',{}).get('measured_source_template')!=original):
                        raise ValueError('Medium discovery must retain its explicitly unmeasured source')
                combined.extend(rows)
            evidence=_evidence(combined)
            confirmed=_evidence(cases['confirmation'])
            successful=evidence['success']>0 and confirmed['success']>0
            safe=(evidence['unsafe']==evidence['numerical_failure']==evidence['other_terminal']==0
                  and evidence['requested']-evidence['initial_invalid']>=3
                  and confirmed['requested']-confirmed['initial_invalid']>=1)
            if not successful and not safe:
                raise ValueError('A size target requires fresh confirmed success or measured safe attempts')
            template=entry.get('template',{})
            if (entry.get('TRAIN_evidence')!=evidence or entry.get('fresh_confirmation_evidence')!=confirmed
                or entry.get('unproven_grasp_candidate') is not (not successful)
                or template.get('source_split')!='train' or template.get('measured_success') is not successful
                or template.get('box_size_m')!=list(BOX_DIMENSIONS_M[kind])
                or template.get('base_minus_initial_box_xy_rack_m')!=[
                    original['base_minus_initial_box_xy_rack_m'][i]+offset[i] for i in range(2)]
                or template.get('base_yaw_rack_rad')!=original['base_yaw_rack_rad']+offset[2]):
                raise ValueError('Size target geometry and success label differ from actual TRAIN evidence')
    return contract


def build_size_workplaces(waypoints,discovery,confirmation,selections,*,results_SHA256,checkpoint_SHA256):
    """Summaries must first come from summarize_workplace_results on closed runs."""
    regional=waypoints.get('region_workplaces') or {}
    validate_region_workplaces(regional)
    if set(selections)!=set(KINDS) or set(results_SHA256)!={'discovery','confirmation'}:
        raise ValueError('Explicit selections for all six supported size targets are required')
    searches={};cases_by_search={}
    for name,result in (('discovery',discovery),('confirmation',confirmation)):
        if (not result.get('unmeasured_size_workplace_probe')
            or not result.get('all_expected_model_tensors_and_initial_counters_frozen')
            or result.get('physical_candidate_attempts')!=128
            or not result.get('failed_and_initial_invalid_requests_retained')
            or not result.get('not_a_DEV_or_FINAL_generalization_score')):
            raise ValueError('Size targets require complete separately validated frozen TRAIN searches')
        rows=deepcopy(result['cases'])
        if len(rows)!=128 or {r['environment'] for r in rows}!=set(range(128)):
            raise ValueError('Every original candidate attempt must remain present')
        seeds=sorted({r['seed'] for r in rows})
        searches[name]=dict(results_SHA256=results_SHA256[name],requested=128,
            unique_TRAIN_seeds=seeds,frozen_tensor_count=result['frozen_tensor_count'],
            all_model_and_normalizer_tensors_frozen=True)
        cases_by_search[name]=rows
    regions={}
    for region,kinds in KINDS.items():
        if set(selections[region])!=set(kinds):raise ValueError('Every supported box size needs an explicit choice')
        regions[region]={}
        for kind in kinds:
            candidate=selections[region][kind]
            cases={name:[r for r in rows if (r['region'],r['box_type'],r['candidate'])==(region,kind,candidate)]
                   for name,rows in cases_by_search.items()}
            if not cases['discovery'] or not cases['confirmation']:
                raise ValueError('Selected candidate must have discovery and fresh confirmation outcomes')
            offset=cases['discovery'][0]['offset_xy_yaw']
            original=regional['regions'][region]['template']
            evidence=_evidence(cases['discovery']+cases['confirmation']);confirmed=_evidence(cases['confirmation'])
            successful=evidence['success']>0 and confirmed['success']>0
            regions[region][kind]=dict(candidate=candidate,offset_xy_yaw=deepcopy(offset),TRAIN_cases=cases,
                TRAIN_evidence=evidence,fresh_confirmation_evidence=confirmed,
                unproven_grasp_candidate=not successful,
                template=dict(source_split='train',measured_success=successful,box_size_m=list(BOX_DIMENSIONS_M[kind]),
                    base_minus_initial_box_xy_rack_m=[original['base_minus_initial_box_xy_rack_m'][i]+offset[i] for i in range(2)],
                    base_yaw_rack_rad=original['base_yaw_rack_rad']+offset[2]))
    contract=dict(name=FORMAT,source_region_workplaces=deepcopy(regional),
        source_shelf_templates=deepcopy(regional['source_shelf_templates']),regions=regions,
        perceived_region_names_by_id=list(PERCEIVED_REGION_NAMES),frozen_TRAIN_searches=searches,
        source_checkpoint_SHA256=checkpoint_SHA256,source_Q_replay_rows_imported=0,
        fresh_matching_Q_replay_required=True,independent_FINAL_used=False,
        all_supported_size_generalization_unproven=True,
        original_box_base_background_firm_flap_randomization_preserved=True)
    validate_size_workplaces(contract)
    result=deepcopy(waypoints);result.pop('region_workplaces')
    return result|dict(size_workplaces=contract)
