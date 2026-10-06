"""TRAIN-derived region targets, including explicitly unsuccessful safe candidates."""
from copy import deepcopy
import math
import re

from .physics_backend_eval import REGIONS

FORMAT='TRAIN_measured_region_workplace_candidates_v1'


def validate_region_workplaces(contract):
    if contract.get('name')!=FORMAT or set(contract.get('regions',{}))!=set(REGIONS) \
            or set(contract.get('source_shelf_templates',{}))!={'middle','upper'} \
            or not re.fullmatch(r'[0-9a-f]{64}',contract.get('source_results_SHA256','')) \
            or contract.get('source_unique_TRAIN_cases')!=16 \
            or contract.get('source_candidate_requests')!=128 \
            or contract.get('source_all178_tensors_and_initial_counters_frozen') is not True \
            or contract.get('source_Q_replay_rows_imported')!=0:
        raise ValueError('Regional workplaces require the complete frozen TRAIN matrix and original templates')
    for region,entry in contract['regions'].items():
        source=contract['source_shelf_templates']['upper' if region.startswith('shelf_3') else 'middle']
        evidence=entry.get('TRAIN_evidence',{});template=entry.get('template',{})
        keys=('success','unsafe','initial_invalid','time_out','numerical_failure','other_terminal')
        counts=[evidence.get(k) for k in keys]
        seeds=evidence.get('unique_TRAIN_seeds',[]);offset=evidence.get('offset_xy_yaw',[])
        if source.get('source_split')!='train' or source.get('measured_success') is not True \
                or evidence.get('requested')!=4 or len(seeds)!=4 or len(set(seeds))!=4 \
                or any(type(v) is not int or v<0 or v>4 for v in counts) or sum(counts)!=4 \
                or len(offset)!=3 or any(type(v) not in (int,float) or not math.isfinite(v) for v in offset) \
                or max(abs(offset[0]),abs(offset[1]))>.15 or abs(offset[2])>math.pi/12:
            raise ValueError('Every regional candidate must retain all four original TRAIN outcomes')
        successful=evidence['success']>0
        safe_candidate=evidence['unsafe']==0 and evidence['initial_invalid']<=1 \
            and evidence['numerical_failure']==evidence['other_terminal']==0
        if not successful and not safe_candidate:
            raise ValueError('An unsuccessful candidate requires at least three valid attempts without safety failures')
        expected=[source['base_minus_initial_box_xy_rack_m'][i]+offset[i] for i in range(2)]
        if template.get('source_split')!='train' or template.get('measured_success') is not successful \
                or template.get('box_size_m')!=source['box_size_m'] \
                or template.get('base_minus_initial_box_xy_rack_m')!=expected \
                or template.get('base_yaw_rack_rad')!=source['base_yaw_rack_rad']+offset[2] \
                or entry.get('unproven_grasp_candidate') is not (not successful):
            raise ValueError('Regional target geometry and success label must match its original TRAIN evidence')
    return contract


def build_region_workplaces(waypoints, results, selections, *, results_SHA256):
    if set(selections)!=set(REGIONS) or not results.get('all178_tensors_and_initial_counters_frozen') \
            or not results.get('failed_and_initial_invalid_requests_retained') \
            or not results.get('not_a_DEV_or_FINAL_generalization_score') \
            or results.get('unique_fresh_TRAIN_cases')!=16 or results.get('physical_candidate_attempts')!=128:
        raise ValueError('Explicit four-region selection requires a complete frozen TRAIN result')
    source=deepcopy(waypoints['shelves']);regions={}
    for region,name in selections.items():
        choices=[e for e in results['regions'][region] if e['candidate']==name]
        if len(choices)!=1:raise ValueError('Selected candidate must occur exactly once in its region')
        evidence=deepcopy(choices[0]);offset=evidence['offset_xy_yaw']
        original=source['upper' if region.startswith('shelf_3') else 'middle']
        template=dict(source_split='train',measured_success=evidence['success']>0,
            box_size_m=deepcopy(original['box_size_m']),
            base_minus_initial_box_xy_rack_m=[original['base_minus_initial_box_xy_rack_m'][i]+offset[i] for i in range(2)],
            base_yaw_rack_rad=original['base_yaw_rack_rad']+offset[2])
        regions[region]=dict(template=template,TRAIN_evidence=evidence,
            unproven_grasp_candidate=evidence['success']==0)
    contract=dict(name=FORMAT,source_shelf_templates=source,regions=regions,source_results_SHA256=results_SHA256,
        source_unique_TRAIN_cases=16,source_candidate_requests=128,
        source_all178_tensors_and_initial_counters_frozen=True,source_Q_replay_rows_imported=0,
        fresh_matching_Q_replay_required=True,source_flap_draws_contact_history_not_matched=True,
        all_region_generalization_unproven=True,box_base_background_randomization_preserved=True)
    validate_region_workplaces(contract)
    return deepcopy(waypoints)|dict(region_workplaces=contract)


def frozen_anchor_templates(templates):
    """Only actor initialization ignores target changes; Q contracts keep them."""
    if isinstance(templates,dict) and templates.get('name')==FORMAT:
        return validate_region_workplaces(templates)['source_shelf_templates']
    return templates
