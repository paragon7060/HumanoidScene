"""Measure supported new box sizes without relabeling a measured waypoint."""
from collections import Counter
from copy import deepcopy

import torch

from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from ..spec import DEFAULT_RACK_REGIONS
from .layout_generalization import GraspLayout

FORMAT = 'frozen_unmeasured_supported_size_workplace_v1'


def validate_size_workplace_request(waves):
    """The caller must first validate the frozen TRAIN16 x candidates8 matrix."""
    cases = {}
    for wave in waves:
        if wave['split'] != 'train':
            raise ValueError('Unmeasured size workplaces are TRAIN diagnostics only')
        for row in wave['layouts']:
            try:
                layout = GraspLayout(**row['layout']).validate()
            except (TypeError,ValueError) as error:
                raise ValueError('Size workplace reset must use the complete supported layout schema') from error
            if layout.split != 'train' or layout.target_box_type is None:
                raise ValueError('Size workplace candidates require explicit supported sizes and TRAIN seeds')
            if row.get('episode_index') != int(layout.target_region.startswith('shelf_3')):
                raise ValueError('Size workplace resets must retain their measured shelf reference')
            cases[layout.seed] = (layout.target_region, layout.target_box_type)
    counts = Counter(cases.values())
    if not all(counts[region, 'medium'] > 0 for region in ('shelf_2_left', 'shelf_2_right')):
        raise ValueError('New size measurement must include medium targets on both middle sides')
    return dict(name=FORMAT, unmeasured_targets_not_claimed_successful=True,
        measured_source_template_and_actor_contract_preserved=True,
        unique_TRAIN_cases_by_region_and_type={region.name: {
            kind: counts[region.name, kind] for kind in region.allowed_box_types
            if counts[region.name, kind]} for region in DEFAULT_RACK_REGIONS},
        frozen_only=True, Q_import_eligible=False, evaluation_data_used=False,
        size_specific_fresh_TRAIN_confirmation_required_before_matching_SAC=True)


def unmeasured_size_candidate(template, token):
    """Keep the original measured source and mark new-size geometry unproven."""
    region_id = int(token[8:12].argmax())
    region = DEFAULT_RACK_REGIONS[region_id]
    kind = ('small', 'medium')[int(token[3:5].argmax())]
    expected_type = token.new_zeros(2)
    expected_type[('small','medium').index(kind)] = 1
    expected_region = token.new_zeros(4)
    expected_region[region_id] = 1
    if (not torch.equal(token[3:5], expected_type) or
        not torch.equal(token[8:12], expected_region) or
        kind not in region.allowed_box_types or
        not torch.allclose(token[5:8], token.new_tensor(BOX_DIMENSIONS_M[kind]), atol=1e-5, rtol=0)):
        raise ValueError('Size workplace candidate needs a real supported perceived asset')
    source = deepcopy(template)
    if not any(torch.allclose(token.new_tensor(source['box_size_m']),
               token.new_tensor(BOX_DIMENSIONS_M[k]),atol=1e-5,rtol=0)
               for k in region.allowed_box_types):
        raise ValueError('Source waypoint dimensions must describe a known supported asset')
    candidate = deepcopy(template)
    candidate.update(box_size_m=list(BOX_DIMENSIONS_M[kind]), measured_success=False)
    audit = dict(name=FORMAT, requested_box_type=kind, requested_box_size_m=list(BOX_DIMENSIONS_M[kind]),
        measured_source_template=source, size_specific_waypoint_measured=False,
        frozen_only=True, Q_import_eligible=False)
    return candidate, audit
