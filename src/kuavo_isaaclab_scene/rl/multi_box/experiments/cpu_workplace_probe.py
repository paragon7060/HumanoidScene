"""Frozen workplace search on fresh TRAIN layouts, separate from SAC replay."""
from collections import Counter, defaultdict

from .physics_backend_eval import REGIONS, INCOMPATIBLE_FLAGS as BACKEND_FLAGS
from .staged_physics import CPU_PHYSICS_BACKEND, staged_solver_contract
from .waypoint_probe import validate_waypoint_probe

SOURCE = 'CPU_PhysX_frozen_TRAIN_workplace_probe_NOT_matching_Q_replay'
INCOMPATIBLE_FLAGS = (BACKEND_FLAGS - {'--base-waypoint-probe', '--cpu-workplace-probe'}) | frozenset({
    '--frozen-physics-backend-eval', '--cpu-physics-training', '--reset-world-frame-probe',
    '--reset-failure-diagnostics', '--reset-contact-pair-diagnostics',
    '--reset-flap-contact-pair-diagnostics', '--reset-flap-contact-physical-pools'})


def validate_cpu_workplace_probe(waves, contract, *, enabled, device, training, steps,
                                  waypoint_enabled, explicit_frozen, other_probe=False):
    if not enabled:
        return None
    if device != 'cpu' or training or not explicit_frozen or steps != 900 \
            or not waypoint_enabled or other_probe:
        raise ValueError('CPU workplace search requires explicit frozen CPU/900steps and only named waypoints')
    if contract.get('physics_dynamics') != staged_solver_contract('PGS', physics_backend=CPU_PHYSICS_BACKEND):
        raise ValueError('CPU workplace search requires the explicit CPU PhysX checkpoint contract')
    if not isinstance(waves, list) or len(waves) != 1 or not isinstance(waves[0], dict):
        raise ValueError('CPU workplace search requires one fresh TRAIN candidate wave')
    wave = waves[0]
    rows = wave.get('layouts')
    if wave.get('split') != 'train' or wave.get('background_placement', 'original') != 'original' \
            or not isinstance(rows, list) or len(rows) != 128:
        raise ValueError('CPU workplace search requires128 TRAIN candidate requests, no DEV/FINAL or changed background')
    validate_waypoint_probe(waves, enabled=True, training=False)
    groups = defaultdict(list)
    for row in rows:
        layout = row.get('layout', {})
        if layout.get('split') != 'train' or type(layout.get('seed')) is not int \
                or layout.get('target_region') not in REGIONS:
            raise ValueError('Every workplace candidate must retain a fresh TRAIN seed and region')
        groups[layout['seed']].append(row)
    if len(groups) != 16 or Counter(g[0]['layout']['target_region'] for g in groups.values()) \
            != Counter(dict.fromkeys(REGIONS, 4)):
        raise ValueError('Workplace search requires16 distinct TRAIN cases, four per region')
    reference = None
    for group in groups.values():
        if len(group) != 8 or any(row['layout'] != group[0]['layout']
                or row.get('episode_index') != group[0].get('episode_index') for row in group):
            raise ValueError('Each TRAIN case must keep the same requested initial state for all eight candidates')
        candidates = {row['waypoint_probe']['name']: row['waypoint_probe']['offset_xy_yaw'] for row in group}
        if len(candidates) != 8 or [0., 0., 0.] not in candidates.values():
            raise ValueError('Eight distinct candidates must include the unchanged workplace')
        if reference is not None and candidates != reference:
            raise ValueError('All TRAIN cases must compare the same workplace candidates')
        reference = candidates
    return dict(name='CPU_PhysX_frozen_TRAIN_workplace_search_v1', source=SOURCE,
        physics_device='cpu', training=False, Q_import_eligible=False,
        unique_TRAIN_layouts=16, unique_TRAIN_layouts_per_region=4,
        candidates_per_layout=8, original_candidate_requests=128,
        requested_initial_box_base_background_states_preserved=True,
        dynamic_flaps_and_success_safety_preserved=True,
        flap_randomization_draws_and_contact_history_not_matched_between_candidates=True,
        diagnostic_not_full_DEV_generalization_score=True, independent_FINAL_used=False)
