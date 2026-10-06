"""A separate CPU-physics MDP with measured TRAIN and an isolated GPU learner."""
from collections import Counter
from types import SimpleNamespace

from .physics_backend_eval import REGIONS
from .staged_physics import CPU_PHYSICS_BACKEND, staged_solver_contract


SOURCE = 'CPU_PhysX_actual_held_TRAIN_v1'
INCOMPATIBLE_FLAGS = frozenset('--' + name for name in (
    'frozen-physics-backend-eval', 'cpu-workplace-probe', 'reset-failure-diagnostics', 'reset-world-frame-probe',
    'contact-stability-probe', 'tgs-zero-velocity-probe', 'contact-last-probe', 'pgs-probe',
    'gripper-drive-probe', 'centered-world-probe', 'packed-background-probe', 'base-waypoint-probe',
    'reset-solver-probe', 'passive-bearing-probe-layer', 'reset-independent-scene-probe',
    'zero-passive-roller-velocities-probe', 'rear5-support-gap-probe-m', 'grasp-observation-audit',
    'full-distribution-grasp-observation-audit', 'reset-contact-pair-diagnostics',
    'reset-flap-contact-pair-diagnostics', 'reset-flap-contact-physical-pools',
))


def validate_cpu_physics_training(waves, contract, *, enabled, physics_device,
                                  learner_device, training, steps, other_probe=False):
    if not enabled:return None
    if physics_device != 'cpu' or learner_device != 'cuda:0' or not training or steps != 900 or other_probe:
        raise ValueError('CPU physics learning requires explicit CPU simulation, cuda:0 learner, TRAIN/900steps and no probes')
    dynamics = contract.get('physics_dynamics')
    if dynamics != staged_solver_contract('PGS', physics_backend=CPU_PHYSICS_BACKEND):
        raise ValueError('CPU physics learning requires its own PGS checkpoint/replay dynamics identity')
    if not isinstance(waves, list) or not waves or not any(w.get('split') == 'train' for w in waves):
        raise ValueError('CPU physics learning requires fresh TRAIN waves')
    train_seeds, dev_seeds, reference_dev = set(), set(), None
    for wave in waves:
        split = wave.get('split')
        rows = wave.get('layouts')
        if split not in ('train', 'validation') or wave.get('background_variant', 'original') != 'original' \
                or not isinstance(rows, list) or len(rows) != 128:
            raise ValueError('CPU learning preserves original128/32 per region; FINAL and changed backgrounds excluded')
        layouts = [r.get('layout', {}) if isinstance(r, dict) else {} for r in rows]
        if any(not isinstance(r, dict) for r in layouts):raise ValueError('Explicit CPU layouts required')
        seeds = [r.get('seed') for r in layouts]
        if any(type(seed) is not int for seed in seeds) or len(set(seeds)) != 128 \
                or Counter(r.get('target_region') for r in layouts) != Counter(dict.fromkeys(REGIONS, 32)) \
                or any(r.get('split') != ('train' if split == 'train' else 'holdout') for r in layouts):
            raise ValueError('CPU learning requires distinct correctly split seeds and32 requests per region')
        if split == 'train':
            if train_seeds.intersection(seeds):raise ValueError('CPU TRAIN waves must use fresh disjoint seeds')
            train_seeds.update(seeds)
        else:
            if reference_dev is not None and rows != reference_dev:
                raise ValueError('CPU development waves must preserve the same original requests')
            reference_dev = rows
            dev_seeds.update(seeds)
    if train_seeds & dev_seeds:raise ValueError('CPU TRAIN/DEV seeds overlap')
    return dict(name=SOURCE, physics_device='cpu', learner_device='cuda:0', training=True,
        physics_backend=CPU_PHYSICS_BACKEND, source_GPU_Q_replay_import_eligible=False,
        same_CPU_contract_continuation_eligible=True, original_requested_per_wave=128,
        requested_per_region=32, fresh_TRAIN_requested=len(train_seeds), independent_FINAL_used=False,
        original_invalid_requests_retained=True, box_base_background_flap_randomization_preserved=True)


def set_measured_learner_context(pilot, stages, ids):
    """Copy measured held coordinates without changing environment identities."""
    context = stages.held_context(ids)
    pilot.stage = SimpleNamespace(**vars(context))
    pilot.stage.target_xy = context.target_xy.to(pilot.device)
    pilot.stage.target_yaw = (context.target_yaw.to(pilot.device)
        if hasattr(context.target_yaw, 'to') else context.target_yaw)
    pilot.anchor = stages.anchors[ids].to(pilot.device).clone()


def act_measured_held_rows(pilot, stages, ids, observation, clocks, *, supplemental_group=None):
    """Transfer NN inputs only; physical commands return to the simulation device."""
    import torch
    set_measured_learner_context(pilot, stages, ids)
    options = {}
    if supplemental_group is not None:
        options['supplemental'] = observation[supplemental_group][ids].to(pilot.device)
    command, previous = pilot.act(observation['policy'][ids].to(pilot.device),
        torch.cat((observation['policy'], observation['critic']), -1)[ids].to(pilot.device),
        clocks.to(pilot.device), exploration_ids=ids.to(pilot.device), **options)
    return command.to(observation['policy'].device), previous
