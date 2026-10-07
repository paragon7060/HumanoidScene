"""Grasp-only drop detection and strict identity for historical checkpoints."""
from copy import deepcopy
from dataclasses import replace
import math

import torch

DROP_REFERENCE = 'settled_initial_box_height_v1'
DEFAULT_DROP_HEIGHT_M = .10
DROP_KEYS = ('box_drop_reference', 'box_drop_height_m')


def grasp_drop_limit(spec):
    # Carry/place may intentionally lower a held box onto the conveyor.
    return spec.max_box_drop_height if spec.skill == 'grasp' else None


def grasp_drop_terminal_contract(spec):
    limit = grasp_drop_limit(spec)
    return {} if limit is None else dict(box_drop_reference=DROP_REFERENCE,
                                        box_drop_height_m=float(limit))


def grasp_drop_safety_thresholds(spec):
    limit = grasp_drop_limit(spec)
    return {} if limit is None else dict(max_box_drop_height_m=float(limit))


def configured_drop_limit(contract):
    terminal = contract.get('terminal_contract', {})
    thresholds = terminal.get('safety_thresholds', {})
    present = any(k in terminal for k in DROP_KEYS) or 'max_box_drop_height_m' in thresholds
    if not present:
        return None  # Explicit historical contract: world-center z < 0.12 only.
    limit = terminal.get('box_drop_height_m')
    if (terminal.get('box_drop_reference') != DROP_REFERENCE
            or isinstance(limit, bool) or not isinstance(limit, (int, float))
            or not math.isfinite(limit) or not 0 < limit <= .50
            or thresholds.get('max_box_drop_height_m') != limit):
        raise ValueError('Unknown or inconsistent grasp box-drop contract')
    return float(limit)


def configure_grasp_drop(cfg, contract):
    limit = configured_drop_limit(contract)
    if limit is not None and cfg.multi_box.skill != 'grasp':
        raise ValueError('Reset-relative drop guard is restricted to the grasp skill')
    cfg.multi_box = replace(cfg.multi_box, max_box_drop_height=limit)


def frozen_drop_actor_contract(contract):
    """Ignore the reviewed change only for a frozen actor; never Q or replay."""
    limit = configured_drop_limit(contract)
    if limit is None:
        return contract
    if limit != DEFAULT_DROP_HEIGHT_M:
        raise ValueError('Only reviewed 10cm drop guard can restore a historical frozen actor')
    result = deepcopy(contract)
    terminal = result['terminal_contract']
    for key in DROP_KEYS:
        terminal.pop(key)
    terminal['safety_thresholds'].pop('max_box_drop_height_m')
    return result


def with_reset_drop_profile(contract):
    """Prepare a new grasp MDP; callers must start fresh Q and reward banks."""
    if contract.get('skill') != 'grasp' or configured_drop_limit(contract) is not None:
        raise ValueError('New reset-relative drop profile requires a historical grasp contract')
    result = deepcopy(contract)
    terminal = result['terminal_contract']
    terminal.update(box_drop_reference=DROP_REFERENCE,box_drop_height_m=DEFAULT_DROP_HEIGHT_M)
    terminal['safety_thresholds']['max_box_drop_height_m']=DEFAULT_DROP_HEIGHT_M
    assert configured_drop_limit(result)==DEFAULT_DROP_HEIGHT_M
    assert frozen_drop_actor_contract(result)==contract
    return result


def grasp_box_drop(height_above_origin, lift_from_reset, finite, limit):
    """A fallen side-resting box may have its center ABOVE the old 12cm cut."""
    dropped = (height_above_origin < .12) | ~finite
    if limit is not None:
        dropped = dropped | (lift_from_reset < -limit) | ~torch.isfinite(lift_from_reset)
    return dropped
