"""Capture an episode's physical seed without stepping or changing the scene.

This supports physical replay diagnostics. It is not a serialized simulator:
PhysX contact warm starts, perception filter history and reward hold timers are
not captured. Legacy pose-only demonstrations cannot recover missing flap
joint states from their flat observations.
"""

from __future__ import annotations

import numpy as np


def _first(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 0 or value.shape[0] != 1:
        raise ValueError("Physical demo seed requires exactly one environment")
    return value[0].copy()


def _scene_arrays(mapping):
    return {name: _scene_arrays(value) if isinstance(value, dict) else _first(value)
            for name, value in mapping.items()}


def capture_rl_initial_state(env, observations) -> dict:
    """Freeze pre-action root/joint velocities, flap joints and pending targets.

    Called only once per collected attempt, not on every training transition.
    Targets and action memory are recorded separately from measured joints;
    substituting measured angles would lose the pending PD command.
    """
    if env.num_envs != 1:
        raise ValueError("Physical demo seed requires exactly one environment")
    scene = _scene_arrays(env.scene.get_state(is_relative=False))
    drives = {}
    for name in env.scene.articulations:
        data = env.scene[name].data
        drives[name] = {
            "joint_position": _first(data.joint_pos_target),
            "joint_velocity": _first(data.joint_vel_target),
            "joint_effort": _first(data.joint_effort_target),
        }
    terms = {}
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        state = {"raw": _first(term.raw_actions),
                 "processed": _first(term.processed_actions)}
        # State needed beyond the public processed command for the project's
        # incremental joint, planar-drive and upright-torso controllers.
        for field in ("_targets", "_joint_targets", "_target_xz", "_origin_xz",
                      "_pitch_reference", "_close_requested"):
            value = getattr(term, field, None)
            if value is not None:
                state[field.removeprefix("_")] = _first(value)
        terms[name] = state
    logical = {}
    for field in ("active", "box_type_ids", "region_ids", "pool_ids"):
        value = getattr(env, "_multi_box_" + field, None)
        if value is not None:
            logical[field] = _first(value)
    return {
        "scene": scene,
        "drive_targets": drives,
        "action_terms": terms,
        "action": _first(env.action_manager.action),
        "previous_action": _first(env.action_manager.prev_action),
        "logical_boxes": logical,
        "observations": {name: _first(value) for name, value in observations.items()},
        "control_step": np.int64(env.common_step_counter),
        "episode_step": _first(env.episode_length_buf),
    }
