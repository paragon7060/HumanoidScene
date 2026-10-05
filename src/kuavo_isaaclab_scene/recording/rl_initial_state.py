"""Capture an episode's physical seed without stepping or changing the scene.

This supports physical replay diagnostics. It is not a serialized simulator:
PhysX contact warm starts, perception filter history and reward hold timers are
not captured. Legacy pose-only demonstrations cannot recover missing flap
joint states from their flat observations.
"""

from __future__ import annotations

import numpy as np


def _select(value, index=0, count=1):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    value = np.asarray(value)
    if value.ndim == 0 or value.shape[0] != count:
        raise ValueError("Physical seed array must match the environment count")
    return value[index].copy()


def _scene_arrays(mapping, index=0, count=1):
    return {name: _scene_arrays(value, index, count) if isinstance(value, dict) else _select(value, index, count)
            for name, value in mapping.items()}


def capture_rl_initial_state(env, observations, *, env_index=None) -> dict:
    """Freeze pre-action root/joint velocities, flap joints and pending targets.

    Called only once per collected attempt, not on every training transition.
    Targets and action memory are recorded separately from measured joints;
    substituting measured angles would lose the pending PD command.
    With randomized flap dynamics, verified hinge properties are included.
    """
    if env_index is None and env.num_envs != 1:
        raise ValueError("Physical demo seed requires exactly one environment")
    index = 0 if env_index is None else env_index
    if type(index) is not int or not 0 <= index < env.num_envs:
        raise ValueError("Physical seed needs an explicit valid environment index")
    def select(value):
        return _select(value, index, env.num_envs)
    scene = _scene_arrays(env.scene.get_state(is_relative=False), index, env.num_envs)
    drives = {}
    for name in env.scene.articulations:
        data = env.scene[name].data
        drives[name] = {
            "joint_position": select(data.joint_pos_target),
            "joint_velocity": select(data.joint_vel_target),
            "joint_effort": select(data.joint_effort_target),
        }
    terms = {}
    for name in env.action_manager.active_terms:
        term = env.action_manager.get_term(name)
        state = {"raw": select(term.raw_actions),
                 "processed": select(term.processed_actions)}
        # State needed beyond the public processed command for the project's
        # incremental joint, planar-drive and upright-torso controllers.
        for field in ("_targets", "_joint_targets", "_target_xz", "_origin_xz",
                      "_pitch_reference", "_close_requested"):
            value = getattr(term, field, None)
            if value is not None:
                state[field.removeprefix("_")] = select(value)
        terms[name] = state
    logical = {}
    for field in ("active", "box_type_ids", "region_ids", "pool_ids"):
        value = getattr(env, "_multi_box_" + field, None)
        if value is not None:
            logical[field] = select(value)
    result = {
        "scene": scene,
        "drive_targets": drives,
        "action_terms": terms,
        "action": select(env.action_manager.action),
        "previous_action": select(env.action_manager.prev_action),
        "logical_boxes": logical,
        "observations": {name: select(value) for name, value in observations.items()},
        "control_step": np.int64(env.common_step_counter),
        "episode_step": select(env.episode_length_buf),
    }
    if getattr(getattr(env, "cfg", None), "flap_dynamics", None) is not None:
        from ..rl.multi_box.scene.flap_dynamics import flap_initial_parameters
        result["flap_joint_properties"] = flap_initial_parameters(env, index)
        result["flap_joint_properties_schema_version"] = np.int64(1)
    return result
