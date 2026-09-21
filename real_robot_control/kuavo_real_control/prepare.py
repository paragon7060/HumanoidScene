"""Convert recorded demonstrations and RL actions into deployment trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .contract import ARM_JOINT_NAMES, ControlConfig
from .safety import SafetyError, SafetySupervisor
from .trajectory import Trajectory


TELEOP_COMMAND_SOURCES = (
    "auto",
    "self_collision_safe_target",
    "measured_state",
)


def _decode_attribute(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _list_attribute(value, name: str) -> Sequence[str]:
    value = _decode_attribute(value)
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = value.split(",")
    else:
        decoded = list(value)
    result = [str(_decode_attribute(item)) for item in decoded]
    if not result:
        raise ValueError("Missing {} metadata".format(name))
    return result


def _boolean_attribute(value, name: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValueError("Episode has no valid {} flag".format(name))


def teleop_hdf5_to_trajectory(
    path: Path,
    episode: Optional[str] = None,
    *,
    command_source: str = "auto",
    require_success: bool = True,
) -> Trajectory:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("Preparing HDF5 requires h5py on the workstation") from exc

    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(str(path), "r") as source:
        if source.attrs.get("format") != "kuavo_quest_teleop_hdf5":
            raise ValueError("Input is not a Kuavo Quest teleop HDF5 file")
        data = source.get("data")
        if data is None or not len(data):
            raise ValueError("HDF5 contains no episodes")
        episode_name = episode or sorted(data.keys())[0]
        if episode_name not in data:
            raise ValueError("Episode does not exist: {}".format(episode_name))
        group = data[episode_name]
        samples = group.get("samples")
        if samples is None:
            raise ValueError("Episode has no samples group")
        if "sim_time_s" not in samples:
            raise ValueError("Episode is missing sim_time_s")

        success = _boolean_attribute(group.attrs.get("success"), "success")
        end_reason = str(_decode_attribute(group.attrs.get("end_reason", "")))
        if require_success and not success:
            raise ValueError(
                "Episode {} is unsuccessful (end_reason={!r}); pass --allow-unsuccessful only for offline review".format(
                    episode_name, end_reason
                )
            )
        if command_source not in TELEOP_COMMAND_SOURCES:
            raise ValueError("Unsupported teleop command source: {}".format(command_source))

        safe_target_available = (
            "self_collision_safe_joint_target" in samples
            and bool(group.attrs.get("self_collision", False))
        )
        selected_source = command_source
        if selected_source == "auto":
            selected_source = "self_collision_safe_target" if safe_target_available else "measured_state"

        collision_modified_count = 0
        minimum_clearance_m = None
        configured_clearance_m = None
        if selected_source == "self_collision_safe_target":
            if not safe_target_available:
                raise ValueError("Episode has no self-collision-safe joint target")
            target_names = _list_attribute(
                group.attrs.get("self_collision_joint_names", ""), "self_collision_joint_names"
            )
            missing = [name for name in ARM_JOINT_NAMES if name not in target_names]
            if missing:
                raise ValueError("Safe target is missing arm joints: {}".format(missing))
            targets = np.asarray(samples["self_collision_safe_joint_target"], dtype=np.float64)
            if targets.ndim != 2 or targets.shape[1] != len(target_names):
                raise ValueError(
                    "self_collision_safe_joint_target shape disagrees with self_collision_joint_names"
                )
            arms = targets[:, [target_names.index(name) for name in ARM_JOINT_NAMES]]
            if "self_collision_modified" in samples:
                modified = np.asarray(samples["self_collision_modified"], dtype=np.uint8).reshape(-1)
                if len(modified) != len(arms):
                    raise ValueError("self_collision_modified length differs from safe target")
                collision_modified_count = int(np.count_nonzero(modified))
            if "self_collision_minimum_distance_m" in samples:
                distances = np.asarray(
                    samples["self_collision_minimum_distance_m"], dtype=np.float64
                ).reshape(-1)
                if len(distances) != len(arms) or not np.isfinite(distances).all():
                    raise ValueError("self-collision distance must be finite and match safe target length")
                minimum_clearance_m = float(np.min(distances))
                configured_clearance_m = float(group.attrs.get("self_collision_clearance_m", float("nan")))
                if not np.isfinite(configured_clearance_m) or configured_clearance_m <= 0.0:
                    raise ValueError("Episode has no valid self-collision clearance")
                if minimum_clearance_m + 1e-7 < configured_clearance_m:
                    raise ValueError(
                        "Safe target clearance {:.6f} m is below configured {:.6f} m".format(
                            minimum_clearance_m, configured_clearance_m
                        )
                    )
            sample_semantics = "post_ik_self_collision_safe_joint_target"
        else:
            if "robot_joint_position" not in samples:
                raise ValueError("Episode is missing robot_joint_position")
            joint_names = _list_attribute(group.attrs.get("joint_names", ""), "joint_names")
            missing = [name for name in ARM_JOINT_NAMES if name not in joint_names]
            if missing:
                raise ValueError("Recorded state is missing arm joints: {}".format(missing))
            states = np.asarray(samples["robot_joint_position"], dtype=np.float64)
            if states.ndim != 2 or states.shape[1] != len(joint_names):
                raise ValueError("robot_joint_position shape disagrees with joint_names")
            arms = states[:, [joint_names.index(name) for name in ARM_JOINT_NAMES]]
            sample_semantics = "post_step_measured_simulated_arm_state"

        timestamps = np.asarray(samples["sim_time_s"], dtype=np.float64).reshape(-1)
        if len(timestamps) != len(arms):
            raise ValueError("sim_time_s and selected arm source lengths differ")
        if not len(timestamps):
            raise ValueError("Episode contains no samples")
        timestamps = timestamps - timestamps[0]

        gripper = None
        action_layout = _list_attribute(group.attrs.get("action_layout", ""), "action_layout")
        if "action" in samples and all(name in action_layout for name in ("left_gripper_close", "right_gripper_close")):
            actions = np.asarray(samples["action"], dtype=np.float64)
            gripper = actions[:, [action_layout.index("left_gripper_close"), action_layout.index("right_gripper_close")]]

        metadata = {
            "schema_version": 1,
            "kind": "joint_position_rad",
            "robot_model": "s63",
            "gripper": "leju-twofinger",
            "arm_joint_names": list(ARM_JOINT_NAMES),
            "source_type": "kuavo_quest_teleop_hdf5",
            "source_path": str(path),
            "source_sha256": digest,
            "source_episode": episode_name,
            "source_episode_success": success,
            "source_episode_end_reason": end_reason,
            "source_command_source": selected_source,
            "sample_semantics": sample_semantics,
            "source_control_dt_s": float(group.attrs.get("control_dt", 0.0)),
            "source_initial_state": str(
                _decode_attribute(group.attrs.get("initial_state_name", ""))
            ),
            "self_collision_enabled": bool(group.attrs.get("self_collision", False)),
            "self_collision_modified_samples": collision_modified_count,
            "self_collision_minimum_distance_m": minimum_clearance_m,
            "self_collision_configured_clearance_m": configured_clearance_m,
            "gripper_encoding": "binary_close: 0=open, 1=close",
            "deployment_ready": False,
        }
    return Trajectory("joint_position_rad", timestamps, arms, gripper, metadata).validate()


def retime_joint_position_trajectory(
    trajectory: Trajectory,
    config: ControlConfig,
    *,
    velocity_fraction: float = 0.9,
) -> Trajectory:
    """Slow, resample and safety-filter a joint path for the real 30 Hz runner.

    The source path is first globally slowed enough to leave velocity headroom.
    It is then linearly sampled at the real publisher rate and passed through the
    same acceleration-limited supervisor used online.  The generated artifact
    therefore contains the actual logical command path instead of relying on an
    online limiter to silently fall behind a faster demonstration.
    """
    trajectory.validate()
    if trajectory.kind != "joint_position_rad":
        raise ValueError("Only absolute joint-position trajectories can be retimed")
    if not np.isfinite(velocity_fraction) or not 0.1 <= velocity_fraction <= 1.0:
        raise ValueError("velocity_fraction must be finite and in [0.1, 1.0]")

    source_t = np.asarray(trajectory.timestamps_s, dtype=np.float64)
    source_q = np.asarray(trajectory.arm_values, dtype=np.float64)
    if len(source_t) == 1:
        metadata = dict(trajectory.metadata)
        metadata["retiming"] = {
            "algorithm": "s63_supervisor_v1",
            "operational_limits_provenance": config.operational_limits_provenance,
            "max_command_velocity_rad_s": config.max_command_velocity_rad_s,
            "max_command_acceleration_rad_s2": config.max_command_acceleration_rad_s2,
            "time_scale": 1.0,
            "source_duration_s": 0.0,
            "source_samples": 1,
            "output_hz": config.publish_hz,
            "output_samples": 1,
            "source_max_velocity_rad_s": 0.0,
            "output_max_velocity_rad_s": 0.0,
            "output_max_acceleration_rad_s2": 0.0,
            "max_path_filter_error_rad": 0.0,
        }
        metadata["deployment_ready"] = bool(metadata.get("source_episode_success", True))
        return Trajectory(
            trajectory.kind, source_t.copy(), source_q.copy(),
            None if trajectory.gripper_close is None else trajectory.gripper_close.copy(), metadata
        ).validate()

    source_dt = np.diff(source_t)
    source_velocity = np.diff(source_q, axis=0) / source_dt[:, None]
    source_max_velocity = float(np.max(np.abs(source_velocity)))
    allowed_velocity = config.max_command_velocity_rad_s * velocity_fraction
    time_scale = max(1.0, source_max_velocity / allowed_velocity)
    period = 1.0 / config.publish_hz
    for retiming_attempt in range(9):
        slowed_duration = trajectory.duration_s * time_scale
        interval_count = max(1, int(np.ceil(slowed_duration * config.publish_hz)))
        raw_t = np.linspace(0.0, slowed_duration, interval_count + 1, dtype=np.float64)
        source_query_t = raw_t / time_scale
        raw_q = np.column_stack([
            np.interp(source_query_t, source_t, source_q[:, joint])
            for joint in range(source_q.shape[1])
        ])

        supervisor = SafetySupervisor(config)
        try:
            supervisor.synchronize(raw_q[0])
            filtered = [raw_q[0].copy()]
            for target in raw_q[1:]:
                command, _ = supervisor.project(target, supervisor.command, period)
                filtered.append(command)

            # Do not truncate the final pose just because the source clock ended
            # while the acceleration limiter was still settling.
            final_target = raw_q[-1]
            max_tail_steps = int(np.ceil(10.0 * config.publish_hz))
            for _ in range(max_tail_steps):
                command, velocity = supervisor.project(final_target, supervisor.command, period)
                filtered.append(command)
                if (
                    np.max(np.abs(command - final_target)) <= 1e-10
                    and np.max(np.abs(velocity)) <= 1e-10
                ):
                    break
            else:
                raise ValueError("Safety retiming could not settle on the final target within 10 seconds")
            break
        except SafetyError as exc:
            if "without violating acceleration limit" not in str(exc) or retiming_attempt == 8:
                raise
            # A sharp source reversal can be infeasible on the first time
            # scale even though its pointwise velocity is below the limit.
            # Slow the complete path and retry; never relax the acceleration.
            time_scale *= 1.25
    else:  # pragma: no cover - the bounded loop either breaks or raises above.
        raise AssertionError("unreachable retiming loop")

    output_q = np.asarray(filtered, dtype=np.float64)
    output_t = np.arange(len(output_q), dtype=np.float64) * period
    raw_for_error = raw_q[: min(len(raw_q), len(output_q))]
    max_filter_error = float(
        np.max(np.abs(output_q[: len(raw_for_error)] - raw_for_error))
    )
    output_velocity = np.diff(output_q, axis=0) / period
    if len(output_velocity) > 1:
        output_acceleration = np.diff(output_velocity, axis=0) / period
        output_max_acceleration = float(np.max(np.abs(output_acceleration)))
    else:
        output_max_acceleration = 0.0
    output_max_velocity = float(np.max(np.abs(output_velocity))) if len(output_velocity) else 0.0

    output_gripper = None
    if trajectory.gripper_close is not None:
        raw_indices = np.searchsorted(source_t, source_query_t, side="right") - 1
        raw_indices = np.clip(raw_indices, 0, len(source_t) - 1)
        raw_gripper = np.asarray(trajectory.gripper_close)[raw_indices]
        if len(output_q) > len(raw_gripper):
            tail = np.repeat(raw_gripper[-1][None, :], len(output_q) - len(raw_gripper), axis=0)
            output_gripper = np.concatenate((raw_gripper, tail), axis=0)
        else:
            output_gripper = raw_gripper[: len(output_q)]

    metadata = dict(trajectory.metadata)
    metadata["retiming"] = {
        "algorithm": "s63_supervisor_v1",
        "operational_limits_provenance": config.operational_limits_provenance,
        "max_command_velocity_rad_s": config.max_command_velocity_rad_s,
        "max_command_acceleration_rad_s2": config.max_command_acceleration_rad_s2,
        "velocity_fraction": float(velocity_fraction),
        "time_scale": float(time_scale),
        "retiming_attempts": retiming_attempt + 1,
        "source_duration_s": trajectory.duration_s,
        "source_samples": len(source_t),
        "output_hz": config.publish_hz,
        "output_samples": len(output_t),
        "source_max_velocity_rad_s": source_max_velocity,
        "output_max_velocity_rad_s": output_max_velocity,
        "output_max_acceleration_rad_s2": output_max_acceleration,
        "max_path_filter_error_rad": max_filter_error,
    }
    metadata["deployment_ready"] = bool(metadata.get("source_episode_success", True))
    return Trajectory(
        trajectory.kind, output_t, output_q, output_gripper, metadata
    ).validate()


def rl_actions_to_trajectory(
    actions: np.ndarray,
    timestamps_s: np.ndarray,
    *,
    source: str,
    delta_scale_rad: float,
) -> Trajectory:
    actions = np.asarray(actions, dtype=np.float64)
    timestamps = np.asarray(timestamps_s, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] not in (14, 16):
        raise ValueError("RL actions must have 14 arm columns or 14 arm + 2 gripper columns")
    gripper = None
    if actions.shape[1] == 16:
        gripper = actions[:, 14:16]
    metadata = {
        "schema_version": 1,
        "kind": "normalized_joint_delta",
        "robot_model": "s63",
        "gripper": "leju-twofinger",
        "arm_joint_names": list(ARM_JOINT_NAMES),
        "source_type": "rl_normalized_actions",
        "source_path": source,
        "delta_scale_rad": float(delta_scale_rad),
        "action_semantics": "integrate action * delta_scale_rad once per timestamp",
        "gripper_encoding": "binary_close: 0=open, 1=close",
    }
    return Trajectory("normalized_joint_delta", timestamps, actions[:, :14], gripper, metadata).validate()


def load_rl_action_rows(path: Path, action_key: str = "action") -> np.ndarray:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(str(path), allow_pickle=False), dtype=np.float64)
    if suffix == ".npz":
        with np.load(str(path), allow_pickle=False) as payload:
            if action_key not in payload:
                raise ValueError("NPZ has no {!r} array".format(action_key))
            return np.asarray(payload[action_key], dtype=np.float64)
    if suffix == ".jsonl":
        rows = []
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if action_key not in row:
                    raise ValueError("JSONL line {} has no {!r}".format(line_number, action_key))
                rows.append(row[action_key])
        return np.asarray(rows, dtype=np.float64)
    raise ValueError("RL action input must be .npy, .npz, or .jsonl")
