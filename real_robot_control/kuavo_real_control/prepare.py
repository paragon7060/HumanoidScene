"""Convert recorded demonstrations and RL actions into deployment trajectories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .contract import ARM_JOINT_NAMES
from .trajectory import Trajectory


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


def teleop_hdf5_to_trajectory(path: Path, episode: Optional[str] = None) -> Trajectory:
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
        required = {"sim_time_s", "robot_joint_position"}
        if not required.issubset(samples.keys()):
            raise ValueError("Episode is missing {}".format(sorted(required - set(samples.keys()))))

        joint_names = _list_attribute(group.attrs.get("joint_names", ""), "joint_names")
        missing = [name for name in ARM_JOINT_NAMES if name not in joint_names]
        if missing:
            raise ValueError("Recorded state is missing arm joints: {}".format(missing))
        indices = [joint_names.index(name) for name in ARM_JOINT_NAMES]
        states = np.asarray(samples["robot_joint_position"], dtype=np.float64)
        if states.ndim != 2 or states.shape[1] != len(joint_names):
            raise ValueError("robot_joint_position shape disagrees with joint_names")
        arms = states[:, indices]
        timestamps = np.asarray(samples["sim_time_s"], dtype=np.float64).reshape(-1)
        if len(timestamps) != len(arms):
            raise ValueError("sim_time_s and robot_joint_position lengths differ")
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
            "sample_semantics": "post-step measured simulated arm state replay",
            "source_control_dt_s": float(group.attrs.get("control_dt", 0.0)),
            "gripper_encoding": "binary_close: 0=open, 1=close",
        }
    return Trajectory("joint_position_rad", timestamps, arms, gripper, metadata).validate()


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
