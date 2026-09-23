"""Stream V2 Quest demonstrations as SAC-compatible, time-aligned transitions."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


TRANSITION_FIELDS = (
    "actor_obs", "critic_obs", "action", "reward", "next_actor_obs",
    "next_critic_obs", "terminated", "truncated", "success", "unsafe",
    "sim_time_s",
)


class RlTransitionRecorder:
    """One exclusive HDF5 file; each operator attempt is a separate episode."""

    def __init__(self, path: str | Path, manifest: dict):
        self.manifest = dict(manifest)
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(self.path, "x")
        self.file.attrs["format"] = "kuavo_v2_grasp_sac_transitions"
        self.file.attrs["format_version"] = 1
        self.file.attrs["manifest_json"] = json.dumps(manifest, sort_keys=True)
        self.episodes = self.file.create_group("episodes")
        self.episode = None
        self.count = 0
        self.finished = 0
        self.file.flush()

    @property
    def recording(self) -> bool:
        return self.episode is not None

    def start_episode(self) -> str:
        if self.recording:
            raise RuntimeError("RL episode is already recording")
        name = f"episode_{self.finished:06d}"
        self.episode = self.episodes.create_group(name)
        self.episode.create_group("transitions")
        self.count = 0
        self.file.flush()
        return name

    def append(self, sample: dict) -> None:
        if self.episode is None:
            raise RuntimeError("Start an RL episode before appending")
        missing = set(TRANSITION_FIELDS) - sample.keys()
        extra = sample.keys() - set(TRANSITION_FIELDS)
        if missing or extra:
            raise ValueError(f"RL transition fields mismatch: missing={missing}, extra={extra}")
        for name, dimension in (
            ("actor_obs", "actor_obs_dim"),
            ("next_actor_obs", "actor_obs_dim"),
            ("critic_obs", "critic_obs_dim"),
            ("next_critic_obs", "critic_obs_dim"),
            ("action", "action_dim"),
        ):
            expected = self.manifest.get(dimension)
            if expected is not None and np.shape(sample[name]) != (expected,):
                raise ValueError(f"RL transition {name} must have shape ({expected},)")
        group = self.episode["transitions"]
        for name in TRANSITION_FIELDS:
            value = np.asarray(sample[name])
            if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
                raise TypeError(f"RL transition {name} must be numeric")
            if name not in group:
                group.create_dataset(
                    name, shape=(0, *value.shape), maxshape=(None, *value.shape),
                    chunks=(1, *value.shape), dtype=value.dtype,
                )
            dataset = group[name]
            if dataset.shape[1:] != value.shape:
                raise ValueError(f"RL transition {name} changed shape")
            dataset.resize(self.count + 1, axis=0)
            dataset[self.count] = value
        self.count += 1
        if self.count % 30 == 0:
            self.file.flush()

    def finish_episode(self, *, success: bool, reason: str) -> str | None:
        if self.episode is None:
            return None
        name = self.episode.name.rsplit("/", 1)[-1]
        self.episode.attrs["num_transitions"] = self.count
        self.episode.attrs["success"] = bool(success)
        self.episode.attrs["end_reason"] = reason
        if not self.count:
            del self.episodes[name]
            name = None
        else:
            self.finished += 1
        self.episode = None
        self.count = 0
        self.file.flush()
        return name

    def close(self) -> None:
        if self.file:
            self.finish_episode(success=False, reason="process_closed")
            self.file.close()
