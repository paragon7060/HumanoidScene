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
        supplemental=self.manifest.get('supplemental_actor_obs_dim',0)
        if type(supplemental)!=int or supplemental<0:
            raise ValueError('Supplemental actor observation width must be a nonnegative integer')
        self.transition_fields=TRANSITION_FIELDS+(
            ('actor_supplemental','next_actor_supplemental') if supplemental else ())
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

    def start_episode(self, *, initial_state: dict | None = None) -> str:
        if self.recording:
            raise RuntimeError("RL episode is already recording")
        # Validate before creating the episode: a rejected seed must not leave
        # an empty/corrupt attempt in an otherwise usable demonstration file.
        snapshot = None if initial_state is None else _numeric_snapshot(initial_state)
        name = f"episode_{self.finished:06d}"
        self.episode = self.episodes.create_group(name)
        self.episode.create_group("transitions")
        if snapshot is not None:
            group = self.episode.create_group("initial_state")
            group.attrs["schema"] = "v2_physical_seed_v1"
            group.attrs["capture_timing"] = "before_first_recorded_action"
            group.attrs["includes_physx_internal_state"] = False
            _write_snapshot(group, snapshot)
        self.count = 0
        self.file.flush()
        return name

    def _validated_values(self, sample: dict) -> dict:
        if self.episode is None:
            raise RuntimeError("Start an RL episode before appending")
        missing = set(self.transition_fields) - sample.keys()
        extra = sample.keys() - set(self.transition_fields)
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
        if self.manifest.get('supplemental_actor_obs_dim'):
            for name in ('actor_supplemental','next_actor_supplemental'):
                if np.shape(sample[name])!=(self.manifest['supplemental_actor_obs_dim'],) \
                        or not np.isfinite(sample[name]).all():
                    raise ValueError('Supplemental perception needs aligned finite declared-width rows')
        result = {}
        for name in self.transition_fields:
            value = np.asarray(sample[name])
            if not np.issubdtype(value.dtype, np.number) and value.dtype != np.bool_:
                raise TypeError(f"RL transition {name} must be numeric")
            result[name] = value
        return result

    def append(self, sample: dict) -> None:
        values = self._validated_values(sample)
        group = self.episode["transitions"]
        for name, value in values.items():
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

    def append_many(self, samples) -> None:
        """Write an already collected episode without per-row HDF5 resizing.

        Batched simulation already owns completed rows in CPU memory. This
        preserves their exact values/alignment and flushes the complete batch;
        realtime Quest callers keep append's existing streaming behavior.
        """
        values = [self._validated_values(sample) for sample in samples]
        if not values:
            return
        group = self.episode["transitions"]
        arrays = {}
        for name in self.transition_fields:
            shape = values[0][name].shape
            if any(value[name].shape != shape for value in values):
                raise ValueError(f"RL transition {name} changed shape")
            if name in group and group[name].shape[1:] != shape:
                raise ValueError(f"RL transition {name} changed shape")
            # Single-row append uses the first row's dtype for this dataset.
            dtype = group[name].dtype if name in group else values[0][name].dtype
            arrays[name] = np.stack([value[name] for value in values]).astype(dtype, copy=False)
        end = self.count + len(values)
        for name, array in arrays.items():
            if name not in group:
                group.create_dataset(name, shape=(0, *array.shape[1:]),
                    maxshape=(None, *array.shape[1:]), chunks=(min(64, len(values)), *array.shape[1:]),
                    dtype=array.dtype, compression="lzf")
            dataset = group[name]
            dataset.resize(end, axis=0)
            dataset[self.count:end] = array
        self.count = end
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


def _numeric_snapshot(mapping: dict) -> dict:
    """Freeze finite numeric arrays, retaining the scene's nested topology."""
    if not isinstance(mapping, dict):
        raise TypeError("Initial state must be a nested dictionary")
    result = {}
    for name, value in mapping.items():
        if not isinstance(name, str) or not name or "/" in name or name in (".", ".."):
            raise ValueError("Invalid initial-state field name")
        if isinstance(value, dict):
            result[name] = _numeric_snapshot(value)
        else:
            array = np.asarray(value)
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError(f"Initial state {name} must be finite numeric data")
            result[name] = array.copy()
    return result


def _write_snapshot(group, mapping: dict) -> None:
    for name, value in mapping.items():
        if isinstance(value, dict):
            _write_snapshot(group.create_group(name), value)
        else:
            group.create_dataset(name, data=value)
