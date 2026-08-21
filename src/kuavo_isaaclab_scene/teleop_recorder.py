"""Incremental HDF5 episode writer for Kuavo Quest teleoperation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

import h5py
import numpy as np


class TeleopRecorder(Protocol):
    """Common interface used by the Quest collector."""

    @property
    def recording(self) -> bool: ...

    def start_episode(self, metadata: dict[str, Any] | None = None) -> str: ...

    def append(self, sample: dict[str, Any]) -> None: ...

    def finish_episode(self, *, success: bool, reason: str) -> str | None: ...

    def close(self) -> None: ...


class TeleopRecorderGroup:
    """Fan out a recording session to one or more storage formats."""

    def __init__(self, recorders: dict[str, TeleopRecorder]):
        if not recorders:
            raise ValueError("At least one recorder is required.")
        self._recorders = dict(recorders)

    @property
    def recording(self) -> bool:
        states = {recorder.recording for recorder in self._recorders.values()}
        if len(states) != 1:
            raise RuntimeError("Recorder backends have inconsistent episode states.")
        return states.pop()

    def start_episode(self, metadata: dict[str, Any] | None = None) -> str:
        names = {
            label: recorder.start_episode(metadata)
            for label, recorder in self._recorders.items()
        }
        return ", ".join(f"{label}={name}" for label, name in names.items())

    def append(self, sample: dict[str, Any]) -> None:
        for recorder in self._recorders.values():
            recorder.append(sample)

    def finish_episode(self, *, success: bool, reason: str) -> str | None:
        names = {
            label: recorder.finish_episode(success=success, reason=reason)
            for label, recorder in self._recorders.items()
        }
        saved = [f"{label}={name}" for label, name in names.items() if name is not None]
        return ", ".join(saved) if saved else None

    def close(self) -> None:
        first_error: Exception | None = None
        for recorder in self._recorders.values():
            try:
                recorder.close()
            except Exception as exc:  # Close every backend before propagating.
                first_error = first_error or exc
        if first_error is not None:
            raise first_error


class TeleopHdf5Recorder:
    """Write heterogeneous simulator samples without retaining them in RAM."""

    def __init__(self, path: str | Path, *, flush_every: int = 30):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self.path, "a")
        self._data = self._file.require_group("data")
        self._file.attrs.setdefault("format", "kuavo_quest_teleop_hdf5")
        self._file.attrs.setdefault("format_version", "1.0")
        self._flush_every = max(1, int(flush_every))
        self._episode: h5py.Group | None = None
        self._sample_count = 0

    @property
    def recording(self) -> bool:
        return self._episode is not None

    @property
    def episode_count(self) -> int:
        return len(self._data)

    def start_episode(self, metadata: dict[str, Any] | None = None) -> str:
        if self.recording:
            raise RuntimeError("An episode is already being recorded.")
        index = 0
        while f"demo_{index:05d}" in self._data:
            index += 1
        name = f"demo_{index:05d}"
        self._episode = self._data.create_group(name)
        self._episode.require_group("samples")
        self._sample_count = 0
        for key, value in (metadata or {}).items():
            self._episode.attrs[key] = self._attribute_value(value)
        self._file.flush()
        return name

    @staticmethod
    def _attribute_value(value: Any) -> Any:
        if isinstance(value, (str, bytes, int, float, bool, np.number)):
            return value
        return json.dumps(value, ensure_ascii=False)

    def append(self, sample: dict[str, Any]) -> None:
        if self._episode is None:
            return
        samples = self._episode["samples"]
        for name, value in sample.items():
            array = np.asarray(value)
            if array.dtype.kind in {"U", "O"}:
                raise TypeError(f"Sample {name!r} must be numeric, got dtype {array.dtype}.")
            if name not in samples:
                shape = (0, *array.shape)
                maxshape = (None, *array.shape)
                kwargs: dict[str, Any] = {"shape": shape, "maxshape": maxshape, "dtype": array.dtype, "chunks": True}
                if array.ndim >= 2:
                    kwargs["compression"] = "lzf"
                samples.create_dataset(name, **kwargs)
            dataset = samples[name]
            if dataset.shape[1:] != array.shape:
                raise ValueError(
                    f"Sample {name!r} changed shape from {dataset.shape[1:]} to {array.shape}."
                )
            dataset.resize(self._sample_count + 1, axis=0)
            dataset[self._sample_count] = array
        self._sample_count += 1
        if self._sample_count % self._flush_every == 0:
            self._file.flush()

    def finish_episode(self, *, success: bool, reason: str) -> str | None:
        if self._episode is None:
            return None
        name = self._episode.name.rsplit("/", 1)[-1]
        self._episode.attrs["num_samples"] = self._sample_count
        self._episode.attrs["success"] = bool(success)
        self._episode.attrs["end_reason"] = reason
        if self._sample_count == 0:
            del self._data[name]
            name = None
        self._episode = None
        self._sample_count = 0
        self._file.flush()
        return name

    def close(self) -> None:
        if self._episode is not None:
            self.finish_episode(success=False, reason="process_closed")
        if self._file:
            self._file.close()

    def __enter__(self) -> "TeleopHdf5Recorder":
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
