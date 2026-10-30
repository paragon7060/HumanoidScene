"""Bounded, episode-aware state/action data; no image or simulator dependency."""

from bisect import bisect_right
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


ACTION_ENCODING = "manager_normalized_incremental_v1"


class EpisodeWriter:
    def __init__(self, path, manifest):
        self.file = h5py.File(path, "x")
        self.file.attrs["format_version"] = 1
        self.file.attrs["action_encoding"] = ACTION_ENCODING
        self.file.attrs["manifest"] = json.dumps(manifest)
        self.count = 0

    def add(self, observations, actions, success):
        observations, actions = np.asarray(observations, dtype=np.float32), np.asarray(actions, dtype=np.float32)
        if observations.ndim != 2 or actions.ndim != 2 or len(observations) != len(actions) or not len(actions):
            raise ValueError("Expected nonempty, aligned [T, O] observations and [T, A] actions")
        if not np.isfinite(observations).all() or not np.isfinite(actions).all() or np.abs(actions).max() > 1.000001:
            raise ValueError("Demonstrations must be finite with normalized actions in [-1, 1]")
        group = self.file.create_group(f"episode_{self.count:06d}")
        group.create_dataset("obs", data=observations, compression="lzf")
        group.create_dataset("action", data=actions, compression="lzf")
        group.attrs["success"] = bool(success)
        group.attrs["complete"] = True
        self.file.flush()
        self.count += 1

    def close(self):
        self.file.close()


class EpisodeDataset(Dataset):
    """Only full action chunks; no padding and no crossing episode boundaries."""

    def __init__(self, path, horizon, successes_only=True):
        if horizon < 1:
            raise ValueError("Action horizon must be positive")
        self.path, self.horizon = str(path), horizon
        self.file = h5py.File(path, "r")
        try:
            if self.file.attrs.get("format_version") != 1 or self.file.attrs.get("action_encoding") != ACTION_ENCODING:
                raise ValueError("Dataset must contain the native RL incremental-action encoding")
            self.manifest = json.loads(self.file.attrs["manifest"])
            self.keys, self.ends = [], []
            self.obs_dim = self.action_dim = None
            total = 0
            for key in sorted(self.file):
                group = self.file[key]
                if not group.attrs.get("complete") or (successes_only and not group.attrs.get("success")):
                    continue
                obs, action = group["obs"], group["action"]
                if obs.ndim != 2 or action.ndim != 2 or len(obs) != len(action):
                    raise ValueError(f"Invalid episode shapes: {key}")
                dimensions = (obs.shape[1], action.shape[1])
                if self.obs_dim is not None and dimensions != (self.obs_dim, self.action_dim):
                    raise ValueError("Mixed observation/action dimensions")
                self.obs_dim, self.action_dim = dimensions
                samples = max(0, len(obs) - horizon + 1)
                if samples:
                    total += samples
                    self.keys.append(key)
                    self.ends.append(total)
            if not total:
                raise ValueError("No eligible full action chunks. Collect successful episodes before diffusion pretraining")
            expected_obs = tuple(self.manifest["observations"]["policy"])
            if expected_obs != (self.obs_dim,) or sum(self.manifest["actions"].values()) != self.action_dim:
                raise ValueError("Dataset dimensions do not match its manifest")
        except BaseException:
            self.file.close()
            raise

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = bisect_right(self.ends, index)
        start = index - (self.ends[episode - 1] if episode else 0)
        group = self.file[self.keys[episode]]
        obs = torch.from_numpy(group["obs"][start].astype(np.float32))
        actions = torch.from_numpy(group["action"][start:start + self.horizon].astype(np.float32))
        if not torch.isfinite(obs).all() or not torch.isfinite(actions).all() or actions.abs().max() > 1.000001:
            raise ValueError("Invalid demonstration values")
        return obs, actions

    def fit_normalizer(self, normalizer):
        device = normalizer.mean.device
        for key in self.keys:
            data = self.file[key]["obs"]
            for start in range(0, len(data), 4096):
                obs = torch.from_numpy(data[start:start + 4096].astype(np.float32)).to(device)
                if not torch.isfinite(obs).all():
                    raise ValueError("Non-finite demonstration observations")
                normalizer.update(obs)

    def close(self):
        self.file.close()
