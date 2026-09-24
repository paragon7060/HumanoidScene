"""Exported demonstrations must stay bit-identical and contract-checked."""

import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest

from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder


spec = importlib.util.spec_from_file_location(
    "export_demo_subset",
    Path(__file__).resolve().parents[1] / "scripts/rl/export_demo_subset.py")
export_demo_subset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export_demo_subset)


MANIFEST = {
    "action_dim": 1, "actor_obs_dim": 2, "critic_obs_dim": 3,
    "control_dt": 1 / 30, "robot_model": "s63", "controller_mapping": "absolute",
    "seed": 42,
}


def _sample(step):
    return {
        "actor_obs": np.array([step, step + 0.1], dtype=np.float32),
        "critic_obs": np.array([step, step + 0.1, 3.0], dtype=np.float32),
        "action": np.array([step * 0.2], dtype=np.float32),
        "reward": np.float32(step + 0.5),
        "next_actor_obs": np.array([step + 1, step + 1.1], dtype=np.float32),
        "next_critic_obs": np.array([step + 1, step + 1.1, 3.0], dtype=np.float32),
        "terminated": np.bool_(True),
        "truncated": np.bool_(False),
        "success": np.bool_(True),
        "unsafe": np.bool_(False),
        "sim_time_s": np.float64(step / 30),
    }


def _write(path, episodes, manifest=None):
    recorder = RlTransitionRecorder(path, manifest or MANIFEST)
    for success, reason, steps in episodes:
        recorder.start_episode()
        for step in range(steps):
            recorder.append(_sample(step))
        if reason is not None:
            recorder.finish_episode(success=success, reason=reason)
    recorder.close()
    return path


def test_only_successful_complete_episodes_are_copied_without_edits(tmp_path):
    source = _write(tmp_path / "run.hdf5", [
        (False, "unsafe", 3), (True, "success", 2), (False, "time_out", 4)])
    output = tmp_path / "demos.hdf5"

    summary = export_demo_subset.export([source], output)

    assert [row["episode"] for row in summary] == ["episode_000001"]
    with h5py.File(source) as original, h5py.File(output) as exported:
        assert list(exported["episodes"]) == ["episode_000000"]
        attrs = exported["episodes/episode_000000"].attrs
        assert attrs["success"] and attrs["end_reason"] == "success"
        assert attrs["source_episode"] == "episode_000001"
        for field in original["episodes/episode_000001/transitions"]:
            np.testing.assert_array_equal(
                original[f"episodes/episode_000001/transitions/{field}"][...],
                exported[f"episodes/episode_000000/transitions/{field}"][...])


def test_incompatible_contracts_are_refused_and_described(tmp_path):
    first = _write(tmp_path / "a.hdf5", [(True, "success", 2)])
    other = _write(tmp_path / "b.hdf5", [(True, "success", 2)],
                   manifest={**MANIFEST, "control_dt": 1 / 60, "robot_model": "s56"})
    with pytest.raises(ValueError, match="control_dt"):
        export_demo_subset.export([first, other], tmp_path / "merged.hdf5")


def test_merged_manifest_marks_only_the_descriptive_keys_that_differ(tmp_path):
    first = _write(tmp_path / "a.hdf5", [(True, "success", 2)])
    other = _write(tmp_path / "b.hdf5", [(True, "success", 3)],
                   manifest={**MANIFEST, "controller_mapping": "scaled"})
    output = tmp_path / "merged.hdf5"

    summary = export_demo_subset.export([first, other], output)

    assert [row["controller_mapping"] for row in summary] == ["absolute", "scaled"]
    with h5py.File(output) as exported:
        import json
        manifest = json.loads(exported.attrs["manifest_json"])
    assert manifest["controller_mapping"] == "mixed"
    assert manifest["action_dim"] == 1


def test_an_interrupted_episode_is_skipped_unless_requested(tmp_path):
    source = _write(tmp_path / "run.hdf5", [(True, None, 2)])
    with pytest.raises(ValueError, match="No episode matched"):
        export_demo_subset.export([source], tmp_path / "empty.hdf5")
    summary = export_demo_subset.export(
        [source], tmp_path / "kept.hdf5", success_only=False, allow_incomplete=True)
    assert summary[0]["transitions"] == 2
