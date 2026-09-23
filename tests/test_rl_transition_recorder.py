"""The Quest RL file must preserve SAC transition alignment and episode labels."""

import h5py
import numpy as np
import pytest

from kuavo_isaaclab_scene.recording.rl_transition_recorder import RlTransitionRecorder


def _sample(step, terminal=False):
    return {
        "actor_obs": np.array([step, step + 0.1], dtype=np.float32),
        "critic_obs": np.array([step, step + 0.1, 3.0], dtype=np.float32),
        "action": np.array([step * 0.2], dtype=np.float32),
        "reward": np.float32(step + 0.5),
        "next_actor_obs": np.array([step + 1, step + 1.1], dtype=np.float32),
        "next_critic_obs": np.array([step + 1, step + 1.1, 3.0], dtype=np.float32),
        "terminated": np.bool_(terminal),
        "truncated": np.bool_(False),
        "success": np.bool_(terminal),
        "unsafe": np.bool_(False),
        "sim_time_s": np.float64(step / 30),
    }


def test_rl_transition_file_preserves_pre_and_post_step_observations(tmp_path):
    path = tmp_path / "demo.hdf5"
    recorder = RlTransitionRecorder(path, {"skill": "grasp", "action_dim": 1})
    recorder.start_episode()
    recorder.append(_sample(0))
    recorder.append(_sample(1, terminal=True))
    recorder.finish_episode(success=True, reason="success")
    recorder.close()

    with h5py.File(path) as file:
        assert file.attrs["format"] == "kuavo_v2_grasp_sac_transitions"
        episode = file["episodes/episode_000000"]
        transitions = episode["transitions"]
        assert episode.attrs["success"]
        assert episode.attrs["num_transitions"] == 2
        np.testing.assert_array_equal(transitions["next_actor_obs"][0], transitions["actor_obs"][1])
        assert transitions["terminated"][:].tolist() == [False, True]
        assert transitions["action"].shape == (2, 1)

    with pytest.raises(FileExistsError):
        RlTransitionRecorder(path, {})
