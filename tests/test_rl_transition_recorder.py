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


def test_completed_batch_matches_streamed_values_and_can_mix_with_streaming(tmp_path):
    rows = [_sample(i, terminal=i == 66) for i in range(67)]
    paths = [tmp_path / 'stream.hdf5', tmp_path / 'batch.hdf5']
    for i, path in enumerate(paths):
        recorder = RlTransitionRecorder(path, {'actor_obs_dim': 2, 'action_dim': 1})
        recorder.start_episode()
        if i:
            recorder.append_many(rows[:32])
            recorder.append(rows[32])
            recorder.append_many(rows[33:])
        else:
            for row in rows:
                recorder.append(row)
        recorder.finish_episode(success=True, reason='success')
        recorder.close()
    with h5py.File(paths[0]) as first, h5py.File(paths[1]) as second:
        a = first['episodes/episode_000000']; b = second['episodes/episode_000000']
        assert dict(a.attrs) == dict(b.attrs)
        for name in a['transitions']:
            np.testing.assert_array_equal(a['transitions'][name][:], b['transitions'][name][:])
            assert a['transitions'][name].dtype == b['transitions'][name].dtype
            assert b['transitions'][name].compression == 'lzf'


def test_bad_final_batch_row_cannot_partially_append_an_episode(tmp_path):
    recorder = RlTransitionRecorder(tmp_path / 'bad_batch.hdf5', {'action_dim': 1})
    recorder.start_episode()
    recorder.append_many([_sample(0), _sample(1)])
    bad = _sample(3); bad['action'] = np.zeros(2)
    with pytest.raises(ValueError, match='must have shape'):
        recorder.append_many([_sample(2), bad])
    assert recorder.count == 2
    assert all(len(value) == 2 for value in recorder.episode['transitions'].values())
    recorder.close()


def test_initial_seed_preserves_flap_joints_and_pending_drive_targets(tmp_path):
    path = tmp_path / "seed.hdf5"
    q = np.array([.3, -.2], dtype=np.float32)
    state = {
        "scene": {"articulation": {"box": {
            "joint_position": q, "joint_velocity": np.array([.01, -.01]),
        }}},
        "drive_targets": {"robot": {"joint_position": q + .05}},
        "logical_boxes": {"active": np.array([True, False])},
    }
    recorder = RlTransitionRecorder(path, {})
    recorder.start_episode(initial_state=state)
    q[:] = 99  # The physical seed must survive reused simulator buffers.
    recorder.append(_sample(0, terminal=True))
    recorder.finish_episode(success=True, reason="success")
    recorder.close()
    with h5py.File(path) as file:
        seed = file["episodes/episode_000000/initial_state"]
        assert seed.attrs["capture_timing"] == "before_first_recorded_action"
        assert not seed.attrs["includes_physx_internal_state"]
        np.testing.assert_allclose(seed["scene/articulation/box/joint_position"][:], [.3, -.2])
        np.testing.assert_allclose(seed["drive_targets/robot/joint_position"][:], [.35, -.15])
        assert file.attrs["format_version"] == 1


@pytest.mark.parametrize("state", [{"q": [np.nan]}, {"q": "missing"}, {"../q": [0]}])
def test_invalid_seed_does_not_create_episode(tmp_path, state):
    recorder = RlTransitionRecorder(tmp_path / "invalid.hdf5", {})
    with pytest.raises(ValueError):
        recorder.start_episode(initial_state=state)
    assert not recorder.recording
    assert len(recorder.episodes) == 0
    recorder.close()
