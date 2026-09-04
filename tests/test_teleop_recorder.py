import h5py
import numpy as np
import pytest
import subprocess
import sys

from kuavo_isaaclab_scene.recording.teleop_recorder import TeleopHdf5EpisodeRecorder, TeleopHdf5Recorder, new_session_path
from kuavo_isaaclab_scene.recording.teleop_lerobot_recorder import build_lerobot_features, sample_to_lerobot_frame


def test_recorder_streams_episode_to_hdf5(tmp_path):
    path = tmp_path / "demo.hdf5"
    with TeleopHdf5Recorder(path, flush_every=1) as recorder:
        assert recorder.start_episode({"joint_names": ["j1", "j2"]}) == "demo_00000"
        recorder.append(
            {
                "action": np.zeros(14, dtype=np.float32),
                "head_rgb": np.zeros((8, 12, 3), dtype=np.uint8),
            }
        )
        assert recorder.finish_episode(success=True, reason="test") == "demo_00000"

    with h5py.File(path, "r") as stream:
        demo = stream["data/demo_00000"]
        assert bool(demo.attrs["success"])
        assert demo.attrs["num_samples"] == 1
        assert demo["samples/action"].shape == (1, 14)
        assert demo["samples/head_rgb"].shape == (1, 8, 12, 3)


def test_session_files_are_distinct_and_existing_recordings_are_preserved(tmp_path):
    first = new_session_path(tmp_path)
    second = new_session_path(tmp_path)
    assert first != second
    with TeleopHdf5Recorder(first) as recorder:
        recorder.start_episode()
        recorder.append({"action": np.array([1.0], dtype=np.float32)})
        recorder.finish_episode(success=False, reason="operator_stop")
    original = first.read_bytes()
    with pytest.raises(FileExistsError):
        TeleopHdf5Recorder(first)
    assert first.read_bytes() == original
    with TeleopHdf5Recorder(second) as recorder:
        assert recorder.episode_count == 0
    assert first.exists() and second.exists()


def test_empty_recording_file_survives_exit_without_python_cleanup(tmp_path):
    path = tmp_path / "interrupted.hdf5"
    subprocess.run(
        [sys.executable, "-c",
         "import os, sys; from kuavo_isaaclab_scene.recording.teleop_recorder import TeleopHdf5Recorder; "
         "recorder = TeleopHdf5Recorder(sys.argv[1]); os._exit(0)", str(path)],
        check=True,
    )
    with h5py.File(path, "r") as stream:
        assert len(stream["data"]) == 0
        assert stream.attrs["format"] == "kuavo_quest_teleop_hdf5"


def test_multiple_attempts_close_separate_files_without_closing_the_session(tmp_path):
    first = tmp_path / "first.hdf5"
    recorder = TeleopHdf5EpisodeRecorder(first)
    assert not first.exists()  # Preview alone creates no dataset.
    recorder.start_episode({"input_mode": "controllers"})
    recorder.append({"action": np.array([1.0])})
    recorder.finish_episode(success=False, reason="operator_stop")
    assert not recorder.recording
    original = first.read_bytes()
    recorder.start_episode({"input_mode": "controllers"})
    second = recorder.path
    assert second != first
    recorder.append({"action": np.array([2.0])})
    recorder.close()  # Interrupted attempt is still preserved.
    assert first.read_bytes() == original
    for path, value, reason in ((first, 1.0, "operator_stop"), (second, 2.0, "process_closed")):
        with h5py.File(path, "r") as f:
            assert len(f["data"]) == 1
            episode = f["data/demo_00000"]
            assert episode.attrs["end_reason"] == reason
            assert episode["samples/action"][0, 0] == value
    with pytest.raises(FileExistsError):
        TeleopHdf5EpisodeRecorder(first)


@pytest.mark.parametrize("record_controllers", [False, True])
def test_lerobot_v3_feature_mapping_uses_policy_and_camera_keys(record_controllers):
    features = build_lerobot_features(
        joint_names=["left_joint", "right_joint"],
        hand_joint_names=["wrist", "index_tip"],
        head_resolution=(12, 8),
        wrist_resolution=(6, 4),
        box_count=1,
        button_joint_count=1,
        record_wrist_cameras=True,
        use_videos=True,
        record_controllers=record_controllers,
    )
    sample = {
        "robot_joint_position": np.zeros(2, dtype=np.float32),
        "robot_joint_velocity": np.zeros(2, dtype=np.float32),
        "left_end_effector_pose_w": np.zeros(7, dtype=np.float32),
        "right_end_effector_pose_w": np.zeros(7, dtype=np.float32),
        "openxr_head_pose": np.zeros(7, dtype=np.float32),
        "openxr_left_hand": np.zeros((2, 7), dtype=np.float32),
        "openxr_right_hand": np.zeros((2, 7), dtype=np.float32),
        "pinch_distance_m": np.zeros(2, dtype=np.float32),
        "tracking_valid": np.ones(3, dtype=np.uint8),
        "sim_time_s": np.float64(0.1),
        "head_rgb": np.zeros((8, 12, 3), dtype=np.uint8),
        "left_wrist_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "right_wrist_rgb": np.zeros((4, 6, 3), dtype=np.uint8),
        "action": np.zeros(14, dtype=np.float32),
        "box_root_pose_w": np.zeros((1, 7), dtype=np.float32),
        "button_joint_position": np.zeros(1, dtype=np.float32),
    }

    if record_controllers:
        sample["openxr_left_controller"] = np.arange(14, dtype=np.float32).reshape(2, 7)
        sample["openxr_right_controller"] = np.arange(14, 28, dtype=np.float32).reshape(2, 7)
    frame = sample_to_lerobot_frame(sample, features)
    if record_controllers:
        np.testing.assert_array_equal(frame["observation.openxr.left_controller"], np.arange(14))
        np.testing.assert_array_equal(frame["observation.openxr.right_controller"], np.arange(14, 28))

    assert set(frame) == set(features)
    assert frame["observation.state"].shape == (2,)
    assert frame["observation.openxr.left_hand"].shape == (14,)
    assert frame["observation.images.head"].shape == (8, 12, 3)
    assert frame["observation.images.left_wrist"].shape == (4, 6, 3)
    assert frame["observation.box_root_pose"].shape == (7,)
    assert features["observation.images.head"]["dtype"] == "video"
    assert frame["next.done"].item() == 0.0


def test_lerobot_action_schema_accepts_gripper_channels():
    action_names = ["left_dx", "right_dx", "left_gripper", "right_gripper"]
    features = build_lerobot_features(
        joint_names=["joint"],
        hand_joint_names=["wrist"],
        head_resolution=(12, 8),
        wrist_resolution=(6, 4),
        box_count=0,
        button_joint_count=0,
        record_wrist_cameras=False,
        use_videos=False,
        action_names=action_names,
    )
    assert features["action"]["shape"] == (4,)
    assert features["action"]["names"] == action_names
