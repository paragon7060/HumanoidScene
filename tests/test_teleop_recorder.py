import h5py
import numpy as np

from kuavo_isaaclab_scene.teleop_recorder import TeleopHdf5Recorder
from kuavo_isaaclab_scene.teleop_lerobot_recorder import build_lerobot_features, sample_to_lerobot_frame


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


def test_lerobot_v3_feature_mapping_uses_policy_and_camera_keys():
    features = build_lerobot_features(
        joint_names=["left_joint", "right_joint"],
        hand_joint_names=["wrist", "index_tip"],
        head_resolution=(12, 8),
        wrist_resolution=(6, 4),
        box_count=1,
        button_joint_count=1,
        record_wrist_cameras=True,
        use_videos=True,
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

    frame = sample_to_lerobot_frame(sample, features)

    assert set(frame) == set(features)
    assert frame["observation.state"].shape == (2,)
    assert frame["observation.openxr.left_hand"].shape == (14,)
    assert frame["observation.images.head"].shape == (8, 12, 3)
    assert frame["observation.images.left_wrist"].shape == (4, 6, 3)
    assert frame["observation.box_root_pose"].shape == (7,)
    assert features["observation.images.head"]["dtype"] == "video"
    assert frame["next.done"].item() == 0.0
