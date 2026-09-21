import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kuavo_real_control.contract import ARM_JOINT_NAMES, load_config
from kuavo_real_control.prepare import rl_actions_to_trajectory, teleop_hdf5_to_trajectory
from kuavo_real_control.safety import SafetyError, SafetySupervisor
from kuavo_real_control.trajectory import load_trajectory, save_trajectory


CONFIG = ROOT / "config" / "s63.json"


class ContractTests(unittest.TestCase):
    def test_verified_s63_contract(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.arm_joint_names, ARM_JOINT_NAMES)
        self.assertEqual(cfg.sensor_joint_indices, tuple(range(4, 18)))
        self.assertEqual(cfg.rl_delta_scale_rad, 0.035)
        self.assertEqual(cfg.ros["arm_command_topic"], "/kuavo_arm_traj")


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        self.nominal = (np.asarray(self.cfg.safe_lower_rad) + np.asarray(self.cfg.safe_upper_rad)) / 2.0

    def test_rl_delta_is_integrated_once_per_sample(self):
        actions = np.zeros((3, 16), dtype=np.float64)
        actions[:, 0] = [1.0, 1.0, -0.5]
        actions[:, 14:] = [[0, 0], [1, 0], [1, 1]]
        trajectory = rl_actions_to_trajectory(
            actions, np.arange(3) / 30.0, source="test", delta_scale_rad=0.035
        )
        target = trajectory.absolute_targets(self.nominal, self.cfg.rl_delta_scale_rad)
        np.testing.assert_allclose(target[:, 0] - self.nominal[0], [0.035, 0.07, 0.0525])
        np.testing.assert_array_equal(trajectory.gripper_close, actions[:, 14:])

    def test_portable_round_trip_has_no_pickle(self):
        trajectory = rl_actions_to_trajectory(
            np.zeros((2, 14)), np.array([0.0, 1.0 / 30.0]), source="test", delta_scale_rad=0.035
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trajectory.npz"
            save_trajectory(path, trajectory)
            loaded = load_trajectory(path)
        self.assertEqual(loaded.kind, "normalized_joint_delta")
        np.testing.assert_array_equal(loaded.arm_values, trajectory.arm_values)

    def test_rejects_out_of_range_normalized_action(self):
        actions = np.zeros((1, 14))
        actions[0, 3] = 1.01
        with self.assertRaisesRegex(ValueError, "Normalized"):
            rl_actions_to_trajectory(actions, np.array([0.0]), source="test", delta_scale_rad=0.035)

    def test_rejects_non_binary_gripper_without_rounding(self):
        actions = np.zeros((1, 16))
        actions[0, 14] = 0.6
        with self.assertRaisesRegex(ValueError, "binary"):
            rl_actions_to_trajectory(actions, np.array([0.0]), source="test", delta_scale_rad=0.035)


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(CONFIG)
        self.nominal = (np.asarray(self.cfg.safe_lower_rad) + np.asarray(self.cfg.safe_upper_rad)) / 2.0
        self.safety = SafetySupervisor(self.cfg)
        self.safety.synchronize(self.nominal)

    def test_velocity_and_acceleration_are_limited(self):
        target = self.nominal.copy()
        target[0] += 0.1
        command, velocity = self.safety.project(target, self.nominal, 1.0 / 30.0)
        self.assertLessEqual(abs(velocity[0]), self.cfg.max_command_acceleration_rad_s2 / 30.0 + 1e-12)
        self.assertLess(command[0], target[0])

    def test_rejects_source_jump(self):
        target = self.nominal.copy()
        target[0] += self.cfg.max_source_step_rad + 0.01
        with self.assertRaisesRegex(SafetyError, "source target jumped"):
            self.safety.project(target, self.nominal, 1.0 / 30.0)

    def test_rejects_limit_instead_of_clipping(self):
        target = self.nominal.copy()
        target[3] = self.cfg.joint_upper_rad[3]
        with self.assertRaisesRegex(SafetyError, "safe limit"):
            self.safety.project(target, self.nominal, 1.0 / 30.0)

    def test_persistent_tracking_error_stops(self):
        target = self.nominal.copy()
        measured = self.nominal.copy()
        for _ in range(4):
            target[0] += 0.03
            self.safety.project(target, measured, 0.05)
        measured[0] -= 0.4
        with self.assertRaises(SafetyError):
            for _ in range(6):
                self.safety.project(target, measured, 0.05)


@unittest.skipUnless(__import__("importlib").util.find_spec("h5py"), "h5py not installed")
class Hdf5PreparationTests(unittest.TestCase):
    def test_extracts_named_arm_state_and_binary_gripper(self):
        import h5py

        names = ["head_dummy", *ARM_JOINT_NAMES, "finger_dummy"]
        layout = ["unused_0", "unused_1", "unused_2", "left_gripper_close", "right_gripper_close"]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quest.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["format"] = "kuavo_quest_teleop_hdf5"
                episode = handle.create_group("data/demo_00000")
                episode.attrs["joint_names"] = json.dumps(names)
                episode.attrs["action_layout"] = ",".join(layout)
                episode.attrs["control_dt"] = 1.0 / 30.0
                samples = episode.create_group("samples")
                state = np.arange(3 * len(names), dtype=np.float32).reshape(3, len(names)) / 100.0
                samples.create_dataset("robot_joint_position", data=state)
                samples.create_dataset("sim_time_s", data=[4.0, 4.1, 4.2])
                action = np.zeros((3, len(layout)), dtype=np.float32)
                action[:, -2:] = [[0, 1], [1, 1], [1, 0]]
                samples.create_dataset("action", data=action)
            trajectory = teleop_hdf5_to_trajectory(path)
        np.testing.assert_allclose(trajectory.timestamps_s, [0.0, 0.1, 0.2])
        np.testing.assert_allclose(trajectory.arm_values, state[:, 1:15])
        np.testing.assert_array_equal(trajectory.gripper_close, action[:, -2:])


if __name__ == "__main__":
    unittest.main()
