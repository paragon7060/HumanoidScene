import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kuavo_real_control.contract import ARM_JOINT_NAMES, load_config
from kuavo_real_control.prepare import (
    retime_joint_position_trajectory,
    rl_actions_to_trajectory,
    teleop_hdf5_to_trajectory,
)
from kuavo_real_control.replay import quintic_approach
from kuavo_real_control.safety import SafetyError, SafetySupervisor
from kuavo_real_control.trajectory import Trajectory, load_trajectory, save_trajectory


CONFIG = ROOT / "config" / "s63.json"


class ContractTests(unittest.TestCase):
    def test_verified_s63_contract(self):
        cfg = load_config(CONFIG)
        self.assertEqual(cfg.arm_joint_names, ARM_JOINT_NAMES)
        self.assertEqual(cfg.sensor_joint_indices, tuple(range(4, 18)))
        self.assertEqual(cfg.rl_delta_scale_rad, 0.035)
        self.assertEqual(
            cfg.operational_limits_provenance,
            "observed_real_vr_p99_operational_cap_not_hardware_rating",
        )
        self.assertEqual(cfg.max_command_velocity_rad_s, 1.5)
        self.assertEqual(cfg.max_command_acceleration_rad_s2, 8.0)
        self.assertAlmostEqual(
            cfg.operational_limits_evidence["joint_command_abs_velocity_p99_rad_s"], 1.495663
        )
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

    def test_retime_slows_and_filters_fast_joint_path(self):
        arm_values = np.repeat(self.nominal[None, :], 4, axis=0)
        arm_values[:, 0] += [0.0, 0.08, 0.16, 0.24]
        trajectory = Trajectory(
            "joint_position_rad",
            np.arange(4, dtype=np.float64) / 30.0,
            arm_values,
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64),
            {
                "robot_model": "s63",
                "arm_joint_names": list(ARM_JOINT_NAMES),
                "source_type": "test",
                "deployment_ready": False,
            },
        ).validate()
        retimed = retime_joint_position_trajectory(trajectory, self.cfg)
        period = 1.0 / self.cfg.publish_hz
        velocity = np.diff(retimed.arm_values, axis=0) / period
        acceleration = np.diff(velocity, axis=0) / period
        self.assertTrue(retimed.metadata["deployment_ready"])
        self.assertGreater(retimed.duration_s, trajectory.duration_s)
        self.assertLessEqual(np.max(np.abs(velocity)), self.cfg.max_command_velocity_rad_s + 1e-9)
        self.assertLessEqual(
            np.max(np.abs(acceleration)), self.cfg.max_command_acceleration_rad_s2 + 1e-9
        )
        np.testing.assert_allclose(retimed.arm_values[-1], arm_values[-1])
        np.testing.assert_array_equal(retimed.gripper_close[-1], [0, 1])


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

    def test_validated_target_is_not_filtered_twice(self):
        period = 1.0 / self.cfg.publish_hz
        target = self.nominal.copy()
        target[0] += 0.001
        command, velocity = self.safety.accept_validated_target(target, self.nominal, period)
        np.testing.assert_array_equal(command, target)
        self.assertAlmostEqual(velocity[0], 0.03)


class ReplayTests(unittest.TestCase):
    def test_quintic_approach_has_exact_endpoints_and_near_zero_endpoint_velocity(self):
        start = np.zeros(14)
        target = np.linspace(-0.2, 0.2, 14)
        timestamps, positions = quintic_approach(start, target, 3.0, 30.0)
        np.testing.assert_array_equal(positions[0], start)
        np.testing.assert_array_equal(positions[-1], target)
        self.assertAlmostEqual(timestamps[-1], 3.0)
        first_velocity = (positions[1] - positions[0]) / (timestamps[1] - timestamps[0])
        last_velocity = (positions[-1] - positions[-2]) / (timestamps[-1] - timestamps[-2])
        self.assertLess(np.max(np.abs(first_velocity)), 0.001)
        self.assertLess(np.max(np.abs(last_velocity)), 0.001)


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
                episode.attrs["success"] = True
                episode.attrs["end_reason"] = "operator_success"
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
        self.assertEqual(trajectory.metadata["source_command_source"], "measured_state")

    def test_prefers_named_self_collision_safe_target(self):
        import h5py

        safe_names = ["wheel_dummy", *ARM_JOINT_NAMES, "finger_dummy"]
        measured_names = list(ARM_JOINT_NAMES)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quest.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["format"] = "kuavo_quest_teleop_hdf5"
                episode = handle.create_group("data/demo_00000")
                episode.attrs["success"] = True
                episode.attrs["end_reason"] = "operator_success"
                episode.attrs["self_collision"] = True
                episode.attrs["self_collision_clearance_m"] = 0.003
                episode.attrs["self_collision_joint_names"] = json.dumps(safe_names)
                episode.attrs["joint_names"] = json.dumps(measured_names)
                episode.attrs["action_layout"] = "left_gripper_close,right_gripper_close"
                samples = episode.create_group("samples")
                safe = np.arange(3 * len(safe_names), dtype=np.float32).reshape(3, -1) / 100.0
                samples.create_dataset("self_collision_safe_joint_target", data=safe)
                samples.create_dataset("self_collision_modified", data=[0, 1, 0])
                samples.create_dataset("self_collision_minimum_distance_m", data=[0.004, 0.0031, 0.005])
                samples.create_dataset("robot_joint_position", data=np.zeros((3, 14)))
                samples.create_dataset("sim_time_s", data=[4.0, 4.1, 4.2])
                samples.create_dataset("action", data=np.zeros((3, 2)))
            trajectory = teleop_hdf5_to_trajectory(path)
        np.testing.assert_allclose(trajectory.arm_values, safe[:, 1:15])
        self.assertEqual(
            trajectory.metadata["source_command_source"], "self_collision_safe_target"
        )
        self.assertEqual(trajectory.metadata["self_collision_modified_samples"], 1)

    def test_rejects_unsuccessful_episode_by_default(self):
        import h5py

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "failed.hdf5"
            with h5py.File(path, "w") as handle:
                handle.attrs["format"] = "kuavo_quest_teleop_hdf5"
                episode = handle.create_group("data/demo_00000")
                episode.attrs["success"] = False
                episode.attrs["end_reason"] = "self_collision"
                episode.attrs["joint_names"] = json.dumps(list(ARM_JOINT_NAMES))
                episode.attrs["action_layout"] = "left_gripper_close,right_gripper_close"
                samples = episode.create_group("samples")
                config = load_config(CONFIG)
                nominal = (
                    np.asarray(config.safe_lower_rad) + np.asarray(config.safe_upper_rad)
                ) / 2.0
                samples.create_dataset("robot_joint_position", data=np.repeat(nominal[None, :], 2, axis=0))
                samples.create_dataset("sim_time_s", data=[0.0, 0.1])
                samples.create_dataset("action", data=np.zeros((2, 2)))
            with self.assertRaisesRegex(ValueError, "unsuccessful"):
                teleop_hdf5_to_trajectory(path)
            review = teleop_hdf5_to_trajectory(path, require_success=False)
        self.assertFalse(review.metadata["source_episode_success"])
        reviewed_retiming = retime_joint_position_trajectory(review, load_config(CONFIG))
        self.assertFalse(reviewed_retiming.metadata["deployment_ready"])


if __name__ == "__main__":
    unittest.main()
