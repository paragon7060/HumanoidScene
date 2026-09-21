"""ROS1 adapter for the verified S63 wheel-WBC arm target interface."""

from __future__ import annotations

import json
import math
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

import numpy as np

from .contract import ControlConfig
from .safety import SafetyError, SafetySupervisor
from .trajectory import Trajectory


class Ros1ArmRuntime:
    """Read live state and optionally publish safety-filtered WBC arm targets."""

    def __init__(
        self,
        config: ControlConfig,
        trajectory: Trajectory,
        *,
        enable_motion: bool,
        enable_gripper: bool,
        log_path: Path,
        dry_run_speed: float = 1.0,
    ):
        try:
            import rospy
            from kuavo_msgs.msg import sensorsData
            from sensor_msgs.msg import JointState
            from std_msgs.msg import Bool
        except ImportError as exc:
            raise RuntimeError(
                "Run this entry point after sourcing ROS Noetic and the active Kuavo workspace"
            ) from exc
        self.rospy = rospy
        self._JointState = JointState
        self.config = config
        self.trajectory = trajectory.validate()
        self.enable_motion = bool(enable_motion)
        self.enable_gripper = bool(enable_gripper)
        if self.enable_gripper and not self.enable_motion:
            raise ValueError("Gripper motion cannot be enabled during arm dry-run")
        if not math.isfinite(dry_run_speed) or dry_run_speed <= 0.0:
            raise ValueError("dry_run_speed must be finite and positive")
        if self.enable_motion and dry_run_speed != 1.0:
            raise ValueError("Live robot execution must use dry_run_speed=1")
        self.dry_run_speed = float(dry_run_speed)
        self._lock = threading.RLock()
        self._state_q: Optional[np.ndarray] = None
        self._state_v: Optional[np.ndarray] = None
        self._state_arrival = 0.0
        self._enabled: Optional[bool] = None
        self._disabled: Optional[bool] = None
        self._foreign_command = None
        self._publisher = None
        self._armed = False
        self._last_gripper = None
        self._node_name = ""
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self._log_path.open("x")

        rospy.init_node("humanoid_scene_real_control", anonymous=True)
        self._node_name = rospy.get_name()
        topics = config.ros
        self._subscriptions = [
            rospy.Subscriber(topics["sensor_topic"], sensorsData, self._sensor_callback, queue_size=5),
            rospy.Subscriber(topics["enable_topic"], Bool, self._enable_callback, queue_size=2),
            rospy.Subscriber(topics["disabled_topic"], Bool, self._disabled_callback, queue_size=2),
            rospy.Subscriber(topics["arm_command_topic"], JointState, self._command_callback, queue_size=20),
        ]

    def _write_log(self, kind: str, **values) -> None:
        row = dict(kind=kind, monotonic_s=time.monotonic(), unix_s=time.time(), **values)
        self._log.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        self._log.flush()

    def _sensor_callback(self, message) -> None:
        now = time.monotonic()
        q = np.asarray(message.joint_data.joint_q, dtype=np.float64)
        v = np.asarray(message.joint_data.joint_v, dtype=np.float64)
        indices = np.asarray(self.config.sensor_joint_indices, dtype=np.int64)
        if q.size <= int(indices[-1]) or v.size <= int(indices[-1]):
            return
        arm_q = q[indices]
        arm_v = v[indices]
        if not np.isfinite(arm_q).all() or not np.isfinite(arm_v).all():
            return
        with self._lock:
            self._state_q = arm_q
            self._state_v = arm_v
            self._state_arrival = now

    def _enable_callback(self, message) -> None:
        with self._lock:
            self._enabled = bool(message.data)

    def _disabled_callback(self, message) -> None:
        with self._lock:
            self._disabled = bool(message.data)

    def _command_callback(self, message) -> None:
        caller = getattr(message, "_connection_header", {}).get("callerid", "")
        if caller == self._node_name:
            return
        with self._lock:
            self._foreign_command = (time.monotonic(), caller)

    def _snapshot(self) -> Tuple[np.ndarray, np.ndarray]:
        with self._lock:
            if self._state_q is None or self._state_v is None:
                raise SafetyError("No valid /sensors_data_raw arm state received")
            age = time.monotonic() - self._state_arrival
            if age > self.config.state_timeout_s:
                raise SafetyError("Measured arm state is stale by {:.3f} s".format(age))
            if self._enabled is not True:
                raise SafetyError("/enable_control_state must be true")
            if self._disabled is not False:
                raise SafetyError("/robot_disabled_flag must be false")
            return self._state_q.copy(), self._state_v.copy()

    def _wait_for_preflight(self) -> Tuple[np.ndarray, np.ndarray]:
        deadline = time.monotonic() + 5.0
        last_error = None
        while time.monotonic() < deadline and not self.rospy.is_shutdown():
            try:
                state = self._snapshot()
                break
            except SafetyError as exc:
                last_error = exc
                time.sleep(0.02)
        else:
            raise SafetyError("ROS preflight timed out: {}".format(last_error))

        version = self.rospy.get_param("/robot_version", None)
        if int(version) != 63:
            raise SafetyError("Expected /robot_version=63, got {!r}".format(version))
        configuration = self.rospy.get_param("/kuavo_configuration", {})
        if isinstance(configuration, str):
            configuration = json.loads(configuration)
        expected = (configuration.get("NUM_JOINT"), configuration.get("NUM_ARM_JOINT"), configuration.get("NUM_HEAD_JOINT"))
        if expected != (20, 14, 2):
            raise SafetyError("Expected S63 joint counts (20,14,2), got {}".format(expected))

        quiet_deadline = time.monotonic() + self.config.external_command_quiet_s
        while time.monotonic() < quiet_deadline:
            with self._lock:
                foreign = self._foreign_command
            if foreign is not None and time.monotonic() - foreign[0] < self.config.external_command_quiet_s:
                raise SafetyError("Foreign /kuavo_arm_traj command received from {}".format(foreign[1]))
            self._snapshot()
            time.sleep(0.02)
        return self._snapshot()

    def _prevalidate_targets(self, initial_q: np.ndarray, targets: np.ndarray) -> None:
        """Reject the complete source before creating a publisher or changing mode."""
        validator = SafetySupervisor(self.config)
        validator.synchronize(initial_q)
        validator.check_first_target(targets[0])
        previous_time = None
        for index, (stamp, target) in enumerate(zip(self.trajectory.timestamps_s, targets)):
            dt_s = 1.0 / self.config.publish_hz if previous_time is None else float(stamp - previous_time)
            try:
                validator.project(target, validator.command, dt_s)
            except SafetyError as exc:
                raise SafetyError("trajectory sample {} failed prevalidation: {}".format(index, exc))
            previous_time = float(stamp)

    def _change_arm_mode(self, mode: int) -> None:
        from kuavo_msgs.srv import changeArmCtrlMode, changeArmCtrlModeRequest

        name = self.config.ros["arm_mode_service"]
        self.rospy.wait_for_service(name, timeout=3.0)
        request = changeArmCtrlModeRequest()
        request.control_mode = int(mode)
        response = self.rospy.ServiceProxy(name, changeArmCtrlMode)(request)
        self._write_log(
            "arm_mode",
            requested_mode=mode,
            result=bool(response.result),
            actual_mode=int(response.mode),
            message=str(response.message),
        )
        if not response.result or int(response.mode) != mode:
            raise SafetyError("WBC rejected arm mode {}: {}".format(mode, response.message))

    def _publish(self, q_rad: np.ndarray, v_rad_s: np.ndarray, phase: str, source_index: int) -> None:
        if self._publisher is None:
            raise RuntimeError("Publisher is not initialized")
        message = self._JointState()
        message.header.stamp = self.rospy.Time.now()
        message.header.frame_id = self._node_name
        message.name = list(self.config.arm_joint_names)
        # The active S63 ArmController::storeMode2Target converts these fields
        # from degrees to radians despite sensor_msgs/JointState conventions.
        message.position = np.rad2deg(q_rad).tolist()
        message.velocity = np.rad2deg(v_rad_s).tolist()
        self._publisher.publish(message)
        self._write_log(
            "command",
            phase=phase,
            source_index=source_index,
            position_rad=q_rad.tolist(),
            velocity_rad_s=v_rad_s.tolist(),
            ros_position_unit="degree",
            ros_velocity_unit="degree/s",
        )

    def _command_gripper(self, close_values: np.ndarray) -> None:
        values = tuple(int(value) for value in np.asarray(close_values).tolist())
        if values == self._last_gripper:
            return
        if values not in ((0, 0), (0, 1), (1, 0), (1, 1)):
            raise SafetyError("Gripper command must be binary 0=open, 1=close")
        from kuavo_msgs.srv import controlLejuClaw, controlLejuClawRequest

        name = self.config.ros["claw_service"]
        self.rospy.wait_for_service(name, timeout=2.0)
        request = controlLejuClawRequest()
        request.data.name = ["left_claw", "right_claw"]
        request.data.position = [100.0 * value for value in values]
        request.data.velocity = [25.0, 25.0]
        request.data.effort = [1.0, 1.0]
        response = self.rospy.ServiceProxy(name, controlLejuClaw)(request)
        self._write_log("gripper", close=list(values), success=bool(response.success), message=str(response.message))
        if not response.success:
            raise SafetyError("Leju claw service rejected command: {}".format(response.message))
        self._last_gripper = values

    def run(self) -> None:
        initial_q, initial_v = self._wait_for_preflight()
        supervisor = SafetySupervisor(self.config)
        supervisor.synchronize(initial_q)
        targets = self.trajectory.absolute_targets(initial_q, self.config.rl_delta_scale_rad)
        if self.trajectory.kind == "normalized_joint_delta":
            expected_scale = float(self.trajectory.metadata.get("delta_scale_rad", float("nan")))
            if not math.isclose(expected_scale, self.config.rl_delta_scale_rad, rel_tol=0.0, abs_tol=1e-12):
                raise SafetyError(
                    "RL trajectory delta scale {:.6f} differs from robot config {:.6f}".format(
                        expected_scale, self.config.rl_delta_scale_rad
                    )
                )
        self._prevalidate_targets(initial_q, targets)
        supervisor.check_first_target(targets[0])
        self._write_log(
            "preflight",
            enable_motion=self.enable_motion,
            enable_gripper=self.enable_gripper,
            trajectory_kind=self.trajectory.kind,
            samples=len(self.trajectory.timestamps_s),
            duration_s=self.trajectory.duration_s,
            initial_position_rad=initial_q.tolist(),
            initial_velocity_rad_s=initial_v.tolist(),
        )

        period = 1.0 / self.config.publish_hz
        if self.enable_motion:
            self._publisher = self.rospy.Publisher(
                self.config.ros["arm_command_topic"], self._JointState, queue_size=1, latch=False
            )
            deadline = time.monotonic() + 2.0
            while self._publisher.get_num_connections() == 0:
                if time.monotonic() > deadline:
                    raise SafetyError("WBC did not subscribe to arm command publisher")
                time.sleep(0.01)
            self._change_arm_mode(2)
            self._armed = True
            hold_end = time.monotonic() + self.config.handover_hold_s
            while time.monotonic() < hold_end and not self.rospy.is_shutdown():
                q, _ = self._snapshot()
                command, velocity = supervisor.hold()
                self._publish(command, velocity, "handover_hold", -1)
                time.sleep(period)

        started = time.monotonic()
        time_scale = 1.0 if self.enable_motion else self.dry_run_speed
        loop_period = period / time_scale
        last_tick = started - loop_period
        last_index = -1
        while not self.rospy.is_shutdown():
            now = time.monotonic()
            dt_s = now - last_tick
            if dt_s < loop_period:
                time.sleep(max(0.0, loop_period - dt_s))
                continue
            source_dt_s = dt_s * time_scale
            if source_dt_s > 0.2:
                raise SafetyError("Control loop stalled for {:.3f} s".format(dt_s))
            source_elapsed = (now - started) * time_scale
            index = int(np.searchsorted(self.trajectory.timestamps_s, source_elapsed, side="right") - 1)
            index = max(0, min(index, len(targets) - 1))
            if index != last_index and last_index >= 0 and index != last_index + 1:
                raise SafetyError("Control timing skipped trajectory samples {} -> {}".format(last_index, index))
            measured_q, _ = self._snapshot()
            shadow_measured = measured_q if self.enable_motion else supervisor.command
            command, velocity = supervisor.project(targets[index], shadow_measured, source_dt_s)
            if self.enable_motion:
                with self._lock:
                    foreign = self._foreign_command
                if foreign is not None and now - foreign[0] < self.config.external_command_quiet_s:
                    raise SafetyError("Foreign arm command received during execution from {}".format(foreign[1]))
                self._publish(command, velocity, "source", index)
                if self.enable_gripper and self.trajectory.gripper_close is not None:
                    self._command_gripper(self.trajectory.gripper_close[index])
            elif index != last_index:
                self._write_log(
                    "dry_run_command",
                    source_index=index,
                    position_rad=command.tolist(),
                    velocity_rad_s=velocity.tolist(),
                )
            last_index = index
            last_tick = now
            if source_elapsed >= self.trajectory.duration_s:
                break

        if self.enable_motion:
            hold_end = time.monotonic() + self.config.finish_hold_s
            while time.monotonic() < hold_end and not self.rospy.is_shutdown():
                self._snapshot()
                command, velocity = supervisor.hold()
                self._publish(command, velocity, "finish_hold", last_index)
                time.sleep(period)
        self._write_log("complete", source_index=last_index)

    def close(self) -> None:
        if self._armed:
            try:
                # Mode 0 locks the controller's current command. Do not return
                # to mode 1 automatically because that would initiate a home move.
                self._change_arm_mode(0)
            except Exception as exc:
                self._write_log("shutdown_error", error=str(exc))
            self._armed = False
        self._log.close()

    def __enter__(self) -> "Ros1ArmRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is not None:
            try:
                self._write_log("error", error=str(exc), error_type=type(exc).__name__)
            except Exception:
                pass
        self.close()
