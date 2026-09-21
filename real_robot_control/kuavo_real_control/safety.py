"""Pure-numpy command supervisor shared by dry-run and live ROS execution."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from .contract import ControlConfig


class SafetyError(RuntimeError):
    """Raised when a command or observed robot state violates the contract."""


def _vector(values: np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise SafetyError("{} must contain 14 finite values".format(name))
    return result


def _discrete_stopping_speed(distance: float, acceleration: float, dt_s: float, upper: float) -> float:
    """Largest next-step speed that can stop within distance on this sample grid."""
    if distance <= 0.0:
        return 0.0

    def stopping_distance(speed: float) -> float:
        steps = int(math.ceil(speed / (acceleration * dt_s)))
        return dt_s * (
            steps * speed
            - acceleration * dt_s * steps * (steps - 1) / 2.0
        )

    low, high = 0.0, upper
    for _ in range(32):
        middle = (low + high) / 2.0
        if stopping_distance(middle) <= distance:
            low = middle
        else:
            high = middle
    return low


class SafetySupervisor:
    """Limit target position, velocity, acceleration, and persistent tracking error."""

    def __init__(self, config: ControlConfig):
        self.config = config
        self._command: Optional[np.ndarray] = None
        self._last_raw: Optional[np.ndarray] = None
        self._velocity = np.zeros(14, dtype=np.float64)
        self._tracking_error_time_s = 0.0

    @property
    def command(self) -> np.ndarray:
        if self._command is None:
            raise SafetyError("Safety supervisor has not been synchronized")
        return self._command.copy()

    def synchronize(self, measured_arm_rad: np.ndarray) -> None:
        measured = _vector(measured_arm_rad, "measured_arm_rad")
        self._check_limits(measured, "measured arm state")
        self._command = measured.copy()
        self._last_raw = measured.copy()
        self._velocity.fill(0.0)
        self._tracking_error_time_s = 0.0

    def check_first_target(self, target_arm_rad: np.ndarray) -> None:
        target = _vector(target_arm_rad, "first target")
        self._check_limits(target, "first target")
        if self._command is None:
            raise SafetyError("Synchronize with measured state before checking a target")
        error = np.abs(target - self._command)
        index = int(np.argmax(error))
        if error[index] > self.config.max_start_error_rad:
            raise SafetyError(
                "first target differs from measured {} by {:.3f} rad (limit {:.3f})".format(
                    self.config.arm_joint_names[index],
                    error[index],
                    self.config.max_start_error_rad,
                )
            )

    def project(
        self,
        raw_target_arm_rad: np.ndarray,
        measured_arm_rad: np.ndarray,
        dt_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self._command is None:
            raise SafetyError("Synchronize with measured state before projecting commands")
        if not math.isfinite(dt_s) or not 0.005 <= dt_s <= 0.2:
            raise SafetyError("control dt must be finite and in [0.005, 0.2] seconds")
        raw = _vector(raw_target_arm_rad, "raw target")
        measured = _vector(measured_arm_rad, "measured arm state")
        self._check_limits(raw, "raw target")
        self._check_limits(measured, "measured arm state")

        assert self._last_raw is not None
        source_step = np.abs(raw - self._last_raw)
        source_index = int(np.argmax(source_step))
        if source_step[source_index] > self.config.max_source_step_rad:
            raise SafetyError(
                "source target jumped at {} by {:.3f} rad (limit {:.3f})".format(
                    self.config.arm_joint_names[source_index],
                    source_step[source_index],
                    self.config.max_source_step_rad,
                )
            )

        self._check_tracking(measured, dt_s)

        distance = raw - self._command
        desired_velocity = np.clip(
            distance / dt_s,
            -self.config.max_command_velocity_rad_s,
            self.config.max_command_velocity_rad_s,
        )
        # Start braking before the target.  A plain velocity/acceleration clip
        # can reach a nearby target with residual velocity and then snap that
        # velocity to zero in the overshoot guard, violating its own declared
        # acceleration bound.
        stopping_speed = np.asarray([
            _discrete_stopping_speed(
                abs(float(delta)),
                self.config.max_command_acceleration_rad_s2,
                dt_s,
                self.config.max_command_velocity_rad_s,
            )
            for delta in distance
        ])
        desired_velocity = np.sign(distance) * np.minimum(
            np.abs(desired_velocity), stopping_speed
        )
        acceleration_step = self.config.max_command_acceleration_rad_s2 * dt_s
        velocity = self._velocity + np.clip(
            desired_velocity - self._velocity,
            -acceleration_step,
            acceleration_step,
        )
        candidate = self._command + velocity * dt_s

        # Do not overshoot a nearby target while acceleration-limited velocity
        # still points in the previous direction.
        crossed = (raw - self._command) * (raw - candidate) <= 0.0
        if np.any(crossed):
            target_velocity = (raw[crossed] - self._command[crossed]) / dt_s
            prior_velocity = self._velocity[crossed]
            if np.any(np.abs(target_velocity - prior_velocity) > acceleration_step + 1e-9):
                raise SafetyError("target cannot be reached without violating acceleration limit")
            candidate[crossed] = raw[crossed]
            velocity[crossed] = target_velocity
        self._check_limits(candidate, "projected command")
        self._command = candidate
        self._last_raw = raw.copy()
        self._velocity = velocity
        return candidate.copy(), velocity.copy()

    def accept_validated_target(
        self,
        target_arm_rad: np.ndarray,
        measured_arm_rad: np.ndarray,
        dt_s: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accept a pre-retimed trajectory sample without filtering it again."""
        if self._command is None:
            raise SafetyError("Synchronize with measured state before accepting commands")
        if not math.isfinite(dt_s) or not 0.005 <= dt_s <= 0.2:
            raise SafetyError("control dt must be finite and in [0.005, 0.2] seconds")
        target = _vector(target_arm_rad, "validated target")
        measured = _vector(measured_arm_rad, "measured arm state")
        self._check_limits(target, "validated target")
        self._check_limits(measured, "measured arm state")
        assert self._last_raw is not None
        source_step = np.abs(target - self._last_raw)
        source_index = int(np.argmax(source_step))
        if source_step[source_index] > self.config.max_source_step_rad:
            raise SafetyError(
                "source target jumped at {} by {:.3f} rad (limit {:.3f})".format(
                    self.config.arm_joint_names[source_index],
                    source_step[source_index],
                    self.config.max_source_step_rad,
                )
            )
        self._check_tracking(measured, dt_s)
        velocity = (target - self._command) / dt_s
        velocity_index = int(np.argmax(np.abs(velocity)))
        if abs(velocity[velocity_index]) > self.config.max_command_velocity_rad_s + 1e-8:
            raise SafetyError(
                "validated target velocity at {} is {:.3f} rad/s (limit {:.3f})".format(
                    self.config.arm_joint_names[velocity_index],
                    velocity[velocity_index],
                    self.config.max_command_velocity_rad_s,
                )
            )
        acceleration = (velocity - self._velocity) / dt_s
        acceleration_index = int(np.argmax(np.abs(acceleration)))
        if abs(acceleration[acceleration_index]) > self.config.max_command_acceleration_rad_s2 + 1e-8:
            raise SafetyError(
                "validated target acceleration at {} is {:.3f} rad/s^2 (limit {:.3f})".format(
                    self.config.arm_joint_names[acceleration_index],
                    acceleration[acceleration_index],
                    self.config.max_command_acceleration_rad_s2,
                )
            )
        self._command = target.copy()
        self._last_raw = target.copy()
        self._velocity = velocity
        return target.copy(), velocity.copy()

    def hold(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._command is None:
            raise SafetyError("Safety supervisor has not been synchronized")
        self._velocity.fill(0.0)
        return self._command.copy(), self._velocity.copy()

    def _check_tracking(self, measured: np.ndarray, dt_s: float) -> None:
        assert self._command is not None
        tracking = np.abs(measured - self._command)
        tracking_index = int(np.argmax(tracking))
        if tracking[tracking_index] > self.config.max_tracking_error_rad:
            self._tracking_error_time_s += dt_s
        else:
            self._tracking_error_time_s = 0.0
        if self._tracking_error_time_s >= self.config.tracking_error_timeout_s:
            raise SafetyError(
                "persistent tracking error at {}: {:.3f} rad for {:.3f} s".format(
                    self.config.arm_joint_names[tracking_index],
                    tracking[tracking_index],
                    self._tracking_error_time_s,
                )
            )

    def _check_limits(self, values: np.ndarray, label: str) -> None:
        lower = np.asarray(self.config.safe_lower_rad)
        upper = np.asarray(self.config.safe_upper_rad)
        invalid = np.nonzero((values < lower) | (values > upper))[0]
        if invalid.size:
            index = int(invalid[0])
            raise SafetyError(
                "{} violates safe limit for {}: {:.3f} not in [{:.3f}, {:.3f}]".format(
                    label,
                    self.config.arm_joint_names[index],
                    values[index],
                    lower[index],
                    upper[index],
                )
            )
