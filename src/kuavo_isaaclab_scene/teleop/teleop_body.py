"""Joystick mapping and planar lift kinematics, independent of Isaac Sim."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BODY_JOINTS = ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"]
BODY_ACTION_NAMES = ("base_forward_m_s", "base_left_m_s", "base_yaw_rad_s", *BODY_JOINTS)
# Shared by controller mapping modes, collection and browser preview.
BASE_LINEAR_SPEED_M_S = 0.15
BASE_LINEAR_ACCEL_M_S2 = 0.25
BASE_YAW_SPEED_RAD_S = 0.45
BASE_YAW_ACCEL_RAD_S2 = 0.75
TORSO_HEIGHT_SPEED_M_S = 0.10
TORSO_HEIGHT_ACCEL_M_S2 = 0.20
TORSO_FORWARD_SPEED_M_S = 0.10
TORSO_FORWARD_ACCEL_M_S2 = 0.20
TORSO_FORWARD_LIMIT_M = 0.15
WAIST_YAW_SPEED_RAD_S = 0.30
WAIST_YAW_ACCEL_RAD_S2 = 0.60
WAIST_YAW_LIMIT_RAD = 1.20


def valid_controller(packet):
    return (packet is not None and np.shape(packet) == (2, 7)
            and np.all(np.isfinite(packet)))


def controller_axis(packet, index, deadzone=0.15):
    if packet is None:
        return 0.0
    packet = np.asarray(packet)
    if packet.shape != (2, 7) or not np.all(np.isfinite(packet)):
        return 0.0
    value = float(np.clip(packet[1, index], -1, 1))
    return np.sign(value) * max(0., abs(value) - deadzone) / (1 - deadzone)


class TeleopBodyMapper:
    """Shared smooth base, upright torso XY/Z adjustment and waist yaw."""

    def __init__(self, urdf: str | Path, *, has_wheel_base: bool = True):
        self.has_wheel_base = has_wheel_base
        if has_wheel_base:
            joints = {j.attrib["name"]: j for j in ET.parse(urdf).findall("joint")}
            self.links = np.array([
                [float(x) for x in joints[name].find("origin").attrib["xyz"].split()][::2]
                for name in ("leg_joint", "waist_pitch_joint")
            ])
            self.limits = np.array([
                [float(joints[name].find("limit").attrib[key]) for key in ("lower", "upper")]
                for name in BODY_JOINTS
            ])
        else:
            self.links = np.empty((0, 2))
            self.limits = np.empty((0, 2))
        self.reset()

    def reset(self, joint_positions=None):
        """Reset smoothing and optionally synchronize to the simulated torso."""
        self.joints = np.zeros(4)
        if joint_positions is not None:
            values = np.asarray(joint_positions, dtype=float)
            if values.shape not in ((3,), (4,)) or not np.all(np.isfinite(values)):
                raise ValueError("Torso reset needs three or four finite joint positions.")
            self.joints[:len(values)] = values
        nominal_z = self.links.sum(axis=0)[1] if self.has_wheel_base else 0.0
        self.height = (
            float(np.clip(self._planar_position(self.joints[:2])[1] - nominal_z, 0.0, .40))
            if self.has_wheel_base else 0.0
        )
        self._forward_origin = (
            float(self._planar_position(self.joints[:2])[0] - self.links.sum(axis=0)[0])
            if self.has_wheel_base else 0.0
        )
        self.forward = 0.0
        self._linear_velocity = np.zeros(2)
        self._yaw_rate = 0.0
        self._height_rate = 0.0
        self._forward_rate = 0.0
        self._waist_yaw_rate = 0.0
        self._body_modifier = False
        self._grip_used_for_body = False

    def gesture_squeeze(self, right):
        """Suppress long-press shortcuts after grip+stick until grip release."""
        if not valid_controller(right):
            return None
        squeeze = float(np.asarray(right)[1, 3])
        if squeeze <= .2:
            self._grip_used_for_body = False
        elif squeeze >= .5 and (controller_axis(right, 0) != 0 or controller_axis(right, 1) != 0):
            self._grip_used_for_body = True
        return None if self._grip_used_for_body else squeeze

    def _planar_position(self, q):
        angles = (q[0], q[0] + q[1])
        return sum(np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ link
                   for a, link in zip(angles, self.links))

    def advance(self, left, right, dt, *, enabled):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Body mapper dt must be finite and positive")
        # A missing/invalid controller is a safety stop, not a neutral stick
        # that should gradually coast to zero.
        enabled = enabled and valid_controller(left) and valid_controller(right)
        if enabled:
            modifier = float(np.asarray(right)[1, 3]) >= .5
            if modifier != self._body_modifier:
                # Switching the meaning of the stick must not carry the old
                # base/height command into the newly selected torso controls.
                self._yaw_rate = self._height_rate = 0.0
                self._forward_rate = self._waist_yaw_rate = 0.0
            self._body_modifier = modifier
            # Native OpenXR axes: +Y is up, unlike the WebXR Gamepad API.
            target_velocity = BASE_LINEAR_SPEED_M_S * np.array(
                [controller_axis(left, 1), -controller_axis(left, 0)]
            )
            target_velocity /= max(1.0, np.linalg.norm(target_velocity) / BASE_LINEAR_SPEED_M_S)
            # Bound vector acceleration, including diagonal motion, stick
            # release and reversal. Do not apply independent per-axis ramps.
            delta = target_velocity - self._linear_velocity
            max_change = BASE_LINEAR_ACCEL_M_S2 * dt
            self._linear_velocity += delta * min(1.0, max_change / max(np.linalg.norm(delta), 1e-12))
            target_yaw_rate = 0.0 if modifier else -BASE_YAW_SPEED_RAD_S * controller_axis(right, 0)
            # Ramp toward the commanded yaw rate instead of snapping instantly;
            # a sudden yaw jump is what flings a grasped box during a fast turn.
            step = BASE_YAW_ACCEL_RAD_S2 * dt
            self._yaw_rate += np.clip(target_yaw_rate - self._yaw_rate, -step, step)
            if self.has_wheel_base:
                target_height_rate = 0.0 if modifier else TORSO_HEIGHT_SPEED_M_S * controller_axis(right, 1)
                height_step = TORSO_HEIGHT_ACCEL_M_S2 * dt
                self._height_rate += np.clip(target_height_rate - self._height_rate, -height_step, height_step)
                # Do not accumulate a velocity into a hard height boundary.
                height_before_clip = self.height + self._height_rate * dt
                requested_height = float(np.clip(
                    height_before_clip, 0.0, .40
                ))
                if requested_height != height_before_clip:
                    self._height_rate = 0.0
                target_forward_rate = TORSO_FORWARD_SPEED_M_S * controller_axis(right, 1) if modifier else 0.0
                forward_step = TORSO_FORWARD_ACCEL_M_S2 * dt
                self._forward_rate += np.clip(target_forward_rate - self._forward_rate, -forward_step, forward_step)
                forward_before_clip = self.forward + self._forward_rate * dt
                requested_forward = float(np.clip(forward_before_clip, -TORSO_FORWARD_LIMIT_M, TORSO_FORWARD_LIMIT_M))
                if requested_forward != forward_before_clip:
                    self._forward_rate = 0.0
                torso_changed = requested_height != self.height or requested_forward != self.forward
                target = self.links.sum(axis=0) + [self._forward_origin + requested_forward, requested_height]
                q = self.joints[:2].copy()
                for _ in range(12 if torso_changed else 0):
                    error = target - self._planar_position(q)
                    if np.linalg.norm(error) < 1e-5:
                        break
                    jac = np.column_stack([
                        (self._planar_position(q + np.eye(2)[i] * 1e-5)
                         - self._planar_position(q)) / 1e-5
                        for i in range(2)
                    ])
                    q += np.clip(
                        np.linalg.solve(
                            jac.T @ jac + np.eye(2) * 1e-5,
                            jac.T @ error,
                        ),
                        -.05,
                        .05,
                    )
                    q = np.clip(q, self.limits[:2, 0], self.limits[:2, 1])
                pitch = -q.sum()
                if not torso_changed:
                    pass  # Preserve captured torso joints when the stick is neutral.
                elif (self.limits[2, 0] <= pitch <= self.limits[2, 1]
                        and np.linalg.norm(self._planar_position(q) - target) < .002):
                    self.height = requested_height
                    self.forward = requested_forward
                    self.joints[:3] = [*q, pitch]
                else:
                    self._height_rate = 0.0
                    self._forward_rate = 0.0
                target_waist_rate = -WAIST_YAW_SPEED_RAD_S * controller_axis(right, 0) if modifier else 0.0
                waist_step = WAIST_YAW_ACCEL_RAD_S2 * dt
                self._waist_yaw_rate += np.clip(target_waist_rate - self._waist_yaw_rate, -waist_step, waist_step)
                if self._waist_yaw_rate != 0.0:
                    waist_before_clip = self.joints[3] + self._waist_yaw_rate * dt
                    self.joints[3] = np.clip(waist_before_clip,
                                            max(self.limits[3, 0], -WAIST_YAW_LIMIT_RAD),
                                            min(self.limits[3, 1], WAIST_YAW_LIMIT_RAD))
                    if self.joints[3] != waist_before_clip:
                        self._waist_yaw_rate = 0.0
        else:
            # Safety stop stays instantaneous; smoothing must never delay it.
            self._linear_velocity[:] = 0.0
            self._yaw_rate = 0.0
            self._height_rate = 0.0
            self._forward_rate = self._waist_yaw_rate = 0.0
        return np.concatenate((self._linear_velocity, [self._yaw_rate], self.joints)).astype(np.float32)
