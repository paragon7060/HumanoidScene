"""Joystick mapping and planar lift kinematics, independent of Isaac Sim."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


BODY_JOINTS = ["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"]
BODY_ACTION_NAMES = ("base_forward_m_s", "base_left_m_s", "base_yaw_rad_s", *BODY_JOINTS[:3])


def controller_axis(packet, index, deadzone=0.15):
    if packet is None:
        return 0.0
    packet = np.asarray(packet)
    if packet.shape != (2, 7) or not np.all(np.isfinite(packet)):
        return 0.0
    value = float(np.clip(packet[1, index], -1, 1))
    return np.sign(value) * max(0., abs(value) - deadzone) / (1 - deadzone)


class TeleopBodyMapper:
    """Move the base in its XY plane and lift the torso without pitching it."""

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

    def reset(self):
        self.height = 0.0
        self.joints = np.zeros(4)

    def _planar_position(self, q):
        angles = (q[0], q[0] + q[1])
        return sum(np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) @ link
                   for a, link in zip(angles, self.links))

    def advance(self, left, right, dt, *, enabled):
        velocity = np.zeros(2)
        yaw_rate = 0.0
        if enabled:
            # Native OpenXR axes: +Y is up, unlike the WebXR Gamepad API.
            velocity = .25 * np.array([controller_axis(left, 1), -controller_axis(left, 0)])
            velocity /= max(1.0, np.linalg.norm(velocity) / .25)
            yaw_rate = -1.2 * controller_axis(right, 0)
            if self.has_wheel_base:
                requested_height = float(np.clip(
                    self.height + .12 * controller_axis(right, 1) * dt, 0.0, .40
                ))
                target = self.links.sum(axis=0) + [0., requested_height]
                q = self.joints[:2].copy()
                for _ in range(12):
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
                if (self.limits[2, 0] <= pitch <= self.limits[2, 1]
                        and np.linalg.norm(self._planar_position(q) - target) < .002):
                    self.height = requested_height
                    self.joints[:3] = [*q, pitch]
            self.joints[3] = 0.0
        return np.concatenate((velocity, [yaw_rate], self.joints[:3])).astype(np.float32)
