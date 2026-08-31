"""Simulator-independent Quest hand/head tracking retargeting for Kuavo teleoperation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


Pose = np.ndarray  # [x, y, z, qw, qx, qy, qz]


def controller_gripper_orientation(grip_quat, aim_quat=None, tool_forward_sign=-1):
    """Tool -Z points along index/aim; tool +X points toward the thumb.

    OpenXR grip -Z runs from little finger toward thumb. Its -Y is the
    fallback forward direction when aim is unavailable. Project the thumb
    axis perpendicular to aim so the gripper closes in the thumb/index plane.
    This is a controller-frame approximation, not tracked finger joints.
    """
    forward = _quat_rotate(aim_quat, [0., 0., -1.]) if aim_quat is not None else _quat_rotate(grip_quat, [0., -1., 0.])
    thumb = _quat_rotate(grip_quat, [0., 0., -1.])
    x = thumb - np.dot(thumb, forward) * forward
    if np.linalg.norm(x) < 1e-5:
        x = _quat_rotate(grip_quat, [1., 0., 0.])
        x -= np.dot(x, forward) * forward
    x /= np.linalg.norm(x)
    z = tool_forward_sign * forward / np.linalg.norm(forward)
    y = np.cross(z, x)
    rotation = np.column_stack((x, y, z))
    # Davenport eigenvector conversion is stable even at 180 degrees.
    r = rotation
    k = np.array([
        [r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
        [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
        [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
        [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)],
    ]) / 3.
    _, vectors = np.linalg.eigh(k)
    quat = vectors[:, -1][[3, 0, 1, 2]]
    return quat if quat[0] >= 0 else -quat


class AbsoluteControllerMapper:
    """Use absolute grip position and a fixed anatomical tool-axis convention."""

    def __init__(self, tool_forward_sign=-1):
        if tool_forward_sign not in (-1, 1):
            raise ValueError("Tool forward sign must be -1 (S200062) or +1 (S63/Robotiq)")
        self._tool_forward_sign = tool_forward_sign
        self.reset()

    def reset(self):
        self._goals_w = {}

    def hold(self, side, tool_pose_w):
        self._goals_w[side] = np.asarray(tool_pose_w, dtype=float).copy()

    def _position_target(self, side, packet, tool, reference):
        return packet[0, :3]

    def target(self, side, controller, tool_pose_w, root_pose_w, *, following, aim_pose=None,
               reference_pose_w=None):
        root = np.asarray(root_pose_w)
        tool = np.asarray(tool_pose_w)
        packet = None if controller is None else np.asarray(controller)
        valid = (packet is not None and packet.shape == (2, 7)
                 and np.all(np.isfinite(packet)) and _valid_pose(packet[0]))
        if not following or side not in self._goals_w:
            self.hold(side, tool)
        if following and valid:
            aim_quat = np.asarray(aim_pose)[3:] if _valid_pose(aim_pose) else None
            reference = root if reference_pose_w is None else np.asarray(reference_pose_w)
            position = self._position_target(side, packet, tool, reference)
            self._goals_w[side] = np.concatenate((position, controller_gripper_orientation(
                packet[0, 3:], aim_quat, self._tool_forward_sign)))
        goal = self._goals_w[side]
        inverse = _quat_conjugate(_normalized_quat(root[3:]))
        return np.concatenate((_quat_rotate(inverse, goal[:3] - root[:3]),
                               _normalized_quat(_quat_multiply(inverse, goal[3:])))).astype(np.float32)


class ScaledControllerMapper(AbsoluteControllerMapper):
    """Clutched, drift-free hand displacement scaled around a torso reference.

    The first valid sample after explicit pause pairs the comfortable human
    hand pose with the current robot tool. Base/torso motion is not amplified.
    Missing tracking preserves both the last world goal and the reference.
    """

    def __init__(self, position_gain=1.5, tool_forward_sign=-1):
        if not np.isfinite(position_gain) or not 1.0 <= position_gain <= 3.0:
            raise ValueError("Scaled position gain must be finite and between 1 and 3")
        self.position_gain = float(position_gain)
        super().__init__(tool_forward_sign)

    def reset(self):
        super().reset()
        self._references = {}

    def target(self, side, controller, tool_pose_w, root_pose_w, *, following, **kwargs):
        if not following:
            self._references.pop(side, None)
        return super().target(side, controller, tool_pose_w, root_pose_w, following=following, **kwargs)

    def _position_target(self, side, packet, tool, reference):
        rotation = _normalized_quat(reference[3:])
        inverse = _quat_conjugate(rotation)
        controller_local = _quat_rotate(inverse, packet[0, :3] - reference[:3])
        if side not in self._references:
            tool_local = _quat_rotate(inverse, tool[:3] - reference[:3])
            self._references[side] = (controller_local.copy(), tool_local)
        controller_zero, tool_zero = self._references[side]
        local = tool_zero + self.position_gain * (controller_local - controller_zero)
        return reference[:3] + _quat_rotate(rotation, local)


@dataclass(frozen=True)
class TeleopMappingCfg:
    """Safety and response settings for the relative bimanual mapping."""

    position_gain: float = 1.5
    rotation_gain: float = 1.0
    position_smoothing: float = 0.45
    rotation_smoothing: float = 0.40
    head_smoothing: float = 0.25
    position_deadband_m: float = 0.0005
    rotation_deadband_rad: float = 0.004
    max_position_step_m: float = 0.025
    max_rotation_step_rad: float = 0.12
    head_yaw_limit_rad: float = 1.40
    head_pitch_limit_rad: float = 0.48


@dataclass(frozen=True)
class TeleopMappingOutput:
    """One control sample and its tracking metadata."""

    action: np.ndarray
    left_valid: bool
    right_valid: bool
    head_valid: bool
    left_pinch_m: float
    right_pinch_m: float

    @property
    def bimanual_valid(self) -> bool:
        return self.left_valid and self.right_valid


def _normalized_quat(quat: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return np.asarray(quat, dtype=np.float64) / norm


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    return np.array([quat[0], -quat[1], -quat[2], -quat[3]], dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quat = _normalized_quat(quat)
    vector_quat = np.array([0.0, *np.asarray(vector, dtype=np.float64)])
    return _quat_multiply(_quat_multiply(quat, vector_quat), _quat_conjugate(quat))[1:]


def _quat_to_rotvec(quat: np.ndarray) -> np.ndarray:
    quat = _normalized_quat(quat)
    if quat[0] < 0.0:
        quat = -quat
    vector_norm = float(np.linalg.norm(quat[1:]))
    if vector_norm < 1.0e-8:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * math.atan2(vector_norm, float(np.clip(quat[0], -1.0, 1.0)))
    return quat[1:] * (angle / vector_norm)


def _quat_to_pitch_yaw(quat: np.ndarray) -> tuple[float, float]:
    w, x, y, z = _normalized_quat(quat)
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(float(np.clip(sin_pitch, -1.0, 1.0)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return pitch, yaw


def _limit_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < 1.0e-9:
        return vector
    return vector * (maximum / norm)


def _valid_pose(pose: Pose | None) -> bool:
    if pose is None:
        return False
    pose = np.asarray(pose)
    return (
        pose.shape == (7,)
        and bool(np.all(np.isfinite(pose)))
        and float(np.linalg.norm(pose[:3])) > 0.05
        and float(np.linalg.norm(pose[3:])) > 0.5
    )


def _pinch_distance(hand: dict[str, Pose] | None) -> float:
    if not hand:
        return float("nan")
    thumb = hand.get("thumb_tip")
    index = hand.get("index_tip")
    if not _valid_pose(thumb) or not _valid_pose(index):
        return float("nan")
    return float(np.linalg.norm(np.asarray(thumb)[:3] - np.asarray(index)[:3]))


class BimanualTeleopMapper:
    """Map OpenXR wrist deltas and HMD orientation to a 14-D Kuavo action.

    The action order is left arm pose delta (6), right arm pose delta (6),
    then absolute head yaw/pitch offsets (2). A newly acquired hand is first
    calibrated and emits zero motion, preventing the common first-frame jump.
    """

    def __init__(self, cfg: TeleopMappingCfg | None = None):
        self.cfg = cfg or TeleopMappingCfg()
        self.reset()

    def reset(self, head_target: np.ndarray | None = None) -> None:
        self.reset_hands()
        self._head_neutral: Pose | None = None
        self._head_offset = np.zeros(2) if head_target is None else np.asarray(head_target, dtype=float).copy()
        if self._head_offset.shape != (2,) or not np.all(np.isfinite(self._head_offset)):
            raise ValueError("Head calibration target must contain finite yaw/pitch offsets.")
        self._filtered_head = self._head_offset.copy()

    def reset_hands(self) -> None:
        """Re-clutch both hands without changing the current head calibration."""
        self._previous_wrist: dict[str, Pose | None] = {"left": None, "right": None}
        self._filtered_position = {"left": np.zeros(3), "right": np.zeros(3)}
        self._filtered_rotation = {"left": np.zeros(3), "right": np.zeros(3)}

    def _hand_delta(
        self,
        side: str,
        hand: dict[str, Pose] | None,
        root_quat_w: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        command = np.zeros(6, dtype=np.float64)
        wrist = hand.get("wrist") if hand else None
        if not _valid_pose(wrist):
            self._previous_wrist[side] = None
            self._filtered_position[side].fill(0.0)
            self._filtered_rotation[side].fill(0.0)
            return command, False

        wrist = np.asarray(wrist, dtype=np.float64)
        previous = self._previous_wrist[side]
        self._previous_wrist[side] = wrist.copy()
        if previous is None:
            return command, True

        root_inverse = _quat_conjugate(_normalized_quat(root_quat_w))
        delta_position_w = wrist[:3] - previous[:3]
        delta_position_b = _quat_rotate(root_inverse, delta_position_w)

        delta_quat_w = _quat_multiply(
            _normalized_quat(wrist[3:]),
            _quat_conjugate(_normalized_quat(previous[3:])),
        )
        delta_rotvec_w = _quat_to_rotvec(delta_quat_w)
        delta_rotvec_b = _quat_rotate(root_inverse, delta_rotvec_w)

        if np.linalg.norm(delta_position_b) < self.cfg.position_deadband_m:
            delta_position_b.fill(0.0)
        if np.linalg.norm(delta_rotvec_b) < self.cfg.rotation_deadband_rad:
            delta_rotvec_b.fill(0.0)

        alpha_pos = self.cfg.position_smoothing
        alpha_rot = self.cfg.rotation_smoothing
        self._filtered_position[side] = (
            alpha_pos * delta_position_b + (1.0 - alpha_pos) * self._filtered_position[side]
        )
        self._filtered_rotation[side] = (
            alpha_rot * delta_rotvec_b + (1.0 - alpha_rot) * self._filtered_rotation[side]
        )
        command[:3] = _limit_norm(
            self._filtered_position[side] * self.cfg.position_gain,
            self.cfg.max_position_step_m,
        )
        command[3:] = _limit_norm(
            self._filtered_rotation[side] * self.cfg.rotation_gain,
            self.cfg.max_rotation_step_rad,
        )
        return command, True

    def _head_target(self, head_pose: Pose | None, root_quat_w: np.ndarray) -> tuple[np.ndarray, bool]:
        if not _valid_pose(head_pose):
            return self._filtered_head.copy(), False
        head_pose = np.asarray(head_pose, dtype=np.float64)
        if self._head_neutral is None:
            self._head_neutral = head_pose.copy()
            return self._filtered_head.copy(), True

        # HMD local axes are +X right, +Y up, -Z forward. Extract
        # heading from the look ray in the neutral *head* frame; world Euler
        # angles couple nod/roll after recenter or base turning.
        relative = _quat_multiply(_quat_conjugate(_normalized_quat(self._head_neutral[3:])),
                                  _normalized_quat(head_pose[3:]))
        forward = _quat_rotate(relative, [0., 0., -1.])
        yaw = math.atan2(-float(forward[0]), -float(forward[2]))
        pitch = math.atan2(-float(forward[1]), float(np.hypot(forward[0], forward[2])))
        target = np.array(
            [
                np.clip(yaw + self._head_offset[0], -self.cfg.head_yaw_limit_rad, self.cfg.head_yaw_limit_rad),
                np.clip(pitch + self._head_offset[1], -self.cfg.head_pitch_limit_rad, self.cfg.head_pitch_limit_rad),
            ],
            dtype=np.float64,
        )
        alpha = self.cfg.head_smoothing
        self._filtered_head = alpha * target + (1.0 - alpha) * self._filtered_head
        return self._filtered_head.copy(), True

    def advance(
        self,
        left_hand: dict[str, Pose] | None,
        right_hand: dict[str, Pose] | None,
        head_pose: Pose | None,
        root_quat_w: np.ndarray,
    ) -> TeleopMappingOutput:
        left_action, left_valid = self._hand_delta("left", left_hand, root_quat_w)
        right_action, right_valid = self._hand_delta("right", right_hand, root_quat_w)
        head_action, head_valid = self._head_target(head_pose, root_quat_w)
        action = np.concatenate([left_action, right_action, head_action]).astype(np.float32)
        return TeleopMappingOutput(
            action=action,
            left_valid=left_valid,
            right_valid=right_valid,
            head_valid=head_valid,
            left_pinch_m=_pinch_distance(left_hand),
            right_pinch_m=_pinch_distance(right_hand),
        )

    def advance_controllers(self, left_controller, right_controller, head_pose, root_quat_w):
        """Use controller grip deltas with the same re-clutch and safety limits.

        The internal wrist key only selects the pose mapper. No hand joints or
        pinch distances are synthesized for recording.
        """
        def pose_target(packet):
            if packet is None:
                return None
            packet = np.asarray(packet)
            if packet.shape != (2, 7) or not np.all(np.isfinite(packet)):
                return None
            return {"wrist": packet[0]}

        return self.advance(pose_target(left_controller), pose_target(right_controller), head_pose, root_quat_w)
