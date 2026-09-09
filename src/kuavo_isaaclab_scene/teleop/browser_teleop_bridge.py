"""WebSocket bridge between a WebXR browser and Kuavo teleoperation.

The bridge deliberately has no Isaac Lab imports.  A simulation loop polls the
latest browser tracking sample and publishes an already composed RGB camera
frame.  The browser sends WebXR coordinates (right, up, backward); poses are
converted to Kuavo base coordinates (forward, left, up) here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import math
import re
import struct
import threading
import time
from typing import Any

import numpy as np
import websockets


Pose = np.ndarray  # [x, y, z, qw, qx, qy, qz]
PROTOCOL_VERSION = 2
FRAME_MAGIC = b"KVR2"
FRAME_HEADER = struct.Struct("<4sBBHIIdff")

_WEBXR_TO_KUAVO = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class BrowserEyeView:
    eye: str
    pose: Pose
    projection_matrix: np.ndarray


@dataclass(frozen=True)
class BrowserControllerState:
    grip: Pose
    thumbstick: tuple[float, float]  # WebXR: right/down positive
    trigger: float

    def native_packet(self) -> np.ndarray:
        """Adapt WebXR axes to the shared OpenXR body mapper (+Y up)."""
        packet = np.zeros((2, 7), dtype=np.float32)
        packet[0] = self.grip
        packet[1, :3] = (self.thumbstick[0], -self.thumbstick[1], self.trigger)
        return packet


@dataclass(frozen=True)
class BrowserTrackingSample:
    sequence: int
    client_timestamp_ms: float
    head: Pose | None
    left_hand: dict[str, Pose] | None
    right_hand: dict[str, Pose] | None
    views: tuple[BrowserEyeView, ...]
    received_at: float
    # Optional protocol-v2 extension: older clients remain safe with no motion.
    left_controller: BrowserControllerState | None = None
    right_controller: BrowserControllerState | None = None


@dataclass(frozen=True)
class BrowserClientMetrics:
    received_fps: float = 0.0
    rendered_fps: float = 0.0
    pose_to_frame_ms: float = float("nan")
    decode_ms: float = float("nan")
    dropped_frames: int = 0
    received_at: float = 0.0


@dataclass(frozen=True)
class PoseEditorCommand:
    """One validated command from the standalone Task1 pose-editor page."""

    sequence: int
    action: str
    joint_name: str | None = None
    value_rad: float | None = None
    joint_positions: dict[str, float] | None = None
    view: str | None = None
    visible: bool | None = None
    grasp_z_offset_m: float | None = None
    control_name: str | None = None
    control_value: float | None = None
    collision_visible: bool | None = None
    torso_height_m: float | None = None


_ARM_JOINT_PATTERN = re.compile(r"^zarm_[lr][1-7]_joint$")
_EDITOR_POSITION_JOINTS = {
    "knee_joint",
    "leg_joint",
    "waist_pitch_joint",
    "waist_yaw_joint",
    "zhead_1_joint",
    "zhead_2_joint",
    *(f"zarm_{side}{index}_joint" for side in "lr" for index in range(1, 8)),
}
_EDITOR_LOGICAL_CONTROLS = {"left_gripper", "right_gripper"}
_EDITOR_VIEWS = {
    "rear_left",
    "rear_right",
    "front_left",
    "front_right",
    "left",
    "right",
    "head",
    "left_wrist",
    "right_wrist",
}


def parse_pose_editor_message(message: str) -> PoseEditorCommand | None:
    """Parse the small, allow-listed pose editor control protocol."""
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "pose_editor"
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        return None
    try:
        sequence = int(payload.get("sequence"))
    except (TypeError, ValueError):
        return None
    if sequence < 0:
        return None
    action = payload.get("action")
    if action == "set_joint":
        joint_name = payload.get("joint_name")
        try:
            value_rad = float(payload.get("value_rad"))
        except (TypeError, ValueError):
            return None
        if not isinstance(joint_name, str) or not _ARM_JOINT_PATTERN.fullmatch(joint_name):
            return None
        if not math.isfinite(value_rad):
            return None
        return PoseEditorCommand(sequence, action, joint_name=joint_name, value_rad=value_rad)
    if action == "set_pose":
        values = payload.get("joint_positions")
        if not isinstance(values, dict) or not values or len(values) > 14:
            return None
        joint_positions = {}
        for joint_name, value in values.items():
            if not isinstance(joint_name, str) or not _ARM_JOINT_PATTERN.fullmatch(joint_name):
                return None
            try:
                value_rad = float(value)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(value_rad):
                return None
            joint_positions[joint_name] = value_rad
        return PoseEditorCommand(sequence, action, joint_positions=joint_positions)
    if action == "set_view":
        view = payload.get("view")
        if view not in _EDITOR_VIEWS:
            return None
        return PoseEditorCommand(sequence, action, view=view)
    if action == "set_grasp_visibility":
        visible = payload.get("visible")
        if not isinstance(visible, bool):
            return None
        return PoseEditorCommand(sequence, action, visible=visible)
    if action == "set_grasp_z_offset":
        try:
            offset_m = float(payload.get("offset_m"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(offset_m) or not -0.15 <= offset_m <= 0.15:
            return None
        return PoseEditorCommand(sequence, action, grasp_z_offset_m=offset_m)
    if action == "set_control":
        control_name = payload.get("control_name")
        try:
            value = float(payload.get("value"))
        except (TypeError, ValueError):
            return None
        if not isinstance(control_name, str) or not math.isfinite(value):
            return None
        if control_name in _EDITOR_LOGICAL_CONTROLS:
            if not 0.0 <= value <= 1.0:
                return None
        elif control_name not in _EDITOR_POSITION_JOINTS:
            return None
        return PoseEditorCommand(
            sequence, action, control_name=control_name, control_value=value
        )
    if action == "set_gripper_collision_visibility":
        visible = payload.get("visible")
        if not isinstance(visible, bool):
            return None
        return PoseEditorCommand(sequence, action, collision_visible=visible)
    if action == "set_torso_height":
        try:
            height_m = float(payload.get("height_m"))
        except (TypeError, ValueError):
            return None
        if not math.isfinite(height_m) or not 0.0 <= height_m <= 0.40:
            return None
        return PoseEditorCommand(sequence, action, torso_height_m=height_m)
    if action in {"reset", "print_pose"}:
        return PoseEditorCommand(sequence, action)
    return None


def _normalized_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_to_matrix(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalized_quat(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale, (matrix[0, 2] - matrix[2, 0]) / scale,
             (matrix[1, 0] - matrix[0, 1]) / scale]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [(matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale,
                 (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale]
            )
        elif axis == 1:
            scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [(matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale,
                 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale]
            )
        else:
            scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [(matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale,
                 (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale]
            )
    return _normalized_quat(quat)


def webxr_pose_to_kuavo(pose: Any) -> Pose | None:
    """Convert one WebXR pose into Kuavo base coordinates."""
    try:
        array = np.asarray(pose, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if array.shape != (7,) or not np.all(np.isfinite(array)):
        return None
    position = _WEBXR_TO_KUAVO @ array[:3]
    rotation = _quat_to_matrix(array[3:])
    converted_rotation = _WEBXR_TO_KUAVO @ rotation @ _WEBXR_TO_KUAVO.T
    return np.concatenate([position, _matrix_to_quat(converted_rotation)]).astype(np.float32)


def parse_tracking_message(message: str, *, received_at: float | None = None) -> BrowserTrackingSample | None:
    """Validate a browser JSON packet and convert all available poses."""
    try:
        payload = json.loads(message)
    except (TypeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(payload, dict)
        or payload.get("type") != "tracking"
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        return None

    def parse_hand(value: Any) -> dict[str, Pose] | None:
        if not isinstance(value, dict):
            return None
        parsed = {}
        for name in ("wrist", "thumb_tip", "index_tip"):
            pose = webxr_pose_to_kuavo(value.get(name))
            if pose is not None:
                parsed[name] = pose
        return parsed if "wrist" in parsed else None

    def parse_controller(value: Any) -> BrowserControllerState | None:
        if not isinstance(value, dict):
            return None
        try:
            grip = webxr_pose_to_kuavo(value.get("grip"))
            stick = np.asarray(value.get("thumbstick"), dtype=np.float32)
            trigger = float(value.get("trigger", 0.0))
        except (TypeError, ValueError, OverflowError):
            return None
        if grip is None or stick.shape != (2,) or not np.all(np.isfinite(stick)) or not np.isfinite(trigger):
            return None
        stick = np.clip(stick, -1.0, 1.0)
        return BrowserControllerState(grip, (float(stick[0]), float(stick[1])), float(np.clip(trigger, 0.0, 1.0)))

    views = []
    for value in payload.get("views", []):
        if not isinstance(value, dict) or value.get("eye") not in {"left", "right"}:
            continue
        pose = webxr_pose_to_kuavo(value.get("pose"))
        try:
            projection = np.asarray(value.get("projection_matrix"), dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if pose is None or projection.shape != (16,) or not np.all(np.isfinite(projection)):
            continue
        views.append(BrowserEyeView(value["eye"], pose, projection))

    try:
        sequence = int(payload.get("sequence"))
        client_timestamp_ms = float(payload.get("timestamp_ms"))
    except (TypeError, ValueError):
        return None
    if sequence < 0 or not np.isfinite(client_timestamp_ms):
        return None

    return BrowserTrackingSample(
        sequence=sequence,
        client_timestamp_ms=client_timestamp_ms,
        head=webxr_pose_to_kuavo(payload.get("head")),
        left_hand=parse_hand(payload.get("left_hand")),
        right_hand=parse_hand(payload.get("right_hand")),
        views=tuple(views),
        received_at=time.monotonic() if received_at is None else received_at,
        left_controller=parse_controller(payload.get("left_controller")),
        right_controller=parse_controller(payload.get("right_controller")),
    )


def pack_frame_packet(
    jpeg: bytes,
    *,
    frame_sequence: int,
    tracking_sequence: int,
    client_timestamp_ms: float,
    encode_ms: float,
    server_fps: float,
) -> bytes:
    """Prefix JPEG bytes with the versioned browser frame header."""
    if not jpeg:
        raise ValueError("JPEG payload must not be empty.")
    header = FRAME_HEADER.pack(
        FRAME_MAGIC,
        PROTOCOL_VERSION,
        0,
        FRAME_HEADER.size,
        int(frame_sequence),
        int(tracking_sequence),
        float(client_timestamp_ms),
        float(encode_ms),
        float(server_fps),
    )
    return header + bytes(jpeg)


def unpack_frame_packet(packet: bytes) -> tuple[dict[str, float | int], bytes]:
    """Decode a frame packet for tests and offline diagnostics."""
    if len(packet) < FRAME_HEADER.size:
        raise ValueError("Frame packet is shorter than its header.")
    magic, version, flags, header_size, frame_seq, tracking_seq, timestamp, encode_ms, server_fps = (
        FRAME_HEADER.unpack_from(packet)
    )
    if magic != FRAME_MAGIC or version != PROTOCOL_VERSION or header_size != FRAME_HEADER.size:
        raise ValueError("Unsupported browser frame protocol.")
    return (
        {
            "version": version,
            "flags": flags,
            "frame_sequence": frame_seq,
            "tracking_sequence": tracking_seq,
            "client_timestamp_ms": timestamp,
            "encode_ms": encode_ms,
            "server_fps": server_fps,
        },
        packet[header_size:],
    )


class BrowserTeleopBridge:
    """Background WebSocket server with latest-sample semantics."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8765, stale_after_s: float = 0.35):
        self.host = host
        self.port = int(port)
        self.stale_after_s = float(stale_after_s)
        self._lock = threading.Lock()
        self._sample: BrowserTrackingSample | None = None
        self._frame_packet: bytes | None = None
        self._frame_sequence = 0
        self._client_metrics = BrowserClientMetrics()
        self._pose_editor_command: PoseEditorCommand | None = None
        self._pose_editor_command_sequence = 0
        self._pose_editor_state: str | None = None
        self._pose_editor_state_sequence = 0
        self._clients = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None
        self._started = threading.Event()
        self._startup_error: BaseException | None = None
        self._thread = threading.Thread(target=self._thread_main, name="kuavo-browser-bridge", daemon=True)

    @property
    def client_count(self) -> int:
        with self._lock:
            return self._clients

    def start(self, timeout_s: float = 5.0) -> None:
        self._thread.start()
        if not self._started.wait(timeout_s):
            raise TimeoutError(f"Timed out starting browser bridge on {self.host}:{self.port}")
        if self._startup_error is not None:
            raise RuntimeError(f"Failed to start browser bridge on {self.host}:{self.port}") from self._startup_error

    def close(self) -> None:
        loop = self._loop
        stop_event = self._stop_event
        if loop is not None and stop_event is not None:
            loop.call_soon_threadsafe(stop_event.set)
        if self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def latest(self) -> BrowserTrackingSample:
        now = time.monotonic()
        with self._lock:
            sample = self._sample
        if sample is None or now - sample.received_at > self.stale_after_s:
            return BrowserTrackingSample(0, 0.0, None, None, None, (), now)
        return sample

    @property
    def client_metrics(self) -> BrowserClientMetrics:
        with self._lock:
            return self._client_metrics

    def publish_frame(
        self,
        jpeg: bytes,
        *,
        tracking_sequence: int,
        client_timestamp_ms: float,
        encode_ms: float,
        server_fps: float,
    ) -> None:
        with self._lock:
            self._frame_sequence += 1
            self._frame_packet = pack_frame_packet(
                jpeg,
                frame_sequence=self._frame_sequence,
                tracking_sequence=tracking_sequence,
                client_timestamp_ms=client_timestamp_ms,
                encode_ms=encode_ms,
                server_fps=server_fps,
            )

    def latest_pose_editor_command(self, after_sequence: int = -1) -> PoseEditorCommand | None:
        """Return the newest editor command once, using its client sequence."""
        with self._lock:
            command = self._pose_editor_command
        if command is None or command.sequence <= after_sequence:
            return None
        return command

    def publish_pose_editor_state(self, state: dict[str, Any]) -> None:
        """Publish a JSON state snapshot to every connected editor client."""
        payload = dict(state)
        payload.update(type="pose_editor_state", protocol_version=PROTOCOL_VERSION)
        encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self._pose_editor_state = encoded
            self._pose_editor_state_sequence += 1

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as error:
            self._startup_error = error
            self._started.set()

    async def _serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            async with websockets.serve(self._handle_client, self.host, self.port, max_size=1 << 20):
                self._started.set()
                await self._stop_event.wait()
        except BaseException as error:
            self._startup_error = error
            self._started.set()
            raise

    async def _handle_client(self, websocket) -> None:
        with self._lock:
            self._clients += 1
        try:
            hello_raw = await asyncio.wait_for(websocket.recv(), timeout=3.0)
            try:
                hello = json.loads(hello_raw) if isinstance(hello_raw, str) else None
            except json.JSONDecodeError:
                hello = None
            if not isinstance(hello, dict) or hello.get("type") != "client_hello":
                await websocket.close(code=1002, reason="Kuavo protocol hello required")
                return
            if hello.get("protocol_version") != PROTOCOL_VERSION:
                await websocket.send(
                    json.dumps(
                        {
                            "type": "protocol_error",
                            "expected": PROTOCOL_VERSION,
                            "received": hello.get("protocol_version"),
                        }
                    )
                )
                await websocket.close(code=1002, reason="Kuavo protocol version mismatch")
                return
            await websocket.send(json.dumps({"type": "server_hello", "protocol_version": PROTOCOL_VERSION}))
            receiver = asyncio.create_task(self._receive_messages(websocket))
            sender = asyncio.create_task(self._send_camera(websocket))
            done, pending = await asyncio.wait((receiver, sender), return_when=asyncio.FIRST_COMPLETED)
            for task in pending:
                task.cancel()
            await asyncio.gather(*done, *pending, return_exceptions=True)
        finally:
            with self._lock:
                self._clients = max(0, self._clients - 1)

    async def _receive_messages(self, websocket) -> None:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            sample = parse_tracking_message(message)
            if sample is not None:
                with self._lock:
                    self._sample = sample
                continue
            editor_command = parse_pose_editor_message(message)
            if editor_command is not None:
                with self._lock:
                    # Use a server-side monotonic sequence so a browser reload
                    # may safely restart its own packet counter from zero.
                    self._pose_editor_command_sequence += 1
                    self._pose_editor_command = PoseEditorCommand(
                        self._pose_editor_command_sequence,
                        editor_command.action,
                        joint_name=editor_command.joint_name,
                        value_rad=editor_command.value_rad,
                        joint_positions=editor_command.joint_positions,
                        view=editor_command.view,
                        visible=editor_command.visible,
                        grasp_z_offset_m=editor_command.grasp_z_offset_m,
                        control_name=editor_command.control_name,
                        control_value=editor_command.control_value,
                        collision_visible=editor_command.collision_visible,
                    )
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("type") == "client_metrics"
                and payload.get("protocol_version") == PROTOCOL_VERSION
            ):
                try:
                    metrics = BrowserClientMetrics(
                        received_fps=float(payload.get("received_fps", 0.0)),
                        rendered_fps=float(payload.get("rendered_fps", 0.0)),
                        pose_to_frame_ms=float(payload.get("pose_to_frame_ms", float("nan"))),
                        decode_ms=float(payload.get("decode_ms", float("nan"))),
                        dropped_frames=int(payload.get("dropped_frames", 0)),
                        received_at=time.monotonic(),
                    )
                except (TypeError, ValueError):
                    continue
                with self._lock:
                    self._client_metrics = metrics

    async def _send_camera(self, websocket) -> None:
        sent_sequence = -1
        sent_editor_sequence = -1
        while True:
            with self._lock:
                packet = self._frame_packet
                sequence = self._frame_sequence
                editor_state = self._pose_editor_state
                editor_sequence = self._pose_editor_state_sequence
            if packet is not None and sequence != sent_sequence:
                await websocket.send(packet)
                sent_sequence = sequence
            if editor_state is not None and editor_sequence != sent_editor_sequence:
                await websocket.send(editor_state)
                sent_editor_sequence = editor_sequence
            await asyncio.sleep(1.0 / 30.0)
