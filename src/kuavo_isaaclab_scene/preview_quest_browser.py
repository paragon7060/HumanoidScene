#!/usr/bin/env python3
"""Interact with the Kuavo teleop environment from a desktop WebXR browser."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher
from .gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
    gripper_teleop_action,
    resolve_gripper_settings,
)
from .robot_model import add_robot_model_cli_args, export_robot_model_cli


parser = argparse.ArgumentParser(description="Preview Kuavo Quest interaction through a local browser/IWER.")
parser.add_argument("--bridge-host", default="127.0.0.1", help="WebSocket bind address; use 0.0.0.0 for Quest/LAN.")
parser.add_argument("--bridge-port", type=int, default=8765)
parser.add_argument("--stream-fps", type=float, default=30.0, help="Browser stereo JPEG rate.")
parser.add_argument("--jpeg-quality", type=int, default=80)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--stereo-eye-width", type=int, default=512)
parser.add_argument("--stereo-eye-height", type=int, default=512)
parser.add_argument(
    "--stereo-eye-separation",
    type=float,
    default=0.064,
    metavar="METERS",
    help="Fallback baseline used only until Quest supplies per-eye poses.",
)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--head-camera-height", type=int, default=360)
parser.add_argument("--wrist-camera-width", type=int, default=240)
parser.add_argument("--wrist-camera-height", type=int, default=180)
parser.add_argument("--position-gain", type=float, default=1.5)
parser.add_argument("--rotation-gain", type=float, default=1.0)
parser.add_argument("--camera-preview", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--domain-randomization", action=argparse.BooleanOptionalAction, default=False)
parser.add_argument("--rack-boxes", type=str, default=None, metavar="SPEC")
parser.add_argument("--rack-box-layout", type=Path, default=None, metavar="JSON")
parser.add_argument("--rack-box-poses", type=Path, default=None, metavar="JSON")
parser.add_argument("--ignore-captured-box-poses", action="store_true")
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)
try:
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))

if args_cli.headless:
    parser.error("Browser interaction preview requires the Isaac Sim GUI; do not pass --headless.")
if not 1 <= args_cli.bridge_port <= 65535:
    parser.error("--bridge-port must be between 1 and 65535.")
if args_cli.stream_fps <= 0.0:
    parser.error("--stream-fps must be positive.")
if not 1 <= args_cli.jpeg_quality <= 100:
    parser.error("--jpeg-quality must be between 1 and 100.")
if min(
    args_cli.head_camera_width,
    args_cli.head_camera_height,
    args_cli.wrist_camera_width,
    args_cli.wrist_camera_height,
    args_cli.stereo_eye_width,
    args_cli.stereo_eye_height,
) <= 0:
    parser.error("Camera width/height values must be positive.")
if not 0.05 <= args_cli.stereo_eye_separation <= 0.075:
    parser.error("--stereo-eye-separation must be between 0.05 and 0.075 meters.")
if args_cli.rack_boxes is not None and args_cli.rack_box_layout is not None:
    parser.error("Use only one of --rack-boxes and --rack-box-layout.")
if args_cli.rack_boxes is not None:
    os.environ["KUAVO_RACK_BOXES"] = args_cli.rack_boxes
    os.environ.pop("KUAVO_RACK_BOX_LAYOUT", None)
elif args_cli.rack_box_layout is not None:
    os.environ["KUAVO_RACK_BOX_LAYOUT"] = str(args_cli.rack_box_layout.expanduser().resolve())
    os.environ.pop("KUAVO_RACK_BOXES", None)
if args_cli.ignore_captured_box_poses:
    os.environ["KUAVO_IGNORE_RACK_BOX_POSES"] = "1"
    os.environ.pop("KUAVO_RACK_BOX_POSES", None)
elif args_cli.rack_box_poses is not None:
    os.environ["KUAVO_RACK_BOX_POSES"] = str(args_cli.rack_box_poses.expanduser().resolve())

# The browser supplies XR tracking.  Do not start Kit OpenXR in this process.
args_cli.xr = False
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from .browser_teleop_bridge import BrowserTeleopBridge, BrowserTrackingSample
from .camera_viewports import open_camera_viewports
from .stereo_camera_calibration import calibrations_from_tracking, camera_world_pose
from .stereo_compositor import compose_stereo_atlas
from .teleop_safety import GripperCommandLatch, TrackingLossGuard
from .teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from .teleop_mapping import BimanualTeleopMapper, TeleopMappingCfg


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _camera_rgb(camera) -> np.ndarray:
    rgb = _to_numpy(camera.data.output["rgb"][0])
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.01 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    return rgb.astype(np.uint8, copy=False)


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
    quat = np.asarray(quat, dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1.0e-8)
    conjugate = np.array([quat[0], -quat[1], -quat[2], -quat[3]])
    pure = np.array([0.0, *np.asarray(vector, dtype=np.float64)])
    return _quat_multiply(_quat_multiply(quat, pure), conjugate)[1:]


def _pose_to_world(pose: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    position = root_pos + _quat_rotate(root_quat, pose[:3])
    orientation = _quat_multiply(root_quat, pose[3:])
    return np.concatenate([position, orientation]).astype(np.float32)


def _sample_to_world(
    sample: BrowserTrackingSample, root_pos: np.ndarray, root_quat: np.ndarray
) -> tuple[dict[str, np.ndarray] | None, dict[str, np.ndarray] | None, np.ndarray | None]:
    def hand_to_world(hand):
        if hand is None:
            return None
        return {name: _pose_to_world(pose, root_pos, root_quat) for name, pose in hand.items()}

    head = None if sample.head is None else _pose_to_world(sample.head, root_pos, root_quat)
    return hand_to_world(sample.left_hand), hand_to_world(sample.right_hand), head


def main() -> None:
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.robustness_camera.width = args_cli.head_camera_width
    cfg.scene.robustness_camera.height = args_cli.head_camera_height
    cfg.scene.left_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.left_wrist_camera.height = args_cli.wrist_camera_height
    cfg.scene.right_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.right_wrist_camera.height = args_cli.wrist_camera_height
    cfg.scene.xr_left_eye_camera.width = args_cli.stereo_eye_width
    cfg.scene.xr_left_eye_camera.height = args_cli.stereo_eye_height
    cfg.scene.xr_right_eye_camera.width = args_cli.stereo_eye_width
    cfg.scene.xr_right_eye_camera.height = args_cli.stereo_eye_height
    half_baseline = args_cli.stereo_eye_separation * 0.5
    cfg.scene.xr_left_eye_camera.offset.pos = (0.08, half_baseline, 0.0)
    cfg.scene.xr_right_eye_camera.offset.pos = (0.08, -half_baseline, 0.0)
    set_domain_randomization(cfg, args_cli.domain_randomization)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=args_cli.seed)
    if args_cli.camera_preview:
        open_camera_viewports(
            env.scene,
            ["robustness_camera", "left_wrist_camera", "right_wrist_camera"],
            headless=False,
            width=240,
            height=180,
            columns=3,
        )

    bridge = BrowserTeleopBridge(args_cli.bridge_host, args_cli.bridge_port)
    bridge.start()
    mapper = BimanualTeleopMapper(
        TeleopMappingCfg(position_gain=args_cli.position_gain, rotation_gain=args_cli.rotation_gain)
    )
    gripper_latch = GripperCommandLatch(GRIPPER_SETTINGS.active_sides)
    tracking_guard = TrackingLossGuard(recovery_frames=5, abort_after_s=1.0)
    robot = env.scene["robot"]
    stream_interval = max(1, int(round((1.0 / float(env.step_dt)) / args_cli.stream_fps)))
    previous_clients = -1
    previous_tracking = None
    previous_safety_pause = False
    last_eye_projections: dict[str, np.ndarray] = {}
    last_reported_ipd_m: float | None = None
    metrics_started_at = time.perf_counter()
    metrics_steps = 0
    metrics_frames = 0
    encode_total_ms = 0.0
    server_fps = 0.0

    print(f"[READY] Browser bridge: ws://{args_cli.bridge_host}:{args_cli.bridge_port}")
    print("[CONTROL] In Chrome/IWER, move the HMD and left/right controllers.")
    print("[CONTROL] The first tracked frame calibrates; subsequent motion drives Kuavo head and arms.")
    print("[VIEW] Browser XR: stereo Isaac scene with small head/left-wrist/right-wrist panels.")
    print("[LIMIT] Browser JPEG preview has no CloudXR pose reprojection; use collect_quest_teleop.sh for recording.")
    print(f"[GRIPPER] preset={GRIPPER_SETTINGS.name}; controller pinch drives open/close.")

    step = 0
    try:
        while simulation_app.is_running():
            sample = bridge.latest()
            root_pos = _to_numpy(robot.data.root_pos_w[0])
            root_quat = _to_numpy(robot.data.root_quat_w[0])
            left_hand, right_hand, head_pose = _sample_to_world(sample, root_pos, root_quat)
            mapped = mapper.advance(left_hand, right_hand, head_pose, root_quat)
            tracking = (mapped.left_valid, mapped.right_valid, mapped.head_valid)
            safety = tracking_guard.advance(all(tracking), time.monotonic())
            if tracking != previous_tracking:
                print(f"[TRACKING] left={tracking[0]}, right={tracking[1]}, head={tracking[2]}")
                previous_tracking = tracking
            if safety.recording_paused != previous_safety_pause:
                state = "tracking lost; arms stopped and grippers held" if safety.recording_paused else "tracking recovered"
                print(f"[SAFETY] {state}")
                previous_safety_pause = safety.recording_paused

            calibrations = calibrations_from_tracking(
                sample, args_cli.stereo_eye_width, args_cli.stereo_eye_height
            )
            if len(calibrations) == 2:
                center_camera = env.scene["robustness_camera"]
                center_position = _to_numpy(center_camera.data.pos_w[0])
                center_orientation = _to_numpy(center_camera.data.quat_w_world[0])
                cameras = {
                    "left": env.scene["xr_left_eye_camera"],
                    "right": env.scene["xr_right_eye_camera"],
                }
                for calibration in calibrations:
                    position, orientation = camera_world_pose(
                        center_position, center_orientation, calibration
                    )
                    camera = cameras[calibration.eye]
                    camera.set_world_poses(
                        positions=position[None, :],
                        orientations=orientation[None, :],
                        convention="world",
                    )
                    previous_projection = last_eye_projections.get(calibration.eye)
                    if previous_projection is None or not np.allclose(
                        previous_projection, calibration.projection_matrix, atol=1.0e-4
                    ):
                        camera.set_intrinsic_matrices(calibration.intrinsic_matrix[None, ...])
                        last_eye_projections[calibration.eye] = calibration.projection_matrix
                ipd_m = float(
                    np.linalg.norm(calibrations[0].local_position - calibrations[1].local_position)
                )
                if last_reported_ipd_m is None or abs(ipd_m - last_reported_ipd_m) > 0.001:
                    print(f"[XR CALIBRATION] Quest eye separation={ipd_m * 1000.0:.1f} mm; FOV updated.")
                    last_reported_ipd_m = ipd_m

            desired_gripper = gripper_teleop_action(
                GRIPPER_SETTINGS,
                mapped.left_pinch_m,
                mapped.right_pinch_m,
            )
            safe_gripper = gripper_latch.advance(
                desired_gripper,
                left_valid=mapped.left_valid,
                right_valid=mapped.right_valid,
            )
            action_np = np.concatenate(
                (
                    mapped.action,
                    np.asarray(safe_gripper, dtype=np.float32),
                )
            )
            if not safety.control_allowed:
                action_np[:12] = 0.0
            action = torch.from_numpy(action_np).to(device=env.device).unsqueeze(0)
            env.step(action)
            metrics_steps += 1

            clients = bridge.client_count
            if clients != previous_clients:
                print(f"[BROWSER] connected_clients={clients}")
                previous_clients = clients
                if clients == 0:
                    mapper.reset()
            if clients and step % stream_interval == 0:
                composite = compose_stereo_atlas(
                    _camera_rgb(env.scene["xr_left_eye_camera"]),
                    _camera_rgb(env.scene["xr_right_eye_camera"]),
                    _camera_rgb(env.scene["robustness_camera"]),
                    _camera_rgb(env.scene["left_wrist_camera"]),
                    _camera_rgb(env.scene["right_wrist_camera"]),
                )
                encode_started_at = time.perf_counter()
                ok, encoded = cv2.imencode(
                    ".jpg", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR),
                    [int(cv2.IMWRITE_JPEG_QUALITY), args_cli.jpeg_quality],
                )
                encode_ms = (time.perf_counter() - encode_started_at) * 1000.0
                if ok:
                    bridge.publish_frame(
                        encoded.tobytes(),
                        tracking_sequence=sample.sequence,
                        client_timestamp_ms=sample.client_timestamp_ms,
                        encode_ms=encode_ms,
                        server_fps=server_fps,
                    )
                    metrics_frames += 1
                    encode_total_ms += encode_ms
            step += 1
            metrics_elapsed = time.perf_counter() - metrics_started_at
            if metrics_elapsed >= 1.0:
                server_fps = metrics_steps / metrics_elapsed
                stream_fps = metrics_frames / metrics_elapsed
                mean_encode_ms = encode_total_ms / max(metrics_frames, 1)
                client = bridge.client_metrics
                client_age = time.monotonic() - client.received_at if client.received_at else float("inf")
                client_text = (
                    f"client={client.rendered_fps:.1f}fps latency={client.pose_to_frame_ms:.1f}ms "
                    f"decode={client.decode_ms:.1f}ms dropped={client.dropped_frames}"
                    if client_age < 2.5
                    else "client=waiting"
                )
                print(
                    f"[XR METRICS] sim={server_fps:.1f}fps stream={stream_fps:.1f}fps "
                    f"encode={mean_encode_ms:.1f}ms {client_text}"
                )
                metrics_started_at = time.perf_counter()
                metrics_steps = 0
                metrics_frames = 0
                encode_total_ms = 0.0
    except KeyboardInterrupt:
        print("\n[INFO] Browser interaction preview interrupted.")
    finally:
        bridge.close()
        env.close()
        print(f"[RESULT] Browser interaction preview closed after {step} simulation step(s).")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
