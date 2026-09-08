#!/usr/bin/env python3
"""Task1-only live browser preview and pregrasp runner.

This file intentionally lives under ``data_collection`` instead of changing
the general-purpose teleop preview.  It keeps the existing browser protocol
and camera display, but starts the deterministic Task1 pregrasp only after a
browser client is connected so the whole settle/move sequence is visible.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys
import time

from isaaclab.app import AppLauncher
from kuavo_isaaclab_scene.robots.gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
    gripper_teleop_action,
    resolve_gripper_settings,
)
from kuavo_isaaclab_scene.robots.robot_model import add_robot_model_cli_args, export_robot_model_cli, resolve_robot_model


parser = argparse.ArgumentParser(description="Task1 box-pick preview through the live browser bridge.")
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
parser.add_argument(
    "--pregrasp",
    action="store_true",
    help="Move both open end-effectors to the fixed-box pregrasp targets once after startup.",
)
parser.add_argument("--pregrasp-distance-m", type=float, default=0.10)
parser.add_argument("--pregrasp-grasp-depth-m", type=float, default=0.015)
parser.add_argument("--pregrasp-steps", type=int, default=300)
parser.add_argument("--pregrasp-initial-state", default="quest_ready_02")
parser.add_argument("--pregrasp-settle-steps", type=int, default=120)
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

# A remote server can stream the RTX camera/composite frames without exposing
# an Isaac desktop window.  In that mode the browser bridge remains active and
# only the local camera viewport panels are skipped.
if args_cli.headless:
    args_cli.camera_preview = False
if not 1 <= args_cli.bridge_port <= 65535:
    parser.error("--bridge-port must be between 1 and 65535.")
if args_cli.stream_fps <= 0.0:
    parser.error("--stream-fps must be positive.")
if args_cli.pregrasp and args_cli.pregrasp_steps <= 0:
    parser.error("--pregrasp-steps must be positive.")
if args_cli.pregrasp and args_cli.pregrasp_settle_steps < 0:
    parser.error("--pregrasp-settle-steps must be nonnegative.")
if args_cli.pregrasp_distance_m <= 0.0 or not math.isfinite(args_cli.pregrasp_distance_m):
    parser.error("--pregrasp-distance-m must be finite and positive.")
if args_cli.pregrasp_grasp_depth_m <= 0.0 or not math.isfinite(args_cli.pregrasp_grasp_depth_m):
    parser.error("--pregrasp-grasp-depth-m must be finite and positive.")
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

from kuavo_isaaclab_scene.teleop.browser_teleop_bridge import BrowserTeleopBridge, BrowserTrackingSample
from kuavo_isaaclab_scene.teleop.browser_teleop_control import browser_body_action, compose_browser_action
from kuavo_isaaclab_scene.display.camera_viewports import open_camera_viewports
from kuavo_isaaclab_scene.display.stereo_camera_calibration import calibrations_from_tracking, camera_world_pose
from kuavo_isaaclab_scene.display.stereo_compositor import compose_stereo_atlas
from kuavo_isaaclab_scene.teleop.teleop_safety import GripperCommandLatch, TrackingLossGuard
from kuavo_isaaclab_scene.envs.teleop_env import KuavoQuestTeleopEnvCfg, set_domain_randomization
from kuavo_isaaclab_scene.teleop.teleop_mapping import BimanualTeleopMapper, TeleopMappingCfg
from kuavo_isaaclab_scene.teleop.teleop_body import TeleopBodyMapper


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


def _pregrasp_targets(env, *, distance_m: float, grasp_depth_m: float):
    """Compute the two flap-normal pregrasp points from the live box pose."""
    import torch
    from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry

    if not np.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("pregrasp distance must be finite and positive")
    if not np.isfinite(grasp_depth_m) or grasp_depth_m <= 0.0:
        raise ValueError("grasp depth must be finite and positive")

    box = env.scene["medium_box_0"]
    geometry = box_geometry(env.cfg.scene.medium_box_0, ("flap_right", "flap_left"))
    flap_ids, flap_names = box.find_bodies(("flap_right", "flap_left"), preserve_order=True)
    body_ids, body_names = box.find_bodies("Body")
    if len(flap_ids) != 2 or len(body_ids) != 1:
        raise RuntimeError(
            f"MediumBox_0 body lookup failed: flaps={flap_names}, body={body_names}"
        )

    flap_pos = box.data.body_link_pos_w[0, flap_ids]
    flap_quat = box.data.body_link_quat_w[0, flap_ids]
    body_pos = box.data.body_link_pos_w[0, body_ids[0]]
    targets = []
    diagnostics = []
    for index, flap_name in enumerate(("flap_right", "flap_left")):
        flap = geometry.flaps[flap_name]
        local_grasp = torch.tensor(flap.center, device=env.device, dtype=flap_pos.dtype)
        local_grasp[2] += flap.half_size[2] - grasp_depth_m
        # Isaac's quat_apply keeps the wxyz convention used by the USD asset.
        from isaaclab.utils.math import quat_apply
        grasp = flap_pos[index] + quat_apply(flap_quat[index], local_grasp)
        local_normal = torch.tensor(
            (1.0, 0.0, 0.0) if flap_name == "flap_right" else (-1.0, 0.0, 0.0),
            device=env.device,
            dtype=flap_pos.dtype,
        )
        outward = quat_apply(flap_quat[index], local_normal)
        outward = outward / outward.norm().clamp_min(1.0e-6)
        if torch.dot(outward, flap_pos[index] - body_pos) < 0:
            outward = -outward
        pregrasp = grasp + outward * distance_m
        targets.append(pregrasp)
        diagnostics.append(
            {
                "flap": flap_name,
                "grasp_position_w": grasp.detach().cpu().tolist(),
                "outward_normal_w": outward.detach().cpu().tolist(),
                "pregrasp_position_w": pregrasp.detach().cpu().tolist(),
            }
        )
    return torch.stack(targets), diagnostics


def _absolute_pose_action(env, robot, positions_w, orientations_w):
    """Build the arm absolute-pose action in the IK root frame."""
    import torch
    from isaaclab.utils.math import subtract_frame_transforms

    root_pos = robot.data.root_pos_w.expand(positions_w.shape[0], -1)
    root_quat = robot.data.root_quat_w.expand(positions_w.shape[0], -1)
    positions_b, orientations_b = subtract_frame_transforms(
        root_pos, root_quat, positions_w, orientations_w
    )
    action = torch.zeros((1, env.action_manager.total_action_dim), device=env.device)
    offset = 0
    for name, dimension in zip(env.action_manager.active_terms, env.action_manager.action_term_dim):
        if name == "left_arm":
            action[0, offset : offset + dimension] = torch.cat((positions_b[0], orientations_b[0]))
        elif name == "right_arm":
            action[0, offset : offset + dimension] = torch.cat((positions_b[1], orientations_b[1]))
        offset += dimension
    return action


class _LivePregraspRunner:
    """One-shot bimanual position move used by the remote browser preview."""

    def __init__(
        self,
        env,
        *,
        distance_m: float,
        grasp_depth_m: float,
        steps: int,
        initial_state: str,
        settle_steps: int,
    ):
        if steps <= 0:
            raise ValueError("pregrasp steps must be positive")
        if settle_steps < 0:
            raise ValueError("pregrasp settle steps must be nonnegative")
        self.env = env
        self.robot = env.scene["robot"]
        self.arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
        self.ee_ids, _ = self.robot.find_bodies(
            ("zarm_l7_end_effector", "zarm_r7_end_effector"), preserve_order=True
        )
        if len(self.ee_ids) != 2:
            raise RuntimeError("Could not resolve both Kuavo TCP links for pregrasp.")

        # Use the same stationary task reset as the smoke runner. This puts
        # the robot in the captured ready posture; q7 is not commanded by the
        # pregrasp move itself.
        from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
        from kuavo_isaaclab_scene.robots.initial_states import apply_initial_state, load_initial_state
        from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model

        state = load_initial_state(
            initial_state,
            robot_model=resolve_robot_model().name,
            gripper=resolve_gripper_settings().name,
        )
        apply_initial_state(env, None, state, initial_state)
        env.scene.write_data_to_sim()
        self.distance_m = distance_m
        self.grasp_depth_m = grasp_depth_m
        self.initial_state = initial_state
        self.remaining = 0
        self.total = int(steps)
        self.settle_remaining = int(settle_steps)
        self.phase = "settle" if self.settle_remaining else "move"
        self.targets_w = None
        self.geometry = None
        self.initial_q7 = None
        self.action = None
        self.finished = False
        self.finalized = False
        self.hold_action = self._make_hold_action()
        if self.phase == "move":
            self._start_motion(initial_state)
        print(
            f"[PREGRASP] browser_connected initial_state={initial_state} "
            f"settle_steps={settle_steps}; motion_steps={self.total}; "
            "q7 direct command=disabled",
            flush=True,
        )

    def _make_hold_action(self):
        return _absolute_pose_action(
            self.env,
            self.robot,
            self.robot.data.body_link_pos_w[0, self.ee_ids].clone(),
            self.robot.data.body_link_quat_w[0, self.ee_ids].clone(),
        )

    def _start_motion(self, initial_state):
        self.targets_w, self.geometry = _pregrasp_targets(
            self.env, distance_m=self.distance_m, grasp_depth_m=self.grasp_depth_m
        )
        self.initial_q7 = self.robot.data.joint_pos[0, [
            self.robot.find_joints("zarm_l7_joint", preserve_order=True)[0][0],
            self.robot.find_joints("zarm_r7_joint", preserve_order=True)[0][0],
        ]].clone()
        orientations_w = self.robot.data.body_link_quat_w[0, self.ee_ids].clone()
        self.action = _absolute_pose_action(self.env, self.robot, self.targets_w, orientations_w)
        # Keep the current redundancy posture, including q7, as a soft
        # null-space target. No direct q7 command is issued by this preview.
        for term in self.arm_terms:
            term.set_posture_target(
                self.robot.data.joint_pos[:, term._joint_ids].clone(), weight=0.60
            )
            term.orientation_weight = 0.5
            term.set_following(True)
        self.remaining = self.total
        self.phase = "move"
        print(
            f"[PREGRASP] targets_ready initial_state={initial_state} "
            f"targets_w={self.geometry}",
            flush=True,
        )

    def next_action(self):
        if self.finished:
            return self.hold_action
        if self.phase == "settle":
            action = self.hold_action
            self.settle_remaining -= 1
            if self.settle_remaining <= 0:
                self._start_motion(self.initial_state)
            return action
        action = self.action
        self.remaining -= 1
        if self.remaining <= 0:
            self.finished = True
        return action

    def finish(self):
        if not self.finished or self.finalized:
            return
        if self.targets_w is None or self.initial_q7 is None:
            return
        final_pos = self.robot.data.body_link_pos_w[0, self.ee_ids]
        errors = (self.targets_w - final_pos).norm(dim=-1)
        joint_ids = [
            self.robot.find_joints("zarm_l7_joint", preserve_order=True)[0][0],
            self.robot.find_joints("zarm_r7_joint", preserve_order=True)[0][0],
        ]
        final_q7 = self.robot.data.joint_pos[0, joint_ids]
        print(
            f"[PREGRASP] done position_error_m={errors.detach().cpu().tolist()} "
            f"q7_start={self.initial_q7.detach().cpu().tolist()} "
            f"q7_end={final_q7.detach().cpu().tolist()}",
            flush=True,
        )
        for term in self.arm_terms:
            term.clear_posture_target()
            # Keep absolute-pose mode for the hold action. The normal browser
            # delta mapper is intentionally not resumed in this one-shot run.
            term.hold_current_pose()
            term.orientation_weight = 0.5
        self.hold_action = _absolute_pose_action(
            self.env,
            self.robot,
            self.robot.data.body_link_pos_w[0, self.ee_ids].clone(),
            self.robot.data.body_link_quat_w[0, self.ee_ids].clone(),
        )
        self.finalized = True


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
    if args_cli.pregrasp:
        # The one-shot pregrasp action is an absolute root-frame pose.  The
        # terms are switched back to relative teleop after the move finishes.
        cfg.actions.left_arm.controller.use_relative_mode = False
        cfg.actions.right_arm.controller.use_relative_mode = False
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
    if args_cli.camera_preview and not args_cli.headless:
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
    robot_model = resolve_robot_model()
    body_mapper = TeleopBodyMapper(robot_model.urdf_path, has_wheel_base=robot_model.has_wheel_base)
    arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
    robot = env.scene["robot"]
    pregrasp_ee_ids, pregrasp_ee_names = robot.find_bodies(
        ("zarm_l7_end_effector", "zarm_r7_end_effector"), preserve_order=True
    )
    if len(pregrasp_ee_ids) != 2:
        raise RuntimeError(
            f"Could not resolve both Kuavo TCP links for pregrasp: {pregrasp_ee_names}"
        )
    # Do not reset or move the robot until the browser has an active bridge
    # client. This keeps the full pregrasp motion visible in the live view.
    pregrasp_runner = None
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
    print("[CONTROL] Left stick=base forward/strafe; right stick=turn/torso lift; index triggers=grippers.")
    print("[CONTROL] Body motion requires both tracked controllers and head; tracking loss stops the base and holds torso height.")
    if args_cli.headless:
        print("[VIEW] Headless server mode: browser XR stream enabled; local Isaac viewports disabled.")
    else:
        print("[VIEW] Browser XR: stereo Isaac scene with small head/left-wrist/right-wrist panels.")
    print("[LIMIT] Browser JPEG preview has no CloudXR pose reprojection; use collect_quest_teleop.sh for recording.")
    print(f"[GRIPPER] preset={GRIPPER_SETTINGS.name}; controller triggers or tracked-hand pinch drive open/close.")

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

            desired_gripper = list(gripper_teleop_action(
                GRIPPER_SETTINGS,
                mapped.left_pinch_m,
                mapped.right_pinch_m,
            ))
            controllers = {"left": sample.left_controller, "right": sample.right_controller}
            for index, side in enumerate(GRIPPER_SETTINGS.active_sides):
                controller = controllers[side]
                if controller is not None:
                    desired_gripper[index] = -1.0 if controller.trigger >= 0.5 else 1.0
            safe_gripper = gripper_latch.advance(
                tuple(desired_gripper),
                left_valid=mapped.left_valid,
                right_valid=mapped.right_valid,
            )
            body_action = browser_body_action(
                sample, body_mapper, env.step_dt, control_allowed=safety.control_allowed
            )
            action_np = compose_browser_action(mapped.action, safe_gripper, body_action)
            if not safety.control_allowed:
                action_np[:12] = 0.0
            if args_cli.pregrasp:
                if pregrasp_runner is None and bridge.client_count > 0:
                    print("[PREGRASP] browser client detected; starting live sequence", flush=True)
                    pregrasp_runner = _LivePregraspRunner(
                        env,
                        distance_m=args_cli.pregrasp_distance_m,
                        grasp_depth_m=args_cli.pregrasp_grasp_depth_m,
                        steps=args_cli.pregrasp_steps,
                        initial_state=args_cli.pregrasp_initial_state,
                        settle_steps=args_cli.pregrasp_settle_steps,
                    )
                # Before the browser connects, hold the current pose with a
                # valid absolute action. Once connected, the runner returns
                # one action per loop so each step can be streamed.
                for term in arm_terms:
                    term.set_following(True)
                action = (
                    _absolute_pose_action(
                        env,
                        robot,
                        robot.data.body_link_pos_w[0, pregrasp_ee_ids].clone(),
                        robot.data.body_link_quat_w[0, pregrasp_ee_ids].clone(),
                    )
                    if pregrasp_runner is None
                    else pregrasp_runner.next_action()
                )
            else:
                for term in arm_terms:
                    term.set_following(safety.control_allowed)
                action = torch.from_numpy(action_np).to(device=env.device).unsqueeze(0)
            env.step(action)
            if pregrasp_runner is not None and pregrasp_runner.finished:
                pregrasp_runner.finish()
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
                # The remote browser preview currently presents the complete
                # atlas upside-down. Flip only the published video frame;
                # tracking and control packets remain unchanged.
                composite = cv2.flip(composite, 0)
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
