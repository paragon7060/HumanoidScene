#!/usr/bin/env python3
"""Task1-only live browser preview and pregrasp runner.

This file intentionally lives under ``data_collection`` instead of changing
the general-purpose teleop preview.  It keeps the existing browser protocol
and camera display, but starts the deterministic Task1 pregrasp only after a
browser client is connected so the whole settle/move sequence is visible.
"""

from __future__ import annotations

import argparse
import json
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
parser.add_argument(
    "--pregrasp-height-m",
    "--pregrasp-distance-m",
    dest="pregrasp_height_m",
    type=float,
    default=0.10,
    help=(
        "Height above the selected left/right upper grasp pair, measured along "
        "the robot-base +Z axis. --pregrasp-distance-m is a compatibility alias."
    ),
)
parser.add_argument("--pregrasp-grasp-depth-m", type=float, default=0.015)
parser.add_argument("--pregrasp-steps", type=int, default=300)
parser.add_argument("--pregrasp-initial-state", default="task1_ready_bent")
parser.add_argument("--pregrasp-settle-steps", type=int, default=120)
parser.add_argument(
    "--wrist6-limit-test",
    action="store_true",
    help=(
        "After pregrasp, keep the TCP positions and bend each hand-pitch q6 "
        "toward the joint limit matching its current sign."
    ),
)
parser.add_argument("--wrist6-test-steps", type=int, default=120)
parser.add_argument(
    "--solve-downward-ready",
    action="store_true",
    help="One-off validation: solve a bent initial pose whose gripper local -Z points base-down.",
)
parser.add_argument(
    "--joint-pose-editor",
    action="store_true",
    help="Serve a static third-person browser editor for the fourteen arm joints.",
)
parser.add_argument("--editor-camera-width", type=int, default=1280)
parser.add_argument("--editor-camera-height", type=int, default=720)
parser.add_argument(
    "--planning-snapshot-output",
    type=Path,
    default=None,
    metavar="DIR",
    help="Write the live Task1 collision/planning snapshot once after pose-editor startup.",
)
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
if args_cli.wrist6_limit_test and not args_cli.pregrasp:
    parser.error("--wrist6-limit-test requires --pregrasp.")
if args_cli.joint_pose_editor and args_cli.pregrasp:
    parser.error("--joint-pose-editor and --pregrasp are separate modes.")
if args_cli.planning_snapshot_output is not None and not args_cli.joint_pose_editor:
    parser.error("--planning-snapshot-output requires --joint-pose-editor.")
if args_cli.wrist6_limit_test and args_cli.wrist6_test_steps <= 0:
    parser.error("--wrist6-test-steps must be positive.")
if args_cli.pregrasp_height_m <= 0.0 or not math.isfinite(args_cli.pregrasp_height_m):
    parser.error("--pregrasp-height-m must be finite and positive.")
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
    args_cli.editor_camera_width,
    args_cli.editor_camera_height,
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


def _pregrasp_targets(env, *, height_m: float, grasp_depth_m: float):
    """Compute upper grasp points, pregrasp points, and inward flap normals."""
    import torch
    from kuavo_isaaclab_scene.rl.scenes.asset_geometry import box_geometry

    if not np.isfinite(height_m) or height_m <= 0.0:
        raise ValueError("pregrasp height must be finite and positive")
    if not np.isfinite(grasp_depth_m) or grasp_depth_m <= 0.0:
        raise ValueError("grasp depth must be finite and positive")

    box = env.scene["medium_box_0"]
    robot = env.scene["robot"]
    geometry = box_geometry(env.cfg.scene.medium_box_0, ("flap_right", "flap_left"))
    flap_ids, flap_names = box.find_bodies(("flap_right", "flap_left"), preserve_order=True)
    if len(flap_ids) != 2:
        raise RuntimeError(f"MediumBox_0 flap lookup failed: flaps={flap_names}")

    flap_pos = box.data.body_link_pos_w[0, flap_ids]
    flap_quat = box.data.body_link_quat_w[0, flap_ids]
    body_ids, body_names = box.find_bodies("Body")
    if len(body_ids) != 1:
        raise RuntimeError(f"MediumBox_0 body lookup failed: body={body_names}")
    body_pos = box.data.body_link_pos_w[0, body_ids[0]]
    from isaaclab.utils.math import quat_apply

    base_up_local = torch.tensor((0.0, 0.0, 1.0), device=env.device, dtype=flap_pos.dtype)
    base_up_w = quat_apply(robot.data.root_quat_w[0], base_up_local)
    base_up_w = base_up_w / base_up_w.norm().clamp_min(1.0e-6)
    targets = []
    grasps = []
    inward_normals = []
    diagnostics = []
    for index, flap_name in enumerate(("flap_right", "flap_left")):
        flap = geometry.flaps[flap_name]
        local_grasp = torch.tensor(flap.center, device=env.device, dtype=flap_pos.dtype)
        local_grasp[2] += flap.half_size[2] - grasp_depth_m
        # Isaac's quat_apply keeps the wxyz convention used by the USD asset.
        grasp = flap_pos[index] + quat_apply(flap_quat[index], local_grasp)
        pregrasp = grasp + base_up_w * height_m
        local_normal = torch.tensor(
            (1.0, 0.0, 0.0) if flap_name == "flap_right" else (-1.0, 0.0, 0.0),
            device=env.device,
            dtype=flap_pos.dtype,
        )
        outward = quat_apply(flap_quat[index], local_normal)
        outward = outward / outward.norm().clamp_min(1.0e-6)
        if torch.dot(outward, flap_pos[index] - body_pos) < 0.0:
            outward = -outward
        inward = -outward
        targets.append(pregrasp)
        grasps.append(grasp)
        inward_normals.append(inward)
        diagnostics.append(
            {
                "flap": flap_name,
                "grasp_position_w": grasp.detach().cpu().tolist(),
                "base_up_w": base_up_w.detach().cpu().tolist(),
                "inward_flap_normal_w": inward.detach().cpu().tolist(),
                "pregrasp_position_w": pregrasp.detach().cpu().tolist(),
            }
        )
    return torch.stack(targets), torch.stack(grasps), torch.stack(inward_normals), diagnostics


def _orientation_from_closing_and_forward(closing_axes_w, forward_axes_w):
    """Return wxyz TCP quaternions for local +X closing and local -Z forward."""
    from isaaclab.utils.math import quat_from_matrix

    closing = torch.nn.functional.normalize(closing_axes_w, dim=-1)
    # Remove any numerical component parallel to the flap normal.
    forward = forward_axes_w - (forward_axes_w * closing).sum(-1, keepdim=True) * closing
    forward = torch.nn.functional.normalize(forward, dim=-1)
    local_z = -forward
    local_y = torch.nn.functional.normalize(torch.cross(local_z, closing, dim=-1), dim=-1)
    rotation = torch.stack((closing, local_y, local_z), dim=-1)
    return torch.nn.functional.normalize(quat_from_matrix(rotation), dim=-1)


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


def _task_q6_constraints(robot, arm_terms, *, exact_maximum: bool = False):
    """Keep both physical wrist pitches in the 45--75 degree bent band."""
    q6_ids, q6_names = robot.find_joints(
        ("zarm_l6_joint", "zarm_r6_joint"), preserve_order=True
    )
    local_indices = []
    targets = []
    for side, term, joint_id in zip(("left", "right"), arm_terms, q6_ids):
        term_ids = [int(value) for value in term._joint_ids]
        local_index = term_ids.index(int(joint_id))
        local_indices.append(local_index)
        physical = robot.data.joint_pos_limits[0, joint_id]
        if side == "left":
            lower, upper, target = float(physical[0]), -math.radians(45.0), float(physical[0])
            if exact_maximum:
                upper = lower
        else:
            lower, upper, target = math.radians(45.0), float(physical[1]), float(physical[1])
            if exact_maximum:
                lower = upper
        term.set_control_joint_bounds((local_index,), (lower,), (upper,))
        posture = robot.data.joint_pos[:, term._joint_ids].clone()
        posture[:, local_index] = target
        term.set_posture_target(
            posture, weight=0.60, direct_indices=(local_index,), direct_gain=2.0
        )
        targets.append(target)
    return q6_ids, q6_names, local_indices, torch.tensor(
        targets, device=robot.data.joint_pos.device, dtype=robot.data.joint_pos.dtype
    )


def _load_task_ready(env, initial_state: str):
    """Load the precomputed bent Task1 pose without a runtime preparation move."""
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
    env.sim.forward()
    env.scene.update(env.step_dt)
    print(f"[TASK_READY] loaded_initial_state={initial_state}; no runtime wrist preparation", flush=True)


def _solve_downward_ready(env, steps: int = 240):
    """One-off IK validation for a downward-looking gripper initial pose."""
    from isaaclab.utils.math import quat_apply

    robot = env.scene["robot"]
    arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
    ee_ids, _ = robot.find_bodies(
        ("zarm_l7_end_effector", "zarm_r7_end_effector"), preserve_order=True
    )
    camera_ids, _ = robot.find_bodies(("l_d405_camera", "r_d405_camera"), preserve_order=True)
    # Prefer the physical 75-degree stop, but permit only the agreed
    # one-sided 30-degree relaxation while generating the static preset.
    q6_ids, _, _, _ = _task_q6_constraints(robot, arm_terms)
    root_quat = robot.data.root_quat_w[0]
    dtype = robot.data.body_link_pos_w.dtype
    base_y = quat_apply(
        root_quat, torch.tensor((0.0, 1.0, 0.0), device=env.device, dtype=dtype)
    )
    base_down = quat_apply(
        root_quat, torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=dtype)
    )
    closing = torch.stack((-base_y, base_y))
    forward = base_down.expand_as(closing)
    orientations = _orientation_from_closing_and_forward(closing, forward)
    positions = robot.data.body_link_pos_w[0, ee_ids].clone()
    action = _absolute_pose_action(env, robot, positions, orientations)
    for term in arm_terms:
        term.orientation_weight = 1.0
        term.set_following(True)
    for _ in range(int(steps)):
        env.step(action)
    tool_forward = quat_apply(
        robot.data.body_link_quat_w[0, ee_ids],
        torch.tensor((0.0, 0.0, -1.0), device=env.device, dtype=dtype).expand(2, -1),
    )
    camera_forward = quat_apply(
        robot.data.body_link_quat_w[0, camera_ids],
        torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=dtype).expand(2, -1),
    )
    base_forward = quat_apply(
        root_quat, torch.tensor((1.0, 0.0, 0.0), device=env.device, dtype=dtype)
    )
    arm_state = {
        name: float(robot.data.joint_pos[0, index].item())
        for index, name in enumerate(robot.joint_names)
        if name.startswith("zarm_")
    }
    print(
        f"[DOWNWARD_READY] q6={robot.data.joint_pos[0, q6_ids].detach().cpu().tolist()} "
        f"tool_down_dot={(tool_forward * base_down).sum(-1).detach().cpu().tolist()} "
        f"camera_forward_dot={(camera_forward * base_forward).sum(-1).detach().cpu().tolist()} "
        f"position_error_m={(positions - robot.data.body_link_pos_w[0, ee_ids]).norm(dim=-1).detach().cpu().tolist()} "
        f"arm_joint_positions={arm_state}",
        flush=True,
    )
    for term in arm_terms:
        term.hold_current_pose()


class _JointPoseEditor:
    """Static arm-pose authoring with third-person camera and joint-axis markers.

    This mode teleports only the robot arm joints for pose authoring.  It is
    deliberately separate from collection and must never produce expert data.
    """

    _VIEWS = {
        "rear_left": ((-2.2, 1.8, 1.65), (0.45, 0.0, 1.15)),
        "rear_right": ((-2.2, -1.8, 1.65), (0.45, 0.0, 1.15)),
        "front_left": ((2.0, 1.8, 1.65), (0.20, 0.0, 1.15)),
        "front_right": ((2.0, -1.8, 1.65), (0.20, 0.0, 1.15)),
    }

    def __init__(self, env, *, pregrasp_height_m: float, grasp_depth_m: float):
        import isaaclab.sim as sim_utils
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from kuavo_isaaclab_scene.planning.robot_model import UrdfModel

        self.env = env
        self.robot = env.scene["robot"]
        self.arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
        self.joint_names = tuple(
            f"zarm_{side}{index}_joint" for side in ("l", "r") for index in range(1, 8)
        )
        self.joint_ids, resolved = self.robot.find_joints(self.joint_names, preserve_order=True)
        if tuple(resolved) != self.joint_names:
            raise RuntimeError(f"Pose editor arm-joint lookup mismatch: {resolved}")
        body_names = tuple(
            f"zarm_{side}{index}_link" for side in ("l", "r") for index in range(1, 8)
        )
        self.body_ids, resolved_bodies = self.robot.find_bodies(body_names, preserve_order=True)
        if tuple(resolved_bodies) != body_names:
            raise RuntimeError(f"Pose editor arm-body lookup mismatch: {resolved_bodies}")
        self.ee_ids, _ = self.robot.find_bodies(
            ("zarm_l7_end_effector", "zarm_r7_end_effector"), preserve_order=True
        )
        self.limits = self.robot.data.joint_pos_limits[0, self.joint_ids].clone()
        self.reset_targets = self.robot.data.joint_pos[:, self.joint_ids].clone()
        self.targets = self.reset_targets.clone()
        self.pregrasp_targets_w, _, _, self.pregrasp_geometry = _pregrasp_targets(
            env, height_m=pregrasp_height_m, grasp_depth_m=grasp_depth_m
        )
        self.pregrasp_height_m = float(pregrasp_height_m)
        model = UrdfModel(resolve_robot_model().urdf_path)
        self.local_axes = torch.tensor(
            [model.joints[name].axis for name in self.joint_names],
            device=env.device,
            dtype=self.targets.dtype,
        )
        print("[POSE_EDITOR_INIT] resolved 14 arm joints and physical axes", flush=True)
        marker_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Task1JointPoseEditor",
            markers={
                "left": sim_utils.CylinderCfg(
                    radius=0.012,
                    height=0.16,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.05, 0.72, 1.0), emissive_color=(0.0, 0.08, 0.16)
                    ),
                ),
                "right": sim_utils.CylinderCfg(
                    radius=0.012,
                    height=0.16,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.12, 0.70), emissive_color=(0.14, 0.0, 0.08)
                    ),
                ),
                "selected": sim_utils.CylinderCfg(
                    radius=0.019,
                    height=0.22,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.72, 0.04), emissive_color=(0.20, 0.10, 0.0)
                    ),
                ),
            },
        )
        self.markers = VisualizationMarkers(marker_cfg)
        print("[POSE_EDITOR_INIT] spawned joint-axis cylinder markers", flush=True)
        self.selected_joint = self.joint_names[5]
        self.view = "rear_left"
        self.last_command_sequence = -1
        self.set_view(self.view)
        print("[POSE_EDITOR_INIT] positioned third-person camera", flush=True)
        self.apply_targets()
        print("[POSE_EDITOR_INIT] static arm pose ready", flush=True)

    def set_view(self, name: str) -> None:
        from isaaclab.utils.math import quat_apply

        if name not in self._VIEWS:
            raise ValueError(f"Unknown editor camera view: {name}")
        eye_b, target_b = self._VIEWS[name]
        root_pos = self.robot.data.root_pos_w[0]
        root_quat = self.robot.data.root_quat_w[0]
        dtype = root_pos.dtype
        eye = root_pos + quat_apply(
            root_quat, torch.tensor(eye_b, device=self.env.device, dtype=dtype)
        )
        target = root_pos + quat_apply(
            root_quat, torch.tensor(target_b, device=self.env.device, dtype=dtype)
        )
        self.env.scene["joint_editor_camera"].set_world_poses_from_view(
            eye.unsqueeze(0), target.unsqueeze(0)
        )
        self.view = name

    def accept(self, command) -> bool:
        if command is None or command.sequence <= self.last_command_sequence:
            return False
        self.last_command_sequence = command.sequence
        if command.action == "set_joint":
            index = self.joint_names.index(command.joint_name)
            value = torch.tensor(command.value_rad, device=self.env.device, dtype=self.targets.dtype)
            self.targets[0, index] = torch.clamp(value, self.limits[index, 0], self.limits[index, 1])
            self.selected_joint = command.joint_name
            self.apply_targets()
        elif command.action == "set_pose":
            for joint_name, value_rad in command.joint_positions.items():
                index = self.joint_names.index(joint_name)
                value = torch.tensor(value_rad, device=self.env.device, dtype=self.targets.dtype)
                self.targets[0, index] = torch.clamp(value, self.limits[index, 0], self.limits[index, 1])
            self.selected_joint = next(iter(command.joint_positions))
            self.apply_targets()
        elif command.action == "reset":
            self.targets.copy_(self.reset_targets)
            self.selected_joint = self.joint_names[5]
            self.apply_targets()
        elif command.action == "set_view":
            self.set_view(command.view)
        elif command.action == "print_pose":
            values = {
                name: float(value) for name, value in zip(self.joint_names, self.targets[0].tolist())
            }
            print(f"[POSE_EDITOR_EXPORT] arm_joint_positions={values}", flush=True)
        return True

    def apply_targets(self) -> None:
        velocities = torch.zeros_like(self.targets)
        self.robot.write_joint_state_to_sim(
            self.targets, velocities, joint_ids=self.joint_ids
        )
        self.robot.set_joint_position_target(self.targets, joint_ids=self.joint_ids)
        self.env.sim.forward()
        self.env.scene.update(self.env.step_dt)
        for term in self.arm_terms:
            # Pausing captures the joint hold only on a following -> paused
            # transition. Re-arm that transition after every editor teleport
            # so the next manager step cannot restore an older pose.
            term.set_following(True)
            term.set_following(False)
        self.update_markers()

    def update_markers(self) -> None:
        from isaaclab.utils.math import quat_apply

        positions = self.robot.data.body_link_pos_w[0, self.body_ids]
        body_quats = self.robot.data.body_link_quat_w[0, self.body_ids]
        axes = torch.nn.functional.normalize(quat_apply(body_quats, self.local_axes), dim=-1)
        z_axis = torch.zeros_like(axes)
        z_axis[:, 2] = 1.0
        vector = torch.cross(z_axis, axes, dim=-1)
        scalar = 1.0 + (z_axis * axes).sum(-1, keepdim=True)
        orientations = torch.nn.functional.normalize(torch.cat((scalar, vector), dim=-1), dim=-1)
        antiparallel = scalar[:, 0] < 1.0e-5
        if torch.any(antiparallel):
            orientations[antiparallel] = torch.tensor(
                (0.0, 1.0, 0.0, 0.0), device=self.env.device, dtype=orientations.dtype
            )
        marker_indices = torch.tensor(
            [0] * 7 + [1] * 7, device=self.env.device, dtype=torch.int32
        )
        marker_indices[self.joint_names.index(self.selected_joint)] = 2
        self.markers.visualize(positions, orientations, marker_indices=marker_indices)

    def state(self) -> dict:
        from isaaclab.utils.math import quat_apply, subtract_frame_transforms

        values = self.targets[0]
        ee_pos = self.robot.data.body_link_pos_w[0, self.ee_ids]
        ee_quat = self.robot.data.body_link_quat_w[0, self.ee_ids]
        root_pos = self.robot.data.root_pos_w.expand(2, -1)
        root_quat = self.robot.data.root_quat_w.expand(2, -1)
        tcp_pos_b, _ = subtract_frame_transforms(root_pos, root_quat, ee_pos, ee_quat)
        pregrasp_pos_b, _ = subtract_frame_transforms(
            root_pos, root_quat, self.pregrasp_targets_w, root_quat
        )
        tool_forward = quat_apply(
            ee_quat,
            torch.tensor((0.0, 0.0, -1.0), device=self.env.device, dtype=ee_pos.dtype).expand(2, -1),
        )
        base_down = quat_apply(
            root_quat,
            torch.tensor((0.0, 0.0, -1.0), device=self.env.device, dtype=ee_pos.dtype).expand(2, -1),
        )
        dots = (tool_forward * base_down).sum(-1).clamp(-1.0, 1.0)
        return {
            "view": self.view,
            "selected_joint": self.selected_joint,
            "joints": [
                {
                    "name": name,
                    "value": float(value),
                    "degrees": math.degrees(float(value)),
                    "lower": float(limit[0]),
                    "upper": float(limit[1]),
                }
                for name, value, limit in zip(self.joint_names, values, self.limits)
            ],
            "tool_down_angle_deg": [math.degrees(math.acos(float(dot))) for dot in dots],
            "tcp_position_b": tcp_pos_b.detach().cpu().tolist(),
            "pregrasp_position_b": pregrasp_pos_b.detach().cpu().tolist(),
            "pregrasp_height_m": self.pregrasp_height_m,
            "pregrasp_source": "medium_box_0 flap_right/flap_left upper grasp pair",
            "authoring_only": True,
        }


class _LivePregraspRunner:
    """One-shot bimanual position move used by the remote browser preview."""

    def __init__(
        self,
        env,
        *,
        height_m: float,
        grasp_depth_m: float,
        steps: int,
        initial_state: str,
        settle_steps: int,
        wrist6_limit_test: bool,
        wrist6_test_steps: int,
        initial_state_prepared: bool = False,
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
        self.q6_joint_ids, self.q6_joint_names = self.robot.find_joints(
            ("zarm_l6_joint", "zarm_r6_joint"), preserve_order=True
        )
        if len(self.q6_joint_ids) != 2:
            raise RuntimeError("Could not resolve both Kuavo hand-pitch q6 joints.")
        self.q6_local_indices = []
        for term, q6_joint_id in zip(self.arm_terms, self.q6_joint_ids):
            term_joint_ids = [int(joint_id) for joint_id in term._joint_ids]
            self.q6_local_indices.append(term_joint_ids.index(int(q6_joint_id)))

        # A normal fallback may still load a named stationary pose here. The
        # web entrypoint preloads Task1 Ready before the browser connects.
        from kuavo_isaaclab_scene.robots.gripper_config import resolve_gripper_settings
        from kuavo_isaaclab_scene.robots.initial_states import apply_initial_state, load_initial_state
        from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model

        if not initial_state_prepared:
            state = load_initial_state(
                initial_state,
                robot_model=resolve_robot_model().name,
                gripper=resolve_gripper_settings().name,
            )
            apply_initial_state(env, None, state, initial_state)
            env.scene.write_data_to_sim()
        self.height_m = height_m
        self.grasp_depth_m = grasp_depth_m
        self.initial_state = initial_state
        self.wrist6_limit_test = bool(wrist6_limit_test)
        self.wrist6_test_steps = int(wrist6_test_steps)
        self.remaining = 0
        self.total = int(steps)
        self.settle_remaining = int(settle_steps)
        self.phase = "settle" if self.settle_remaining else "move"
        self.targets_w = None
        self.grasps_w = None
        self.pregrasp_orientations_w = None
        self.readygrasp_orientations_w = None
        self.geometry = None
        self.initial_q7 = None
        self.q6_start = None
        self.q6_targets = None
        self.action = None
        self.finished = False
        self.finalized = False
        self.hold_action = self._make_hold_action()
        if self.phase == "move":
            self._start_motion(initial_state)
        print(
            f"[PREGRASP] browser_connected initial_state={initial_state} "
            f"settle_steps={settle_steps}; motion_steps={self.total}; "
            f"wrist6_limit_test={self.wrist6_limit_test}",
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
        self.targets_w, self.grasps_w, inward_normals_w, self.geometry = _pregrasp_targets(
            self.env, height_m=self.height_m, grasp_depth_m=self.grasp_depth_m
        )
        self.initial_q7 = self.robot.data.joint_pos[0, [
            self.robot.find_joints("zarm_l7_joint", preserve_order=True)[0][0],
            self.robot.find_joints("zarm_r7_joint", preserve_order=True)[0][0],
        ]].clone()
        self.pregrasp_orientations_w = self.robot.data.body_link_quat_w[0, self.ee_ids].clone()
        grasp_directions = torch.nn.functional.normalize(self.grasps_w - self.targets_w, dim=-1)
        self.readygrasp_orientations_w = _orientation_from_closing_and_forward(
            inward_normals_w, grasp_directions
        )
        self.action = _absolute_pose_action(
            self.env, self.robot, self.targets_w, self.pregrasp_orientations_w
        )
        # q6 is already bent in the initial state. During motion it remains
        # inside the mirrored 45--75 degree task band; there is no bend phase.
        _task_q6_constraints(self.robot, self.arm_terms)
        for term in self.arm_terms:
            # Camera-forward is an initial-state contract, not a hard
            # orientation constraint throughout the Cartesian approach.
            term.orientation_weight = 0.20
            term.set_following(True)
        self.remaining = self.total
        self.phase = "move"
        print(
            f"[PREGRASP] targets_ready initial_state={initial_state} "
            f"targets_w={self.geometry} orientation_contract="
            "initial_camera_body_+X=robot_base_+X,q6_bend_band=45_to_75_deg",
            flush=True,
        )

    def _start_wrist6_motion(self):
        current = self.robot.data.joint_pos[0, self.q6_joint_ids].clone()
        limits = self.robot.data.joint_pos_limits[0, self.q6_joint_ids]
        targets = torch.where(current < 0.0, limits[:, 0], limits[:, 1])
        self.q6_start = current
        self.q6_targets = targets
        self.action = _absolute_pose_action(
            self.env, self.robot, self.grasps_w, self.readygrasp_orientations_w
        )
        for index, term in enumerate(self.arm_terms):
            posture = self.robot.data.joint_pos[:, term._joint_ids].clone()
            posture[:, self.q6_local_indices[index]] = targets[index]
            term.set_posture_target(
                posture,
                weight=0.60,
                direct_indices=(self.q6_local_indices[index],),
                direct_gain=2.0,
            )
            term.orientation_weight = 0.35
            term.set_following(True)
        self.remaining = self.wrist6_test_steps
        self.phase = "grasp_approach"
        print(
            f"[WRIST6] names={self.q6_joint_names} "
            f"start_rad={current.detach().cpu().tolist()} "
            f"target_rad={targets.detach().cpu().tolist()} "
            f"limits_rad={limits.detach().cpu().tolist()} steps={self.remaining} "
            "motion=pregrasp_to_live_grasp closing_axis=flap_normal forward_axis=grasp_direction",
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
        # Follow a moving flap in its own frame, but freeze the last few
        # approach steps so contact cannot create a chase/oscillation loop.
        if self.phase in ("move", "grasp_approach") and self.remaining > 12:
            live_pregrasp, live_grasps, live_normals, _ = _pregrasp_targets(
                self.env, height_m=self.height_m, grasp_depth_m=self.grasp_depth_m
            )
            if self.phase == "move":
                self.targets_w = live_pregrasp
                positions = self.targets_w
            else:
                self.grasps_w = live_grasps
                positions = self.grasps_w
                grasp_directions = torch.nn.functional.normalize(
                    live_grasps - live_pregrasp, dim=-1
                )
                self.readygrasp_orientations_w = _orientation_from_closing_and_forward(
                    live_normals, grasp_directions
                )
                self.pregrasp_orientations_w = self.readygrasp_orientations_w
            self.action = _absolute_pose_action(
                self.env, self.robot, positions, self.pregrasp_orientations_w
            )
        action = self.action
        self.remaining -= 1
        if self.remaining <= 0:
            if self.phase == "move" and self.wrist6_limit_test:
                self._start_wrist6_motion()
            else:
                self.finished = True
        return action

    def finish(self):
        if not self.finished or self.finalized:
            return
        if self.targets_w is None or self.initial_q7 is None:
            return
        final_pos = self.robot.data.body_link_pos_w[0, self.ee_ids]
        final_targets = self.grasps_w if self.q6_start is not None else self.targets_w
        errors = (final_targets - final_pos).norm(dim=-1)
        joint_ids = [
            self.robot.find_joints("zarm_l7_joint", preserve_order=True)[0][0],
            self.robot.find_joints("zarm_r7_joint", preserve_order=True)[0][0],
        ]
        final_q7 = self.robot.data.joint_pos[0, joint_ids]
        print(
            f"[PREGRASP] done final_target={'grasp' if self.q6_start is not None else 'pregrasp'} "
            f"position_error_m={errors.detach().cpu().tolist()} "
            f"q7_start={self.initial_q7.detach().cpu().tolist()} "
            f"q7_end={final_q7.detach().cpu().tolist()}",
            flush=True,
        )
        if self.q6_start is not None and self.q6_targets is not None:
            final_q6 = self.robot.data.joint_pos[0, self.q6_joint_ids]
            print(
                f"[WRIST6] done start_rad={self.q6_start.detach().cpu().tolist()} "
                f"target_rad={self.q6_targets.detach().cpu().tolist()} "
                f"end_rad={final_q6.detach().cpu().tolist()}",
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


def _write_planning_snapshot(env, pose_editor: _JointPoseEditor, output_dir: Path) -> None:
    """Persist the exact live collision world before any editor motion."""
    from kuavo_isaaclab_scene.planning.world import snapshot_colliders, world_config

    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Planning snapshot output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    robot = env.scene["robot"]
    root_pose_w = robot.data.root_pose_w[0].detach().cpu().tolist()
    collision_snapshot = snapshot_colliders(env.sim.stage, "/World/envs/env_0/Kuavo")
    runtime = {
        "joint_names": list(robot.joint_names),
        "joint_positions": robot.data.joint_pos[0].detach().cpu().tolist(),
        "joint_limits": robot.data.joint_pos_limits[0].detach().cpu().tolist(),
        "body_names": list(robot.body_names),
        "body_poses_w": robot.data.body_link_pose_w[0].detach().cpu().tolist(),
        "root_pose_w": root_pose_w,
        "pose_editor_state": pose_editor.state(),
    }
    world = world_config(collision_snapshot, root_pose_w)
    for name, value in (
        ("collision_snapshot.json", collision_snapshot),
        ("runtime.json", runtime),
        ("world.json", world),
    ):
        (output_dir / name).write_text(
            json.dumps(value, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    print(f"[PLANNING_SNAPSHOT] wrote live scene to {output_dir}", flush=True)


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
    cfg.scene.joint_editor_camera.width = args_cli.editor_camera_width
    cfg.scene.joint_editor_camera.height = args_cli.editor_camera_height
    half_baseline = args_cli.stereo_eye_separation * 0.5
    cfg.scene.xr_left_eye_camera.offset.pos = (0.08, half_baseline, 0.0)
    cfg.scene.xr_right_eye_camera.offset.pos = (0.08, -half_baseline, 0.0)
    if args_cli.joint_pose_editor:
        # The editor has no XR tracking or policy observations.  Rendering
        # only its third-person camera avoids needlessly paying for five
        # additional RGB sensors on the shared preview GPU.
        cfg.scene.xr_left_eye_camera = None
        cfg.scene.xr_right_eye_camera = None
        cfg.scene.robustness_camera = None
        cfg.scene.waist_camera = None
        cfg.scene.left_wrist_camera = None
        cfg.scene.right_wrist_camera = None
    else:
        cfg.scene.joint_editor_camera = None
    set_domain_randomization(cfg, args_cli.domain_randomization)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=args_cli.seed)
    if args_cli.pregrasp:
        _load_task_ready(env, args_cli.pregrasp_initial_state)
        if args_cli.solve_downward_ready:
            _solve_downward_ready(env)
    elif args_cli.joint_pose_editor:
        print(
            "[POSE_EDITOR_INIT] using KuavoQuestTeleopEnvCfg default initial pose",
            flush=True,
        )
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
    pose_editor = (
        _JointPoseEditor(
            env,
            pregrasp_height_m=args_cli.pregrasp_height_m,
            grasp_depth_m=args_cli.pregrasp_grasp_depth_m,
        )
        if args_cli.joint_pose_editor
        else None
    )
    if args_cli.planning_snapshot_output is not None:
        _write_planning_snapshot(env, pose_editor, args_cli.planning_snapshot_output)
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
    if pose_editor is not None:
        print("[POSE_EDITOR] authoring-only mode; dataset recording and pregrasp are disabled.")
        print("[POSE_EDITOR] Open data_collection/task1_pose_editor.html through the HTTP server.")
    print("[CONTROL] In Chrome/IWER, move the HMD and left/right controllers.")
    print("[CONTROL] The first tracked frame calibrates; subsequent motion drives Kuavo head and arms.")
    print("[CONTROL] Left stick=base forward/strafe; right stick=turn/torso lift; index triggers=grippers.")
    print("[CONTROL] Body motion requires both tracked controllers and head; tracking loss stops the base and holds torso height.")
    if args_cli.headless:
        print("[VIEW] Headless server mode: browser XR stream enabled; local Isaac viewports disabled.")
    else:
        print("[VIEW] Browser XR: stereo Isaac scene with small left/right wrist panels.")
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
            editor_changed = False
            if pose_editor is not None:
                command = bridge.latest_pose_editor_command(pose_editor.last_command_sequence)
                editor_changed = pose_editor.accept(command)
                for term in arm_terms:
                    term.set_following(False)
                action = torch.zeros(
                    (1, env.action_manager.total_action_dim),
                    device=env.device,
                    dtype=robot.data.joint_pos.dtype,
                )
            elif args_cli.pregrasp:
                if pregrasp_runner is None and bridge.client_count > 0:
                    print("[PREGRASP] browser client detected; starting live sequence", flush=True)
                    pregrasp_runner = _LivePregraspRunner(
                        env,
                        height_m=args_cli.pregrasp_height_m,
                        grasp_depth_m=args_cli.pregrasp_grasp_depth_m,
                        steps=args_cli.pregrasp_steps,
                        initial_state=args_cli.pregrasp_initial_state,
                        settle_steps=args_cli.pregrasp_settle_steps,
                        wrist6_limit_test=args_cli.wrist6_limit_test,
                        wrist6_test_steps=args_cli.wrist6_test_steps,
                        initial_state_prepared=True,
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
            if pose_editor is not None:
                # Re-assert static joint targets after the manager action and
                # refresh the visible joint-axis cylinders.
                pose_editor.robot.set_joint_position_target(
                    pose_editor.targets, joint_ids=pose_editor.joint_ids
                )
                pose_editor.update_markers()
                if editor_changed or step % max(1, stream_interval) == 0:
                    bridge.publish_pose_editor_state(pose_editor.state())
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
                if pose_editor is not None:
                    composite = _camera_rgb(env.scene["joint_editor_camera"])
                else:
                    composite = compose_stereo_atlas(
                        _camera_rgb(env.scene["xr_left_eye_camera"]),
                        _camera_rgb(env.scene["xr_right_eye_camera"]),
                        None,
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
