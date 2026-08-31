#!/usr/bin/env python3
"""Collect Kuavo bimanual demonstrations from Quest/OpenXR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys

from isaaclab.app import AppLauncher
from .gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
    gripper_teleop_action,
    resolve_gripper_settings,
    teleop_action_names,
)
from .robot_model import add_robot_model_cli_args, export_robot_model_cli, resolve_robot_model
from .teleop_recorder import new_session_path


parser = argparse.ArgumentParser(description="Collect Kuavo Quest hand-tracking demonstrations.")
parser.add_argument("--input-mode", choices=("controllers", "hands"), default="controllers",
                    help="Arm input: controller grip pose + trigger (default), or bare-hand wrist + pinch.")
parser.add_argument("--profile-steps", type=int, default=0, help="Profile this many control steps after startup (0 disables).")
parser.add_argument("--capture-xr", action="store_true", help="Save an XR display diagnostic capture after 60 frames.")
parser.add_argument("--render-quality", choices=("performance", "quality"), default="performance")
parser.add_argument("--xr-resolution-scale", type=float, default=1.0,
                    help="XR render-buffer scale (0.1–2.0); lower values reduce sharpness, not material quality.")
parser.add_argument("--controller-mapping", choices=("absolute", "relative"), default="absolute",
                    help="Absolute aligns robot tools to VR grip positions; relative enables the legacy delta mapping.")
parser.add_argument("--arm-stiffness", type=float, default=800.0)
parser.add_argument("--arm-damping", type=float, default=50.0)
parser.add_argument("--arm-orientation-weight", type=float, default=0.5,
                    help="Rotation weight in pose IK; 0 disables controller rotation, 0.5 balances position and orientation.")
parser.add_argument("--control-hz", type=int, choices=(30, 60), default=60,
                    help="Simulation control timestep; actual wall-clock rate is printed as [PERF].")
parser.add_argument("--scene-detail", choices=("full", "compact"), default="compact",
                    help="Compact removes background warehouse props and unused legacy task bodies; preserves materials.")
parser.add_argument("--desktop-render", action=argparse.BooleanOptionalAction, default=False,
                    help="Render the full desktop viewport; otherwise keep only a tiny render while camera frames are needed.")
parser.add_argument(
    "--dataset-format",
    choices=("hdf5", "lerobot", "both"),
    default="hdf5",
    help="Recording backend. 'both' writes HDF5 and LeRobot in parallel.",
)
parser.add_argument(
    "--dataset",
    type=Path,
    default=None,
    help="New HDF5 file to create. Default: a unique timestamped file in datasets/. Existing files are never reused.",
)
parser.add_argument(
    "--lerobot-root",
    type=Path,
    default=Path("datasets/kuavo_quest_lerobot"),
    help="Local LeRobot dataset root (created or resumed).",
)
parser.add_argument(
    "--lerobot-python",
    type=Path,
    default=Path(os.environ["LEROBOT_PYTHON"]) if os.environ.get("LEROBOT_PYTHON") else None,
    help="Isolated Python containing a LeRobot Dataset v3 writer (or set LEROBOT_PYTHON).",
)
parser.add_argument(
    "--lerobot-repo-id",
    default="local/kuavo_quest_teleop",
    help="LeRobot/Hugging Face repository identifier stored in dataset metadata.",
)
parser.add_argument(
    "--lerobot-task",
    default="Move all open boxes from the rack to free conveyor spaces, then press the green button.",
    help="Natural-language task attached to every LeRobot frame.",
)
parser.add_argument(
    "--lerobot-fps",
    type=int,
    default=0,
    help="Dataset FPS; 0 derives it from the Isaac Lab control timestep.",
)
parser.add_argument(
    "--lerobot-use-videos",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Encode LeRobot camera streams as MP4; disable to retain individual images.",
)
parser.add_argument(
    "--lerobot-save-failed",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Keep failed/stopped LeRobot episodes. By default only successful episodes are saved.",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--max-episodes", type=int, default=0, help="0 (default) keeps the application open between attempts.")
parser.add_argument("--episode-seconds", type=float, default=0.0, help="Episode timeout in simulation seconds; 0 disables it.")
parser.add_argument(
    "--auto-start",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Start recording automatically after both selected input devices become valid.",
)
parser.add_argument(
    "--domain-randomization",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Enable material, lighting, physics, obstacle and cargo randomization.",
)
parser.add_argument(
    "--camera-preview",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Open Kuavo head/wrist camera windows in the desktop Isaac Sim UI.",
)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--head-camera-height", type=int, default=360)
parser.add_argument("--wrist-camera-width", type=int, default=240)
parser.add_argument("--wrist-camera-height", type=int, default=180)
parser.add_argument(
    "--wrist-cameras",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Create wrist camera sensors. Disabling also disables their Quest panels and recording.",
)
parser.add_argument(
    "--quest-camera-overlay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show small wrist-camera panels at the upper left/right of the Quest view.",
)
parser.add_argument("--xr-overlay-distance", type=float, default=0.35, metavar="METERS")
parser.add_argument(
    "--xr-overlay-forward-axis",
    choices=("-z", "+z"),
    default="-z",
    help="Direction in front of the OpenXR head frame; use +z only if the panel appears behind you.",
)
parser.add_argument(
    "--record-depth",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Store head-camera depth in addition to RGB.",
)
parser.add_argument(
    "--record-wrist-cameras",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Store left/right wrist RGB frames in the selected dataset backend(s).",
)
parser.add_argument("--position-gain", type=float, default=1.5)
parser.add_argument("--rotation-gain", type=float, default=1.0)
parser.add_argument(
    "--xr-runtime-json",
    type=Path,
    default=None,
    help="Path to CloudXR Runtime's openxr_cloudxr.json (sets XR_RUNTIME_JSON).",
)
parser.add_argument(
    "--rack-boxes",
    type=str,
    default=None,
    metavar="SPEC",
    help="Rack contents, e.g. '1:small*2;2:medium,large;3:xlarge'.",
)
parser.add_argument("--rack-box-layout", type=Path, default=None, metavar="JSON")
parser.add_argument("--rack-box-poses", type=Path, default=None, metavar="JSON")
parser.add_argument("--ignore-captured-box-poses", action="store_true")
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args_cli = parser.parse_args()
if not args_cli.wrist_cameras:
    args_cli.quest_camera_overlay = False
    args_cli.record_wrist_cameras = False
if args_cli.max_episodes < 0 or args_cli.episode_seconds < 0:
    parser.error("Episode count and timeout must be non-negative (0 means unlimited).")
if not 0.1 <= args_cli.xr_resolution_scale <= 2.0:
    parser.error("--xr-resolution-scale must be between 0.1 and 2.0.")
if args_cli.arm_stiffness <= 0 or args_cli.arm_damping < 0 or not 0 <= args_cli.arm_orientation_weight <= 1:
    parser.error("Arm stiffness must be positive; damping non-negative; orientation weight between 0 and 1.")
if args_cli.profile_steps < 0:
    parser.error("--profile-steps must be non-negative.")
if args_cli.profile_steps or args_cli.capture_xr:
    Path("artifacts").mkdir(exist_ok=True)
args_cli.dataset = (args_cli.dataset or new_session_path()).expanduser().resolve()
if args_cli.dataset_format in {"hdf5", "both"} and args_cli.dataset.exists():
    parser.error(
        f"HDF5 file already exists: {args_cli.dataset}. Existing sessions are never overwritten or appended to. "
        "Omit --dataset for a new session file, or choose a different filename."
    )
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)
try:
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))

if min(
    args_cli.head_camera_width,
    args_cli.head_camera_height,
    args_cli.wrist_camera_width,
    args_cli.wrist_camera_height,
) <= 0:
    parser.error("Head and wrist camera width/height values must be positive.")
if args_cli.xr_overlay_distance <= 0.08:
    parser.error("--xr-overlay-distance must be greater than the 0.08 m XR near plane.")
if args_cli.max_episodes < 0:
    parser.error("--max-episodes must be 0 or greater.")
if args_cli.lerobot_fps < 0:
    parser.error("--lerobot-fps must be 0 or greater.")
if not args_cli.lerobot_repo_id.strip() or not args_cli.lerobot_task.strip():
    parser.error("--lerobot-repo-id and --lerobot-task must not be empty.")
if args_cli.dataset_format in {"lerobot", "both"} and args_cli.lerobot_python is None:
    parser.error(
        "LeRobot recording requires --lerobot-python /path/to/python or the LEROBOT_PYTHON environment variable."
    )
if args_cli.xr_runtime_json is not None:
    runtime_json = args_cli.xr_runtime_json.expanduser().resolve()
    if not runtime_json.is_file():
        parser.error(f"OpenXR runtime manifest does not exist: {runtime_json}")
    os.environ["XR_RUNTIME_JSON"] = str(runtime_json)
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

# Hand tracking requires the Isaac Lab OpenXR experience. RTX cameras are
# intentionally retained because the real Kuavo head camera is part of data.
args_cli.xr = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import time

import numpy as np
import torch

from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention

from .camera_viewports import open_camera_viewports
from .teleop_camera import camera_rgb as _camera_rgb, camera_depth as _camera_depth
from .manager_env import LOCAL_BOX_SCENE_KEYS
from .quest_openxr import RawQuestOpenXRDevice, start_quest_xr_session
from .teleop_env import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    KuavoQuestTeleopEnvCfg,
    set_domain_randomization,
)
from .teleop_mapping import AbsoluteControllerMapper, BimanualTeleopMapper, TeleopMappingCfg
from .teleop_body import BODY_ACTION_NAMES, BODY_JOINTS, TeleopBodyMapper, controller_axis
from .teleop_scene import configure_scene_detail
from .paths import ASSET_DIR
from .teleop_lerobot_recorder import LeRobotTeleopRecorder
from .teleop_recorder import TeleopHdf5EpisodeRecorder, TeleopRecorderGroup
from .xr_camera_overlay import QuestCameraOverlay, QuestCameraOverlayCfg


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _pose_or_default(pose: np.ndarray | None) -> np.ndarray:
    if pose is None:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return np.asarray(pose, dtype=np.float32)


def _hand_array(hand: dict[str, np.ndarray] | None) -> np.ndarray:
    hand = hand or {}
    return np.stack([_pose_or_default(hand.get(name)) for name in HAND_JOINT_NAMES])


def _scene_asset_or_none(scene, name: str):
    try:
        return scene[name]
    except KeyError:
        return None


def main() -> None:
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.sim.device = args_cli.device
    cfg.teleop_devices.devices["quest_handtracking"].sim_device = cfg.sim.device
    cfg.scene.robot.actuators["arms"].stiffness = args_cli.arm_stiffness
    cfg.scene.robot.actuators["arms"].damping = args_cli.arm_damping
    cfg.decimation = 120 // args_cli.control_hz
    cfg.sim.render_interval = cfg.decimation
    absolute_control = args_cli.input_mode == "controllers" and args_cli.controller_mapping == "absolute"
    arm_action_size = 14 if absolute_control else 12
    if absolute_control:
        cfg.actions.left_arm.controller.use_relative_mode = False
        cfg.actions.right_arm.controller.use_relative_mode = False
    cfg.seed = args_cli.seed
    cfg.scene.robustness_camera.width = args_cli.head_camera_width
    cfg.scene.robustness_camera.height = args_cli.head_camera_height
    cfg.scene.left_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.left_wrist_camera.height = args_cli.wrist_camera_height
    cfg.scene.right_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.right_wrist_camera.height = args_cli.wrist_camera_height
    cfg.sim.render.antialiasing_mode = "DLSS"
    if args_cli.render_quality == "performance":
        cfg.sim.render.dlss_mode = 0
        cfg.sim.render.enable_reflections = False
        cfg.sim.render.enable_translucency = False
        cfg.sim.render.enable_global_illumination = False
        cfg.sim.render.enable_ambient_occlusion = False
        cfg.sim.render.samples_per_pixel = 1
    else:
        cfg.sim.render.dlss_mode = 2
        cfg.sim.render.enable_reflections = True
        cfg.sim.render.enable_translucency = True
        cfg.sim.render.enable_global_illumination = True
        cfg.sim.render.enable_ambient_occlusion = True
        cfg.sim.render.samples_per_pixel = 2
    # Wrist depth is never recorded. Do not render/copy unused depth buffers.
    cfg.scene.left_wrist_camera.data_types = ["rgb"]
    cfg.scene.right_wrist_camera.data_types = ["rgb"]
    if not args_cli.wrist_cameras:
        cfg.scene.left_wrist_camera = None
        cfg.scene.right_wrist_camera = None
    if not args_cli.record_depth:
        cfg.scene.robustness_camera.data_types = ["rgb"]
    set_domain_randomization(cfg, args_cli.domain_randomization)
    configure_scene_detail(cfg, args_cli.scene_detail)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=args_cli.seed)
    print(f"[CAMERA] Wrist sensors={'ON' if args_cli.wrist_cameras else 'OFF (not created)'}; "
          f"head={args_cli.head_camera_width}x{args_cli.head_camera_height}; "
          f"depth={'ON' if args_cli.record_depth else 'OFF'}", flush=True)
    if args_cli.camera_preview:
        open_camera_viewports(
            env.scene,
            (["robustness_camera", "left_wrist_camera", "right_wrist_camera"]
             if args_cli.wrist_cameras else ["robustness_camera"]),
            headless=args_cli.headless,
            width=240,
            height=180,
            columns=3,
        )

    robot = env.scene["robot"]
    head_body_id = robot.find_bodies(resolve_robot_model().head_camera_body)[0][0]
    camera_offset = cfg.scene.robustness_camera.offset
    camera_offset_pos = torch.tensor([camera_offset.pos], device=env.device)
    camera_offset_quat = convert_camera_frame_orientation_convention(
        torch.tensor([camera_offset.rot], device=env.device), origin=camera_offset.convention, target="opengl"
    )

    def head_camera_pose():
        pos, quat = combine_frame_transforms(
            robot.data.body_pos_w[:, head_body_id], robot.data.body_quat_w[:, head_body_id],
            camera_offset_pos, camera_offset_quat,
        )
        return _to_numpy(pos[0]), _to_numpy(quat[0])
    arm_joint_ids, arm_joint_names = robot.find_joints(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS + BODY_JOINTS)
    state_names = list(arm_joint_names)
    gripper_state_sources = []
    for side in GRIPPER_SETTINGS.active_sides:
        gripper = env.scene[GRIPPER_SETTINGS.asset_name_for(side)]
        gripper_joint_ids, gripper_joint_names = gripper.find_joints(
            list(GRIPPER_SETTINGS.joint_names_for(side))
        )
        gripper_state_sources.append((side, gripper, gripper_joint_ids))
        state_names.extend(
            gripper_joint_names
            if GRIPPER_SETTINGS.integrated
            else (f"{side}_{name}" for name in gripper_joint_names)
        )
    action_names = teleop_action_names(GRIPPER_SETTINGS)
    if absolute_control:
        action_names = tuple(f"{side}_{axis}_base" for side in ("left", "right")
                             for axis in ("x", "y", "z", "qw", "qx", "qy", "qz")) + action_names[12:]
    left_body_ids, _ = robot.find_bodies("zarm_l7_end_effector")
    right_body_ids, _ = robot.find_bodies("zarm_r7_end_effector")
    if len(left_body_ids) != 1 or len(right_body_ids) != 1:
        raise RuntimeError("Could not resolve both Kuavo end-effector bodies.")
    box_names = [name for name in LOCAL_BOX_SCENE_KEYS if _scene_asset_or_none(env.scene, name) is not None]
    button = _scene_asset_or_none(env.scene, "button_station")
    button_joint_count = int(button.data.joint_pos.shape[-1]) if button is not None else 0

    device_cfg = cfg.teleop_devices.devices["quest_handtracking"]
    # Isaac Lab v2.3 feature-gates raw OpenXR queries by retargeter
    # requirements. Kuavo uses its own safety mapper, so the adapter requests
    # hand/head tracking while retaining the raw upstream dictionary format.
    xr_device = RawQuestOpenXRDevice(device_cfg, input_mode=args_cli.input_mode)
    try:
        start_quest_xr_session(simulation_app, enable_ui=args_cli.quest_camera_overlay,
                               resolution_scale=args_cli.xr_resolution_scale,
                               render_quality=args_cli.render_quality)
    except Exception:
        env.close()
        raise
    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device=env.device))
    desktop_viewport = None
    if not args_cli.desktop_render:
        from omni.kit.viewport.utility import get_active_viewport
        desktop_viewport = get_active_viewport()
        if desktop_viewport is not None:
            desktop_viewport.fill_frame = False
            desktop_viewport.resolution = (160, 90)
    mapper = BimanualTeleopMapper(
        TeleopMappingCfg(
            position_gain=args_cli.position_gain,
            rotation_gain=args_cli.rotation_gain,
        )
    )
    absolute_mapper = AbsoluteControllerMapper()
    urdf = (ASSET_DIR / "kuavo_s200062/urdf/biped_s200062.urdf" if args_cli.robot_model == "s200062"
            else ASSET_DIR / "kuavo_s63/urdf/kuavo_s63.urdf")
    body_mapper = TeleopBodyMapper(urdf)

    quest_overlay = None
    if args_cli.quest_camera_overlay:
        try:
            quest_overlay = QuestCameraOverlay(
                head_resolution=(args_cli.head_camera_width, args_cli.head_camera_height),
                wrist_resolution=(args_cli.wrist_camera_width, args_cli.wrist_camera_height),
                cfg=QuestCameraOverlayCfg(
                    distance_m=args_cli.xr_overlay_distance,
                    forward_axis=args_cli.xr_overlay_forward_axis,
                ),
            )
        except Exception:
            env.close()
            raise

    requests = {
        key: False for key in ("start", "stop", "reset", "success", "toggle", "calibrate", "preview", "overlay")
    }
    xr_device.add_callback("START", lambda: requests.__setitem__("start", True))
    xr_device.add_callback("STOP", lambda: requests.__setitem__("stop", True))
    xr_device.add_callback("RESET", lambda: requests.__setitem__("reset", True))
    keyboard.add_callback("P", lambda: requests.__setitem__("toggle", True))
    keyboard.add_callback("R", lambda: requests.__setitem__("reset", True))
    keyboard.add_callback("M", lambda: requests.__setitem__("success", True))
    keyboard.add_callback("C", lambda: requests.__setitem__("calibrate", True))
    keyboard.add_callback("T", lambda: requests.__setitem__("preview", True))
    keyboard.add_callback("H", lambda: requests.__setitem__("overlay", True))
    xr_device.bind_button("left", "x", lambda: requests.__setitem__("calibrate", True))
    xr_device.bind_button("right", "a", lambda: requests.__setitem__("preview", True))
    xr_device.bind_button("right", "b", lambda: requests.__setitem__("toggle", True))
    xr_device.bind_button("left", "y", lambda: requests.__setitem__("overlay", True))

    dataset_path = args_cli.dataset.expanduser().resolve()
    lerobot_root = args_cli.lerobot_root.expanduser().resolve()
    recorders = {}
    dataset_descriptions = []
    if args_cli.dataset_format in {"hdf5", "both"}:
        recorders["hdf5"] = TeleopHdf5EpisodeRecorder(dataset_path)
        dataset_descriptions.append(f"HDF5 first file={dataset_path}; subsequent attempts get new files in {dataset_path.parent}")
    if args_cli.dataset_format in {"lerobot", "both"}:
        control_fps = 1.0 / float(env.step_dt)
        lerobot_fps = args_cli.lerobot_fps or int(round(control_fps))
        if abs(control_fps - lerobot_fps) > 1.0e-4:
            print(
                f"[WARN] Control rate {control_fps:.6f} Hz is not integer; "
                f"LeRobot timestamps will use {lerobot_fps} FPS."
            )
        recorders["lerobot"] = LeRobotTeleopRecorder(
            lerobot_root,
            repo_id=args_cli.lerobot_repo_id,
            fps=lerobot_fps,
            task=args_cli.lerobot_task,
            joint_names=state_names,
            hand_joint_names=HAND_JOINT_NAMES,
            head_resolution=(args_cli.head_camera_width, args_cli.head_camera_height),
            wrist_resolution=(args_cli.wrist_camera_width, args_cli.wrist_camera_height),
            box_count=len(box_names),
            button_joint_count=button_joint_count,
            record_wrist_cameras=args_cli.record_wrist_cameras,
            use_videos=args_cli.lerobot_use_videos,
            save_failed=args_cli.lerobot_save_failed,
            writer_python=args_cli.lerobot_python,
            action_names=action_names + BODY_ACTION_NAMES,
            record_controllers=args_cli.input_mode == "controllers",
        )
        lerobot_recorder = recorders["lerobot"]
        dataset_descriptions.append(
            f"LeRobot={lerobot_root} (repo_id={args_cli.lerobot_repo_id}, fps={lerobot_fps}, "
            f"lerobot={lerobot_recorder.lerobot_version}, format={lerobot_recorder.dataset_version})"
        )
    recorder = TeleopRecorderGroup(recorders)
    completed_this_run = 0
    episode_steps = 0
    pending_start = False
    manual_pause = False
    start_wall_time = time.perf_counter()
    last_tracking_state: tuple[bool, bool, bool] | None = None
    preview_enabled = False
    camera_reported = False
    camera_wait_reported = False
    held_absolute_targets = np.zeros(len(action_names) - arm_action_size, dtype=np.float32)
    held_absolute_targets[2:] = 1.0  # Binary gripper actions: nonnegative=open, negative=close.
    free_view = False
    view_initialized = False
    waist_yaw_id = robot.find_joints("waist_yaw_joint")[0][0]
    last_waist_yaw = float(robot.data.joint_pos[0, waist_yaw_id])
    arm_terms = [env.action_manager.get_term(name) for name in ("left_arm", "right_arm")]
    for term in arm_terms:
        term.orientation_weight = args_cli.arm_orientation_weight
    last_motion_report = time.perf_counter()
    last_ee_positions = _to_numpy(robot.data.body_pos_w[0, [left_body_ids[0], right_body_ids[0]]]).copy()

    def hold_arms():
        for term in arm_terms:
            term.hold_current_pose()

    def reset_simulation() -> None:
        nonlocal episode_steps, free_view, view_initialized, last_waist_yaw
        env.reset()
        xr_device.reset()
        mapper.reset()
        absolute_mapper.reset()
        body_mapper.reset()
        held_absolute_targets.fill(0.0)
        held_absolute_targets[2:] = 1.0
        free_view = False
        view_initialized = False
        last_waist_yaw = float(robot.data.joint_pos[0, waist_yaw_id])
        episode_steps = 0

    def finish_episode(success: bool, reason: str) -> None:
        nonlocal completed_this_run
        name = recorder.finish_episode(success=success, reason=reason)
        if name is not None:
            completed_this_run += 1
            print(f"[DATA] Finished {name}: success={success}, reason={reason}")
        hold_arms()
        mapper.reset(head_target=held_absolute_targets[:2])
        print("[CONTROL] Recording closed. Scene stays open; A=follow, B=new recording, R=reset.")

    print("[INFO] Quest/OpenXR Kuavo teleoperation is ready.")
    print(f"[INFO] Dataset: {'; '.join(dataset_descriptions)}")
    print(f"[INFO] Arm input mode: {args_cli.input_mode}; "
          + ("controller grip pose + index trigger" if args_cli.input_mode == "controllers" else "bare-hand wrist + pinch"))
    if args_cli.dataset_format in {"lerobot", "both"} and not args_cli.lerobot_save_failed:
        print("[INFO] LeRobot keeps successful episodes only; STOP/RESET/time-limit attempts are discarded.")
    if quest_overlay is not None:
        print("[INFO] Quest view: scene with small wrist-camera panels at the upper left/right.")
    print("[CONTROL] Quest START/STOP/RESET or desktop P=start/stop, R=reset, M=finish as success.")
    print("[CONTROL] C=recenter/calibrate, T=motion preview without recording, H=camera overlay on/off.")
    print("[CONTROL] Quest controllers: X=calibrate, A=motion start/stop, B=record start/stop, Y=panels on/off.")
    print("[CONTROL] Left stick=base forward/strafe; right stick=waist yaw/lift; left squeeze=hold free view. "
          "Release squeeze to return to robot head. Index triggers: released=open, pressed=close.")
    print("[CONTROL] Arm motion is paused until P starts recording or T enables motion preview.")
    print(f"[CONTROL] Arm mapping: {'absolute VR grip position, calibrated tool orientation' if absolute_control else 'persistent relative pose'}; "
          f"orientation weight={args_cli.arm_orientation_weight:.2f} (0=position only). "
          "While following, the last valid goal is retained on tracking loss; A explicitly stops motion.")
    print(f"[CONTROL] Session limit: {args_cli.max_episodes or 'unlimited'} episodes; "
          f"timeout: {args_cli.episode_seconds or 'disabled'}. Each attempt gets a separate HDF5 file.")
    if quest_overlay is not None:
        print(f"[VIEW] Wrist camera panels: {args_cli.xr_overlay_distance:.2f} m in front of the headset.")
    print(
        f"[GRIPPER] preset={GRIPPER_SETTINGS.name}, sides={GRIPPER_SETTINGS.active_sides or 'none'}, "
        f"pinch close threshold={GRIPPER_SETTINGS.pinch_close_threshold_m:.3f} m."
    )

    stop_requested = False

    def request_shutdown(signum, frame) -> None:
        nonlocal stop_requested
        stop_requested = True

    # Kit's default SIGINT handler can exit before HDF5 cleanup. Finish the
    # current frame, then let the collector's finally block close the file.
    previous_handlers = {
        sig: signal.signal(sig, request_shutdown) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    profile = None
    profile_steps = 0
    report_steps = 0
    report_time = time.perf_counter()
    if args_cli.profile_steps:
        import cProfile
        profile = cProfile.Profile()
        profile.enable()
    try:
        while not stop_requested and simulation_app.is_running():
            if args_cli.max_episodes and completed_this_run >= args_cli.max_episodes:
                print(f"[CONTROL] Exiting because --max-episodes {args_cli.max_episodes} was reached.")
                break

            if requests["overlay"]:
                requests["overlay"] = False
                if quest_overlay is not None:
                    visible = quest_overlay.toggle_visible()
                    print(f"[VIEW] Camera overlay {'shown when frames are ready' if visible else 'hidden'}.")
                else:
                    print("[VIEW] Camera overlay was disabled by --no-quest-camera-overlay.")
            if requests["preview"]:
                requests["preview"] = False
                if recorder.recording:
                    preview_enabled = False
                    requests["stop"] = True
                    print("[CONTROL] Motion stopped; finishing the current recording.")
                else:
                    preview_enabled = not preview_enabled
                    pending_start = False
                    manual_pause = True
                    mapper.reset(head_target=held_absolute_targets[:2])
                    hold_arms()
                    print(f"[CONTROL] Motion preview {'ON' if preview_enabled else 'OFF'}; no samples are being recorded.")
            if requests["calibrate"]:
                requests["calibrate"] = False
                if recorder.recording:
                    finish_episode(False, "operator_calibration")
                if last_tracking_state is None or not last_tracking_state[2]:
                    print("[CALIBRATION] Head tracking is unavailable; wear the headset and reconnect first.")
                else:
                    preview_enabled = False
                    pending_start = False
                    manual_pause = True
                    xr_device.recenter_view(*head_camera_pose())
                    for _ in range(3):
                        simulation_app.update()
                    xr_device.reset()
                    mapper.reset(head_target=held_absolute_targets[:2])
                    absolute_mapper.reset()
                    hold_arms()
                    print("[CALIBRATION] View centered at Kuavo head camera; tool orientation calibrated from current controller orientation. "
                          "A starts following the controller's actual VR position. Motion preview OFF.")

            raw = xr_device.advance()
            if not view_initialized and raw.get(RawQuestOpenXRDevice.TrackingTarget.HEAD) is not None:
                xr_device.recenter_view(*head_camera_pose())
                for _ in range(3):
                    simulation_app.update()
                xr_device.reset()
                mapper.reset(head_target=held_absolute_targets[:2])
                absolute_mapper.reset()
                raw = xr_device.advance()
                view_initialized = True
                print("[VIEW] Initial viewpoint attached to robot head.", flush=True)
            left_hand = raw.get(RawQuestOpenXRDevice.TrackingTarget.HAND_LEFT)
            right_hand = raw.get(RawQuestOpenXRDevice.TrackingTarget.HAND_RIGHT)
            left_controller = raw.get(RawQuestOpenXRDevice.TrackingTarget.CONTROLLER_LEFT)
            right_controller = raw.get(RawQuestOpenXRDevice.TrackingTarget.CONTROLLER_RIGHT)
            head_pose = raw.get(RawQuestOpenXRDevice.TrackingTarget.HEAD)
            next_free_view = left_controller is not None and float(left_controller[1, 3]) >= .5
            if next_free_view != free_view:
                free_view = next_free_view
                if not free_view:
                    xr_device.recenter_view(*head_camera_pose())
                    # Apply the teleport before reading poses or pinning the
                    # next frame; otherwise the old orientation overwrites it.
                    for _ in range(3):
                        simulation_app.update()
                    raw = xr_device.advance()
                    left_hand = raw.get(RawQuestOpenXRDevice.TrackingTarget.HAND_LEFT)
                    right_hand = raw.get(RawQuestOpenXRDevice.TrackingTarget.HAND_RIGHT)
                    left_controller = raw.get(RawQuestOpenXRDevice.TrackingTarget.CONTROLLER_LEFT)
                    right_controller = raw.get(RawQuestOpenXRDevice.TrackingTarget.CONTROLLER_RIGHT)
                    head_pose = raw.get(RawQuestOpenXRDevice.TrackingTarget.HEAD)
                xr_device.reset_head_reference()
                mapper.reset(head_target=held_absolute_targets[:2])
                print(f"[VIEW] {'FREE: room-scale view; arm goals held' if free_view else 'HEAD: robot camera anchor'}", flush=True)
            motion_head = xr_device.motion_head_pose(head_pose)
            root_quat = _to_numpy(robot.data.root_quat_w[0])
            map_inputs = mapper.advance_controllers if args_cli.input_mode == "controllers" else mapper.advance
            left_input, right_input = ((left_controller, right_controller) if args_cli.input_mode == "controllers"
                                       else (left_hand, right_hand))
            mapped = map_inputs(left_input, right_input, motion_head, root_quat)
            tracking_state = (mapped.left_valid, mapped.right_valid, mapped.head_valid)
            if tracking_state != last_tracking_state:
                print(
                    "[TRACKING] "
                    f"left={tracking_state[0]}, right={tracking_state[1]}, head={tracking_state[2]} "
                    f"input={args_cli.input_mode}"
                )
                last_tracking_state = tracking_state

            if requests["toggle"]:
                requests["toggle"] = False
                if recorder.recording:
                    requests["stop"] = True
                elif pending_start:
                    pending_start = False
                    manual_pause = True
                    print("[CONTROL] Pending recording cancelled.")
                else:
                    requests["start"] = True
            if requests["reset"]:
                requests["reset"] = False
                preview_enabled = False
                if recorder.recording:
                    finish_episode(False, "operator_reset")
                reset_simulation()
                pending_start = False
                manual_pause = True
                continue
            if requests["success"]:
                requests["success"] = False
                preview_enabled = False
                if recorder.recording:
                    finish_episode(True, "operator_success")
                pending_start = False
                manual_pause = not args_cli.auto_start
                continue
            if requests["stop"]:
                requests["stop"] = False
                preview_enabled = False
                if recorder.recording:
                    finish_episode(False, "operator_stop")
                pending_start = False
                manual_pause = True
                continue
            if requests["start"]:
                requests["start"] = False
                pending_start = True
                manual_pause = False
                if not mapped.bimanual_valid:
                    print(f"[CONTROL] Waiting for both tracked {args_cli.input_mode} before recording; B/P cancels.")
            if args_cli.auto_start and not manual_pause and not recorder.recording:
                pending_start = True

            if pending_start and mapped.bimanual_valid and not recorder.recording:
                if not preview_enabled:
                    # Do not apply head/hand movement accumulated while paused
                    # when B/P starts recording directly (without motion preview).
                    mapper.reset(head_target=held_absolute_targets[:2])
                    mapped = map_inputs(left_input, right_input, motion_head, root_quat)
                episode_name = recorder.start_episode(
                    {
                        "seed": args_cli.seed,
                        "input_mode": args_cli.input_mode,
                        "arm_control": "absolute_controller_pose_v1" if absolute_control else "persistent_relative_pose_v1",
                        "arm_orientation_weight": args_cli.arm_orientation_weight,
                        "xr_resolution_scale": args_cli.xr_resolution_scale,
                        "render_quality": args_cli.render_quality,
                        "record_depth": args_cli.record_depth,
                        "scene_detail": args_cli.scene_detail,
                        "wrist_cameras_enabled": args_cli.wrist_cameras,
                        "control_dt": float(env.step_dt),
                        "action_layout": ",".join(action_names + BODY_ACTION_NAMES),
                        "base_control": "kinematic_fixed_root_xy_v1",
                        "joint_names": state_names,
                        "hand_joint_names": HAND_JOINT_NAMES,
                        "box_scene_keys": box_names,
                        "domain_randomization": bool(args_cli.domain_randomization),
                        "head_camera_resolution": [args_cli.head_camera_width, args_cli.head_camera_height],
                        "wrist_camera_resolution": [args_cli.wrist_camera_width, args_cli.wrist_camera_height],
                    }
                )
                print(f"[DATA] Recording {episode_name}")
                pending_start = False
                episode_steps = 0

            action_np = np.concatenate(
                (
                    mapped.action,
                    np.asarray(
                        gripper_teleop_action(
                            GRIPPER_SETTINGS,
                            mapped.left_pinch_m,
                            mapped.right_pinch_m,
                        ),
                        dtype=np.float32,
                    ),
                )
            )
            if args_cli.input_mode == "controllers":
                controllers = {"left": left_controller, "right": right_controller}
                valid = {"left": mapped.left_valid, "right": mapped.right_valid}
                for index, side in enumerate(GRIPPER_SETTINGS.active_sides):
                    # Loss of controller tracking holds the gripper, too.
                    action_np[14 + index] = (
                        (-1.0 if controllers[side][1, 2] >= 0.5 else 1.0)
                        if valid[side] else held_absolute_targets[2 + index]
                    )
            if not (recorder.recording or preview_enabled):
                action_np[:12] = 0.0
                if args_cli.input_mode == "controllers":
                    action_np[12:14] = held_absolute_targets[:2]
                    held_absolute_targets[2:] = action_np[14:]
                else:
                    action_np[12:] = held_absolute_targets
                hold_arms()
            else:
                if free_view:
                    action_np[:12] = 0.0
                    action_np[12:14] = held_absolute_targets[:2]
                held_absolute_targets[:] = action_np[12:]
            if absolute_control:
                root_pose = np.concatenate((_to_numpy(robot.data.root_pos_w[0]), root_quat))
                arm_goals = []
                for side, packet, body_id in (("left", left_controller, left_body_ids[0]),
                                              ("right", right_controller, right_body_ids[0])):
                    tool_pose = np.concatenate((_to_numpy(robot.data.body_pos_w[0, body_id]),
                                                _to_numpy(robot.data.body_quat_w[0, body_id])))
                    arm_goals.append(absolute_mapper.target(
                        side, None if free_view else packet, tool_pose, root_pose,
                        following=recorder.recording or preview_enabled,
                    ))
                action_np = np.concatenate((*arm_goals, action_np[12:]))
            body_action = body_mapper.advance(left_controller, right_controller, env.step_dt,
                                               enabled=(recorder.recording or preview_enabled) and not free_view)
            action_np = np.concatenate((action_np, body_action))
            for term in arm_terms:
                term.set_following(recorder.recording or preview_enabled)
            waist_yaw = float(robot.data.joint_pos[0, waist_yaw_id])
            if head_pose is not None and not free_view:
                xr_device.pin_view_position(head_camera_pose()[0], waist_yaw - last_waist_yaw)
            last_waist_yaw = waist_yaw
            action = torch.from_numpy(action_np).to(device=env.device).unsqueeze(0)
            if desktop_viewport is not None:
                # XR + disabled desktop viewport stops RTX annotator output
                # on Kit 107.3. Keep a tiny desktop render while sensors are needed;
                # this does not change either XR eye or recording resolution.
                desktop_viewport.updates_enabled = bool(recorder.recording or quest_overlay is not None
                                                        or not camera_reported)
            env.step(action)
            profile_steps += 1
            report_steps += 1
            if args_cli.capture_xr and profile_steps == 60:
                from omni.kit.xr.core import XRCore
                capture_path = str(Path("artifacts/quest-xr-display").resolve())
                XRCore.get_singleton().schedule_capture_display_frame(capture_path)
                print(f"[VIEW] Requested XR display capture: {capture_path}", flush=True)
                if quest_overlay is not None:
                    quest_overlay.describe()
            if profile is not None and profile_steps >= args_cli.profile_steps:
                import pstats
                profile.disable()
                profile.dump_stats("artifacts/quest-control.prof")
                pstats.Stats(profile).sort_stats("cumulative").print_stats(30)
                profile = None
            if time.perf_counter() - report_time >= 5.0:
                elapsed = time.perf_counter() - report_time
                print(f"[PERF] loop={report_steps / elapsed:.1f} Hz, {1000 * elapsed / report_steps:.0f} ms/frame; "
                      f"recording={recorder.recording}", flush=True)
                report_steps = 0
                report_time = time.perf_counter()
            if time.perf_counter() - last_motion_report >= 3.0:
                positions = _to_numpy(robot.data.body_pos_w[0, [left_body_ids[0], right_body_ids[0]]])
                if recorder.recording or preview_enabled:
                    movement = np.linalg.norm(positions - last_ee_positions, axis=1) * 1000.0
                    errors = [float(term.target_position_error()[0]) * 1000.0 for term in arm_terms]
                    print(f"[MOTION] input={args_cli.input_mode} tracking={mapped.left_valid}/{mapped.right_valid}; "
                          f"hand displacement L/R={movement[0]:.1f}/{movement[1]:.1f} mm; "
                          f"target error L/R={errors[0]:.1f}/{errors[1]:.1f} mm; "
                          f"rotation error L/R={np.rad2deg(float(arm_terms[0].target_orientation_error()[0])):.1f}/"
                          f"{np.rad2deg(float(arm_terms[1].target_orientation_error()[0])):.1f} deg", flush=True)
                if args_cli.input_mode == "controllers":
                    body_term = env.action_manager.get_term("body")
                    actual_body = _to_numpy(robot.data.joint_pos[0, body_term._joint_ids])
                    print(f"[BODY] right stick={controller_axis(right_controller, 0):.2f}/"
                          f"{controller_axis(right_controller, 1):.2f}; enabled="
                          f"{(recorder.recording or preview_enabled) and not free_view}; "
                          f"height goal={body_mapper.height:.3f} m; "
                          f"joint goal={np.round(body_action[2:], 3).tolist()}; "
                          f"actual={np.round(actual_body, 3).tolist()}", flush=True)
                last_ee_positions = positions.copy()
                last_motion_report = time.perf_counter()

            head_rgb = None
            left_wrist_rgb = None
            right_wrist_rgb = None
            if quest_overlay is not None or recorder.recording or not camera_reported:
                if recorder.recording or not camera_reported:
                    head_rgb = _camera_rgb(env.scene["robustness_camera"])
                if quest_overlay is not None or args_cli.record_wrist_cameras:
                    left_wrist_rgb = _camera_rgb(env.scene["left_wrist_camera"])
                    right_wrist_rgb = _camera_rgb(env.scene["right_wrist_camera"])
                if quest_overlay is not None and left_wrist_rgb is not None and right_wrist_rgb is not None:
                    quest_overlay.update(head_rgb, left_wrist_rgb, right_wrist_rgb)
                    quest_overlay.set_status(
                        following=recorder.recording or preview_enabled,
                        recording=recorder.recording,
                        hands_valid=mapped.bimanual_valid,
                        waiting=pending_start,
                        input_mode=args_cli.input_mode,
                    )
                if not camera_reported and head_rgb is not None and np.any(head_rgb):
                    print(f"[CAMERA] Head RGB {head_rgb.shape}: min={head_rgb.min()}, max={head_rgb.max()}, mean={head_rgb.mean():.1f}")
                    camera_reported = True

            if recorder.recording:
                head_depth = _camera_depth(env.scene["robustness_camera"]) if args_cli.record_depth else None
                if head_rgb is None or (args_cli.record_wrist_cameras
                                        and (left_wrist_rgb is None or right_wrist_rgb is None)) or (
                                            args_cli.record_depth and head_depth is None):
                    if not camera_wait_reported:
                        print("[CAMERA] Waiting for valid RGB; no empty image samples are written.", flush=True)
                        camera_wait_reported = True
                    continue
                if camera_wait_reported:
                    print("[CAMERA] Valid RGB received; recording samples resumed.", flush=True)
                    camera_wait_reported = False
                box_poses = []
                for name in box_names:
                    asset = env.scene[name]
                    box_poses.append(
                        np.concatenate(
                            [_to_numpy(asset.data.root_pos_w[0]), _to_numpy(asset.data.root_quat_w[0])]
                        )
                    )
                left_ee_pose = np.concatenate(
                    [_to_numpy(robot.data.body_pos_w[0, left_body_ids[0]]), _to_numpy(robot.data.body_quat_w[0, left_body_ids[0]])]
                )
                right_ee_pose = np.concatenate(
                    [_to_numpy(robot.data.body_pos_w[0, right_body_ids[0]]), _to_numpy(robot.data.body_quat_w[0, right_body_ids[0]])]
                )
                joint_positions = [_to_numpy(robot.data.joint_pos[0, arm_joint_ids])]
                joint_velocities = [_to_numpy(robot.data.joint_vel[0, arm_joint_ids])]
                for _, gripper, gripper_joint_ids in gripper_state_sources:
                    joint_positions.append(_to_numpy(gripper.data.joint_pos[0, gripper_joint_ids]))
                    joint_velocities.append(_to_numpy(gripper.data.joint_vel[0, gripper_joint_ids]))
                sample = {
                    "sim_time_s": np.float64(env.common_step_counter * env.step_dt),
                    "wall_time_s": np.float64(time.perf_counter() - start_wall_time),
                    "action": action_np,
                    "robot_joint_position": np.concatenate(joint_positions).astype(np.float32),
                    "robot_joint_velocity": np.concatenate(joint_velocities).astype(np.float32),
                    "left_end_effector_pose_w": left_ee_pose.astype(np.float32),
                    "right_end_effector_pose_w": right_ee_pose.astype(np.float32),
                    "openxr_left_hand": _hand_array(left_hand),
                    "openxr_right_hand": _hand_array(right_hand),
                    "openxr_head_pose": _pose_or_default(head_pose),
                    "pinch_distance_m": np.array([mapped.left_pinch_m, mapped.right_pinch_m], dtype=np.float32),
                    "tracking_valid": np.array(
                        [mapped.left_valid, mapped.right_valid, mapped.head_valid], dtype=np.uint8
                    ),
                    "robot_root_pose_w": _to_numpy(robot.data.root_pose_w[0]).astype(np.float32),
                    "free_view": np.uint8(free_view),
                    "box_root_pose_w": np.asarray(box_poses, dtype=np.float32).reshape(-1, 7),
                    "head_rgb": head_rgb,
                }
                if args_cli.record_wrist_cameras:
                    assert left_wrist_rgb is not None
                    assert right_wrist_rgb is not None
                    sample["left_wrist_rgb"] = left_wrist_rgb
                    sample["right_wrist_rgb"] = right_wrist_rgb
                if args_cli.input_mode == "controllers":
                    # No finger tracking is requested in controller mode.
                    # Preserve that absence instead of writing fake hand poses.
                    sample["openxr_left_hand"] = np.full((len(HAND_JOINT_NAMES), 7), np.nan, dtype=np.float32)
                    sample["openxr_right_hand"] = np.full((len(HAND_JOINT_NAMES), 7), np.nan, dtype=np.float32)
                    for side, packet in (("left", left_controller), ("right", right_controller)):
                        sample[f"openxr_{side}_controller"] = (
                            np.asarray(packet, dtype=np.float32) if packet is not None
                            else np.full((2, 7), np.nan, dtype=np.float32)
                        )
                if args_cli.record_depth:
                    sample["head_depth_m"] = head_depth
                if button is not None:
                    sample["button_joint_position"] = _to_numpy(button.data.joint_pos[0]).astype(np.float32)
                recorder.append(sample)
                episode_steps += 1
                if args_cli.episode_seconds and episode_steps * env.step_dt >= args_cli.episode_seconds:
                    preview_enabled = False
                    finish_episode(False, "time_limit")
                    manual_pause = not args_cli.auto_start

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if recorder.recording:
            recorder.finish_episode(success=False, reason="process_interrupted")
        recorder.close()
        if quest_overlay is not None:
            quest_overlay.close()
        env.close()
        print(f"[RESULT] Completed {completed_this_run} saved episode(s); {'; '.join(dataset_descriptions)}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
