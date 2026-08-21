#!/usr/bin/env python3
"""Collect Kuavo bimanual demonstrations from Quest/OpenXR."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Collect Kuavo Quest hand-tracking demonstrations.")
parser.add_argument(
    "--dataset-format",
    choices=("hdf5", "lerobot", "both"),
    default="hdf5",
    help="Recording backend. 'both' writes HDF5 and LeRobot in parallel.",
)
parser.add_argument(
    "--dataset",
    type=Path,
    default=Path("datasets/kuavo_quest_teleop.hdf5"),
    help="HDF5 file to create or append to.",
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
parser.add_argument("--max-episodes", type=int, default=20, help="0 means unlimited.")
parser.add_argument("--episode-seconds", type=float, default=30.0, help="Automatic episode timeout.")
parser.add_argument(
    "--auto-start",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Start recording automatically after both hands become valid.",
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
    default=True,
    help="Open Kuavo head/wrist camera windows in the desktop Isaac Sim UI.",
)
parser.add_argument("--head-camera-width", type=int, default=640)
parser.add_argument("--head-camera-height", type=int, default=360)
parser.add_argument("--wrist-camera-width", type=int, default=240)
parser.add_argument("--wrist-camera-height", type=int, default=180)
parser.add_argument(
    "--quest-camera-overlay",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Replace the Quest view with head RGB and left/right wrist overlays.",
)
parser.add_argument("--xr-overlay-distance", type=float, default=0.65, metavar="METERS")
parser.add_argument(
    "--xr-overlay-forward-axis",
    choices=("-z", "+z"),
    default="-z",
    help="Direction in front of the OpenXR head frame; use +z only if the panel appears behind you.",
)
parser.add_argument(
    "--record-depth",
    action=argparse.BooleanOptionalAction,
    default=True,
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
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if min(
    args_cli.head_camera_width,
    args_cli.head_camera_height,
    args_cli.wrist_camera_width,
    args_cli.wrist_camera_height,
) <= 0:
    parser.error("Head and wrist camera width/height values must be positive.")
if args_cli.xr_overlay_distance <= 0.08:
    parser.error("--xr-overlay-distance must be greater than the 0.08 m XR near plane.")
if args_cli.episode_seconds <= 0.0:
    parser.error("--episode-seconds must be positive.")
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

from isaaclab.devices import OpenXRDevice, Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES

from .camera_viewports import open_camera_viewports
from .manager_env import LOCAL_BOX_SCENE_KEYS
from .teleop_env import (
    HEAD_JOINTS,
    LEFT_ARM_JOINTS,
    RIGHT_ARM_JOINTS,
    KuavoQuestTeleopEnvCfg,
    set_domain_randomization,
)
from .teleop_mapping import BimanualTeleopMapper, TeleopMappingCfg
from .teleop_lerobot_recorder import LeRobotTeleopRecorder
from .teleop_recorder import TeleopHdf5Recorder, TeleopRecorderGroup
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


def _camera_rgb(camera) -> np.ndarray:
    rgb = _to_numpy(camera.data.output["rgb"][0])
    if rgb.shape[-1] > 3:
        rgb = rgb[..., :3]
    if rgb.dtype.kind == "f":
        scale = 255.0 if float(np.nanmax(rgb)) <= 1.01 else 1.0
        rgb = np.clip(rgb * scale, 0.0, 255.0).astype(np.uint8)
    return rgb.astype(np.uint8, copy=False)


def _camera_depth(camera) -> np.ndarray:
    depth = _to_numpy(camera.data.output["distance_to_image_plane"][0])
    return depth.astype(np.float16)


def _scene_asset_or_none(scene, name: str):
    try:
        return scene[name]
    except KeyError:
        return None


def main() -> None:
    cfg = KuavoQuestTeleopEnvCfg()
    cfg.seed = args_cli.seed
    cfg.scene.robustness_camera.width = args_cli.head_camera_width
    cfg.scene.robustness_camera.height = args_cli.head_camera_height
    cfg.scene.left_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.left_wrist_camera.height = args_cli.wrist_camera_height
    cfg.scene.right_wrist_camera.width = args_cli.wrist_camera_width
    cfg.scene.right_wrist_camera.height = args_cli.wrist_camera_height
    cfg.sim.render.antialiasing_mode = "DLSS"
    set_domain_randomization(cfg, args_cli.domain_randomization)

    env = ManagerBasedRLEnv(cfg=cfg)
    env.reset(seed=args_cli.seed)
    if args_cli.camera_preview:
        open_camera_viewports(
            env.scene,
            ["robustness_camera", "left_wrist_camera", "right_wrist_camera"],
            headless=args_cli.headless,
            width=240,
            height=180,
            columns=3,
        )

    robot = env.scene["robot"]
    arm_joint_ids, arm_joint_names = robot.find_joints(LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS + HEAD_JOINTS)
    left_body_ids, _ = robot.find_bodies("zarm_l7_end_effector")
    right_body_ids, _ = robot.find_bodies("zarm_r7_end_effector")
    if len(left_body_ids) != 1 or len(right_body_ids) != 1:
        raise RuntimeError("Could not resolve both Kuavo end-effector bodies.")
    box_names = [name for name in LOCAL_BOX_SCENE_KEYS if _scene_asset_or_none(env.scene, name) is not None]
    button = _scene_asset_or_none(env.scene, "button_station")
    button_joint_count = int(button.data.joint_pos.shape[-1]) if button is not None else 0

    device_cfg = cfg.teleop_devices.devices["quest_handtracking"]
    xr_device = OpenXRDevice(device_cfg)
    keyboard = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.0, rot_sensitivity=0.0, sim_device=env.device))
    mapper = BimanualTeleopMapper(
        TeleopMappingCfg(
            position_gain=args_cli.position_gain,
            rotation_gain=args_cli.rotation_gain,
        )
    )

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

    requests = {"start": False, "stop": False, "reset": False, "success": False, "toggle": False}
    xr_device.add_callback("START", lambda: requests.__setitem__("start", True))
    xr_device.add_callback("STOP", lambda: requests.__setitem__("stop", True))
    xr_device.add_callback("RESET", lambda: requests.__setitem__("reset", True))
    keyboard.add_callback("P", lambda: requests.__setitem__("toggle", True))
    keyboard.add_callback("R", lambda: requests.__setitem__("reset", True))
    keyboard.add_callback("M", lambda: requests.__setitem__("success", True))

    dataset_path = args_cli.dataset.expanduser().resolve()
    lerobot_root = args_cli.lerobot_root.expanduser().resolve()
    recorders = {}
    dataset_descriptions = []
    if args_cli.dataset_format in {"hdf5", "both"}:
        recorders["hdf5"] = TeleopHdf5Recorder(dataset_path)
        dataset_descriptions.append(f"HDF5={dataset_path}")
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
            joint_names=arm_joint_names,
            hand_joint_names=HAND_JOINT_NAMES,
            head_resolution=(args_cli.head_camera_width, args_cli.head_camera_height),
            wrist_resolution=(args_cli.wrist_camera_width, args_cli.wrist_camera_height),
            box_count=len(box_names),
            button_joint_count=button_joint_count,
            record_wrist_cameras=args_cli.record_wrist_cameras,
            use_videos=args_cli.lerobot_use_videos,
            save_failed=args_cli.lerobot_save_failed,
            writer_python=args_cli.lerobot_python,
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

    def reset_simulation() -> None:
        nonlocal episode_steps
        env.reset()
        xr_device.reset()
        mapper.reset()
        episode_steps = 0

    def finish_episode(success: bool, reason: str) -> None:
        nonlocal completed_this_run
        name = recorder.finish_episode(success=success, reason=reason)
        if name is not None:
            completed_this_run += 1
            print(f"[DATA] Finished {name}: success={success}, reason={reason}")
        reset_simulation()

    print("[INFO] Quest/OpenXR Kuavo teleoperation is ready.")
    print(f"[INFO] Dataset: {'; '.join(dataset_descriptions)}")
    if args_cli.dataset_format in {"lerobot", "both"} and not args_cli.lerobot_save_failed:
        print("[INFO] LeRobot keeps successful episodes only; STOP/RESET/time-limit attempts are discarded.")
    if quest_overlay is not None:
        print("[INFO] Quest view: opaque Kuavo head camera with left/right wrist camera overlays.")
    print("[CONTROL] Quest START/STOP/RESET or desktop P=start/stop, R=reset, M=finish as success.")
    print("[CONTROL] Hold both tracked hands in view; the first valid frame only calibrates and never moves the arms.")
    print("[NOTE] Pinch and all 26 joints/hand are recorded, but the current Kuavo USD has no actuated finger joints.")

    try:
        while simulation_app.is_running():
            if args_cli.max_episodes and completed_this_run >= args_cli.max_episodes:
                break

            raw = xr_device.advance()
            left_hand = raw.get(OpenXRDevice.TrackingTarget.HAND_LEFT)
            right_hand = raw.get(OpenXRDevice.TrackingTarget.HAND_RIGHT)
            head_pose = raw.get(OpenXRDevice.TrackingTarget.HEAD)
            root_quat = _to_numpy(robot.data.root_quat_w[0])
            mapped = mapper.advance(left_hand, right_hand, head_pose, root_quat)
            tracking_state = (mapped.left_valid, mapped.right_valid, mapped.head_valid)
            if tracking_state != last_tracking_state:
                print(
                    "[TRACKING] "
                    f"left={tracking_state[0]}, right={tracking_state[1]}, head={tracking_state[2]}"
                )
                last_tracking_state = tracking_state

            if requests["toggle"]:
                requests["toggle"] = False
                if recorder.recording:
                    requests["stop"] = True
                else:
                    requests["start"] = True
            if requests["reset"]:
                requests["reset"] = False
                if recorder.recording:
                    finish_episode(False, "operator_reset")
                else:
                    reset_simulation()
                pending_start = False
                manual_pause = True
                continue
            if requests["success"]:
                requests["success"] = False
                if recorder.recording:
                    finish_episode(True, "operator_success")
                pending_start = False
                manual_pause = not args_cli.auto_start
                continue
            if requests["stop"]:
                requests["stop"] = False
                if recorder.recording:
                    finish_episode(False, "operator_stop")
                pending_start = False
                manual_pause = True
                continue
            if requests["start"]:
                requests["start"] = False
                pending_start = True
                manual_pause = False
            if args_cli.auto_start and not manual_pause and not recorder.recording:
                pending_start = True

            if pending_start and mapped.bimanual_valid and not recorder.recording:
                episode_name = recorder.start_episode(
                    {
                        "seed": args_cli.seed,
                        "control_dt": float(env.step_dt),
                        "action_layout": "left_delta_pose_6,right_delta_pose_6,head_yaw_pitch_2",
                        "joint_names": arm_joint_names,
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

            action_np = mapped.action.copy()
            if not recorder.recording:
                action_np[:12] = 0.0
            action = torch.from_numpy(action_np).to(device=env.device).unsqueeze(0)
            env.step(action)

            head_rgb = None
            left_wrist_rgb = None
            right_wrist_rgb = None
            if quest_overlay is not None or recorder.recording:
                head_rgb = _camera_rgb(env.scene["robustness_camera"])
                left_wrist_rgb = _camera_rgb(env.scene["left_wrist_camera"])
                right_wrist_rgb = _camera_rgb(env.scene["right_wrist_camera"])
                if quest_overlay is not None:
                    quest_overlay.update(head_rgb, left_wrist_rgb, right_wrist_rgb)

            if recorder.recording:
                assert head_rgb is not None
                assert left_wrist_rgb is not None
                assert right_wrist_rgb is not None
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
                sample = {
                    "sim_time_s": np.float64(env.common_step_counter * env.step_dt),
                    "wall_time_s": np.float64(time.perf_counter() - start_wall_time),
                    "action": action_np,
                    "robot_joint_position": _to_numpy(robot.data.joint_pos[0, arm_joint_ids]).astype(np.float32),
                    "robot_joint_velocity": _to_numpy(robot.data.joint_vel[0, arm_joint_ids]).astype(np.float32),
                    "left_end_effector_pose_w": left_ee_pose.astype(np.float32),
                    "right_end_effector_pose_w": right_ee_pose.astype(np.float32),
                    "openxr_left_hand": _hand_array(left_hand),
                    "openxr_right_hand": _hand_array(right_hand),
                    "openxr_head_pose": _pose_or_default(head_pose),
                    "pinch_distance_m": np.array([mapped.left_pinch_m, mapped.right_pinch_m], dtype=np.float32),
                    "tracking_valid": np.array(
                        [mapped.left_valid, mapped.right_valid, mapped.head_valid], dtype=np.uint8
                    ),
                    "box_root_pose_w": np.asarray(box_poses, dtype=np.float32).reshape(-1, 7),
                    "head_rgb": head_rgb,
                }
                if args_cli.record_wrist_cameras:
                    sample["left_wrist_rgb"] = left_wrist_rgb
                    sample["right_wrist_rgb"] = right_wrist_rgb
                if args_cli.record_depth:
                    sample["head_depth_m"] = _camera_depth(env.scene["robustness_camera"])
                if button is not None:
                    sample["button_joint_position"] = _to_numpy(button.data.joint_pos[0]).astype(np.float32)
                recorder.append(sample)
                episode_steps += 1
                if episode_steps * env.step_dt >= args_cli.episode_seconds:
                    finish_episode(False, "time_limit")
                    manual_pause = not args_cli.auto_start

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
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
