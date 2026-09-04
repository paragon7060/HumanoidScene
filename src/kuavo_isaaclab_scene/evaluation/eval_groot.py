#!/usr/bin/env python3
"""Evaluate a LeRobot GR00T policy in the Kuavo Isaac Lab workcell."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from isaaclab.app import AppLauncher

from ..core.paths import default_artifacts_dir
from .eval_metrics import (
    control_decimation,
    percentile_nearest_rank,
    termination_reason_counts,
    wilson_score_interval,
)
from ..robots.gripper_config import add_gripper_cli_args, export_gripper_cli, resolve_gripper_settings
from ..robots.robot_model import add_robot_model_cli_args, export_robot_model_cli, resolve_robot_model
from ..workcell.rack_box_layout import resolve_rack_box_pose_path


parser = argparse.ArgumentParser(
    description="Run serial GR00T rollouts in the Kuavo rack-to-conveyor task."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Local LeRobot pretrained_model directory or Hugging Face repository ID.",
)
parser.add_argument(
    "--policy-profile",
    choices=("default", "rwh-kuavo-v2-s56"),
    default="default",
    help="Policy-specific state/action/camera adapter.",
)
parser.add_argument(
    "--lerobot-python",
    type=Path,
    default=Path(os.environ["LEROBOT_PYTHON"]) if os.environ.get("LEROBOT_PYTHON") else None,
    help="Run GR00T in this separate LeRobot Conda Python instead of the Isaac Lab process.",
)
parser.add_argument(
    "--base-model-path",
    type=str,
    default=None,
    help="Override a checkpoint base_model_path saved on another training host.",
)
parser.add_argument(
    "--allow-checkpoint-key-mismatch",
    action="store_true",
    help="Load with strict=False for compatible checkpoints saved by a nearby LeRobot revision.",
)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Maximum control steps per episode; 0 uses the environment time limit.",
)
parser.add_argument(
    "--control-hz",
    type=float,
    default=None,
    help=(
        "Environment action/observation rate. It must divide the physics rate exactly "
        "(default: environment configuration)."
    ),
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--task",
    type=str,
    default=(
        "Move every open box from the rack into an unoccupied conveyor space, "
        "keep the contents inside, then press the green button."
    ),
    help="Language instruction supplied to GR00T as the LeRobot 'task' field.",
)
parser.add_argument(
    "--policy-device",
    type=str,
    default=None,
    help="GR00T device, e.g. cuda:1 to keep Isaac Sim on cuda:0 (default: --device).",
)
parser.add_argument(
    "--actions-per-inference",
    type=int,
    default=None,
    help=(
        "Decoded chunk actions to execute before re-planning "
        "(default: checkpoint n_action_steps)."
    ),
)
parser.add_argument(
    "--state-mode",
    choices=("manager", "joint_position"),
    default=None,
    help="Representation used by observation.state; it must match the training dataset.",
)
parser.add_argument(
    "--action-mode",
    choices=("manager", "joint_position", "joint_delta"),
    default=None,
    help="Representation of the decoded policy action; it must match the training dataset.",
)
parser.add_argument(
    "--action-clip",
    type=float,
    default=None,
    help=(
        "Clamp manager actions to +/-VALUE; use 0 to disable clipping. Defaults to 0 "
        "for the RwH S56 profile and 1 otherwise. S56 joint limits are always enforced."
    ),
)
parser.add_argument(
    "--initial-pose",
    choices=("default", "checkpoint-q50", "dataset-medoid"),
    default=None,
    help=(
        "Robot reset pose. The RwH S56 profile defaults to the checkpoint's "
        "observation.state q50 arm pose with open claws; dataset-medoid uses an "
        "actual episode-start pose from parcel-related training data."
    ),
)
parser.add_argument(
    "--initial-head-pitch-deg",
    type=float,
    default=0.0,
    metavar="DEGREES",
    help=(
        "Initial zhead_2_joint pitch in degrees. Positive values look downward; "
        "the S56 joint range is +/-30 degrees (default: 0)."
    ),
)
parser.add_argument(
    "--camera-map",
    action="append",
    default=None,
    metavar="POLICY_KEY=SCENE_CAMERA",
    help=(
        "Map a checkpoint image key to an Isaac camera. Repeat as needed; short keys such "
        "as head=robustness_camera are accepted."
    ),
)
parser.add_argument(
    "--camera-preview",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Open the four small camera windows in non-headless Isaac Sim.",
)
parser.add_argument("--camera-width", type=int, default=None)
parser.add_argument("--camera-height", type=int, default=None)
parser.add_argument(
    "--domain-randomization",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Evaluate with material/physics/light/obstacle randomization.",
)
parser.add_argument(
    "--metrics-out",
    type=Path,
    default=None,
    help="JSON output path (default: artifacts/eval/groot_eval_<UTC>.json).",
)
parser.add_argument(
    "--trace-out",
    type=Path,
    default=None,
    metavar="JSON",
    help=(
        "Optional per-step JSON trace containing policy targets, observed 16-D state "
        "before/after stepping, and adapted manager commands."
    ),
)
parser.add_argument(
    "--video-out",
    type=Path,
    default=None,
    metavar="MP4",
    help=(
        "Record policy cameras as one left-to-right MP4, including in headless mode. "
        "Multiple episodes add _epNNN to the filename."
    ),
)
parser.add_argument(
    "--video-fps",
    type=float,
    default=None,
    help="Recorded video FPS (default: environment control rate).",
)
parser.add_argument(
    "--video-height",
    type=int,
    default=None,
    help="Resize each camera tile to this height while preserving aspect ratio.",
)
parser.add_argument(
    "--video-scene-view",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
        "Include a fixed side view of the full robot and workcell in --video-out "
        "(default: enabled)."
    ),
)
parser.add_argument(
    "--scene-camera-eye",
    type=float,
    nargs=3,
    default=(-0.2, -3.2, 1.8),
    metavar=("X", "Y", "Z"),
    help="World-space eye position for the recorded fixed scene camera.",
)
parser.add_argument(
    "--scene-camera-target",
    type=float,
    nargs=3,
    default=(-0.2, 0.45, 1.0),
    metavar=("X", "Y", "Z"),
    help="World-space look-at target for the recorded fixed scene camera.",
)
parser.add_argument(
    "--overwrite-video",
    action="store_true",
    help="Allow --video-out to replace an existing MP4.",
)
parser.add_argument(
    "--local-files-only",
    action="store_true",
    help="Never download checkpoint/model files from Hugging Face Hub.",
)
parser.add_argument(
    "--mock-policy",
    action="store_true",
    help="Run zero actions without importing/loading LeRobot; useful for a smoke test.",
)

rack_group = parser.add_mutually_exclusive_group()
rack_group.add_argument("--rack-boxes", type=str, default=None, metavar="SPEC")
rack_group.add_argument("--rack-box-layout", type=Path, default=None, metavar="JSON")
pose_group = parser.add_mutually_exclusive_group()
pose_group.add_argument("--rack-box-poses", type=Path, default=None, metavar="JSON")
pose_group.add_argument("--ignore-captured-box-poses", action="store_true")

add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
has_robot_arg = any(
    value == "--robot-model" or value.startswith("--robot-model=") for value in sys.argv[1:]
)
has_gripper_arg = any(
    value == "--gripper" or value.startswith("--gripper=") for value in sys.argv[1:]
)
has_head_pitch_arg = any(
    value == "--initial-head-pitch-deg" or value.startswith("--initial-head-pitch-deg=")
    for value in sys.argv[1:]
)
if args_cli.policy_profile == "rwh-kuavo-v2-s56":
    if has_robot_arg and args_cli.robot_model != "s56":
        parser.error("--policy-profile rwh-kuavo-v2-s56 only supports --robot-model s56.")
    if has_gripper_arg and args_cli.gripper not in ("s56_twofinger", "s56_qiangnao"):
        parser.error(
            "--policy-profile rwh-kuavo-v2-s56 requires an integrated S56 gripper "
            "(--gripper s56_twofinger or s56_qiangnao)."
        )
    args_cli.robot_model = "s56"
    args_cli.gripper = args_cli.gripper if has_gripper_arg else "s56_twofinger"
    args_cli.state_mode = args_cli.state_mode or "joint_position"
    args_cli.action_mode = args_cli.action_mode or "joint_position"
    if args_cli.state_mode != "joint_position" or args_cli.action_mode != "joint_position":
        parser.error(
            "--policy-profile rwh-kuavo-v2-s56 requires --state-mode joint_position "
            "and --action-mode joint_position."
        )
    args_cli.camera_width = args_cli.camera_width or 848
    args_cli.camera_height = args_cli.camera_height or 480
    args_cli.action_clip = 0.0 if args_cli.action_clip is None else args_cli.action_clip
    args_cli.initial_pose = args_cli.initial_pose or "checkpoint-q50"
    if args_cli.initial_pose == "dataset-medoid" and not has_head_pitch_arg:
        args_cli.initial_head_pitch_deg = 25.0
else:
    args_cli.state_mode = args_cli.state_mode or "manager"
    args_cli.action_mode = args_cli.action_mode or "manager"
    args_cli.action_clip = 1.0 if args_cli.action_clip is None else args_cli.action_clip
    args_cli.initial_pose = args_cli.initial_pose or "default"
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)

if not args_cli.mock_policy and not args_cli.checkpoint:
    parser.error("--checkpoint is required unless --mock-policy is selected.")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive.")
if args_cli.max_steps < 0:
    parser.error("--max-steps must be zero or positive.")
if args_cli.control_hz is not None and args_cli.control_hz <= 0:
    parser.error("--control-hz must be positive.")
if args_cli.actions_per_inference is not None and args_cli.actions_per_inference <= 0:
    parser.error("--actions-per-inference must be positive.")
if args_cli.action_clip < 0:
    parser.error("--action-clip must be zero or positive.")
if not math.isfinite(args_cli.initial_head_pitch_deg) or not -30.0 <= args_cli.initial_head_pitch_deg <= 30.0:
    parser.error("--initial-head-pitch-deg must be finite and between -30 and 30 degrees.")
if args_cli.video_fps is not None and args_cli.video_fps <= 0:
    parser.error("--video-fps must be positive.")
if args_cli.video_height is not None and args_cli.video_height <= 0:
    parser.error("--video-height must be positive.")
for name in ("scene_camera_eye", "scene_camera_target"):
    if not all(math.isfinite(value) for value in getattr(args_cli, name)):
        parser.error(f"--{name.replace('_', '-')} values must be finite.")
if args_cli.video_out is not None and args_cli.video_out.suffix.lower() != ".mp4":
    parser.error("--video-out must use the .mp4 extension.")
if args_cli.video_out is not None:
    if shutil.which("ffmpeg") is None:
        parser.error("--video-out requires ffmpeg on PATH.")
    video_base = args_cli.video_out.expanduser().resolve()
    planned_videos = (
        [video_base]
        if args_cli.episodes == 1
        else [
            video_base.with_name(f"{video_base.stem}_ep{index:03d}{video_base.suffix}")
            for index in range(args_cli.episodes)
        ]
    )
    existing_videos = [str(path) for path in planned_videos if path.exists()]
    if existing_videos and not args_cli.overwrite_video:
        parser.error(
            "video output already exists; choose another --video-out or pass "
            f"--overwrite-video: {', '.join(existing_videos)}"
        )
for name in ("camera_width", "camera_height"):
    value = getattr(args_cli, name)
    if value is not None and value <= 0:
        parser.error(f"--{name.replace('_', '-')} must be positive.")

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
try:
    captured_pose_path = resolve_rack_box_pose_path(
        args_cli.rack_box_poses,
        ignore=args_cli.ignore_captured_box_poses,
    )
    ACTIVE_ROBOT_MODEL = resolve_robot_model()
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))
if captured_pose_path is not None:
    os.environ["KUAVO_RACK_BOX_POSES"] = str(captured_pose_path)

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import CameraCfg

from ..envs import manager_mdp as workcell_mdp
from ..display.camera_viewports import open_camera_viewports
from ..display.eval_video import FfmpegVideoWriter, camera_mosaic_rgb, episode_video_path
from ..robots.gripper_runtime import build_gripper_action_cfg
from .groot_lerobot_bridge import (
    CONTROLLED_JOINT_NAMES,
    KuavoLeRobotBridge,
    LeRobotGrootRunner,
    RWH_KUAVO_V2_CAMERA_MAP,
    RWH_KUAVO_V2_S56_PROFILE,
    SubprocessLeRobotGrootRunner,
    ZeroLeRobotPolicyRunner,
    parse_camera_map,
)
from ..envs.manager_env import (
    ACTIVE_RACK_BOX_SCENE_KEYS,
    KuavoRobustWorkcellEnvCfg,
    TASK_OBJECT_PARAMS,
)


EVAL_SCENE_CAMERA_NAME = "eval_scene_camera"
EVAL_SCENE_VIDEO_KEY = "scene_side"


def _checkpoint_state_q50(checkpoint: str | None) -> torch.Tensor:
    """Load the 16-D state median saved by the selected LeRobot checkpoint."""
    if not checkpoint:
        raise ValueError(
            "--initial-pose checkpoint-q50 requires a local --checkpoint; "
            "use --initial-pose default for a checkpoint-free mock run."
        )
    root = Path(checkpoint).expanduser()
    if not (root / "policy_preprocessor.json").is_file() and (
        root / "pretrained_model" / "policy_preprocessor.json"
    ).is_file():
        root = root / "pretrained_model"
    processor_path = root / "policy_preprocessor.json"
    if not processor_path.is_file():
        raise ValueError(
            f"Cannot read checkpoint preprocessing statistics from {processor_path}; "
            "download the checkpoint locally or use --initial-pose default."
        )
    payload = json.loads(processor_path.read_text(encoding="utf-8"))
    state_file = None
    for step in payload.get("steps", []):
        if str(step.get("registry_name", "")).startswith("groot_pack_inputs"):
            state_file = step.get("state_file")
            break
    if not state_file:
        raise ValueError(f"{processor_path} has no GR00T pack-input statistics file.")
    from safetensors.torch import load_file

    tensors = load_file(str(root / str(state_file)), device="cpu")
    try:
        ready_state = tensors["observation.state.q50"].float()
    except KeyError as exc:
        raise ValueError(
            f"{root / str(state_file)} has no observation.state.q50 tensor."
        ) from exc
    if ready_state.shape != (16,) or not torch.isfinite(ready_state).all():
        raise ValueError(
            "Checkpoint observation.state.q50 must be a finite 16-D tensor; "
            f"received {tuple(ready_state.shape)}."
        )
    return ready_state


def _configure_rwh_ready_pose(
    cfg: KuavoRobustWorkcellEnvCfg,
    checkpoint: str | None,
) -> list[float]:
    """Set the S56 arms to the checkpoint median while keeping both claws open."""
    state = _checkpoint_state_q50(checkpoint)
    arm_values = torch.cat((state[:7], state[8:15]))
    arm_names = (
        *(f"zarm_l{index}_joint" for index in range(1, 8)),
        *(f"zarm_r{index}_joint" for index in range(1, 8)),
    )
    cfg.scene.robot.init_state.joint_pos.pop("zarm_.*_joint", None)
    cfg.scene.robot.init_state.joint_pos.update(
        {name: float(value) for name, value in zip(arm_names, arm_values, strict=True)}
    )
    return [float(value) for value in state]


# Medoid of frame_index==0 across 2,485 episodes from the six parcel/box-like
# tasks sampled from LejuRobotics/LET-KUAVO-VLA-1.0-Dataset. Unlike a synthetic
# per-joint median, this complete vector occurred in one real demonstration.
DATASET_MEDOID_SOURCE = {
    "dataset": "LejuRobotics/LET-KUAVO-VLA-1.0-Dataset",
    "revision": "61e3256ac917721c3c9d2098db28b1f5efc2d54a",
    "sampled_episode_count": 2485,
    "task": "078-scan-cardboard-parcels",
    "episode_index": 234,
}
DATASET_MEDOID_INITIAL_STATE = (
    -0.10273750,
    0.08924951,
    -0.34955031,
    -1.81849384,
    -0.09739541,
    -0.08999525,
    -0.05490084,
    0.0,
    0.01990609,
    -0.09497114,
    0.36557317,
    -1.83718586,
    -0.11203432,
    0.18078724,
    0.00728037,
    0.0,
)


def _configure_dataset_medoid_pose(cfg: KuavoRobustWorkcellEnvCfg) -> list[float]:
    """Use one real episode-start state; do not alter authored camera/link transforms."""
    cfg.scene.robot.init_state.joint_pos.pop("zarm_.*_joint", None)
    arm_values = (*DATASET_MEDOID_INITIAL_STATE[:7], *DATASET_MEDOID_INITIAL_STATE[8:15])
    arm_names = (
        *(f"zarm_l{index}_joint" for index in range(1, 8)),
        *(f"zarm_r{index}_joint" for index in range(1, 8)),
    )
    cfg.scene.robot.init_state.joint_pos.update(
        {
            name: float(value)
            for name, value in zip(arm_names, arm_values, strict=True)
        }
    )
    return list(DATASET_MEDOID_INITIAL_STATE)


def _disable_domain_randomization(cfg: KuavoRobustWorkcellEnvCfg) -> None:
    """Keep reset-to-default while disabling stochastic reset/interval events."""
    for name in (
        "robot_material",
        "robot_arm_mass",
        "actuator_gains",
        "left_gripper_material",
        "right_gripper_material",
        "left_gripper_gains",
        "right_gripper_gains",
        "tote_physics",
        "cargo_physics",
        "gravity",
        "reset_flap_friction",
        "reset_workcell",
        "reset_movers",
        "lighting",
        "move_movers",
        "cargo_disturbance",
    ):
        setattr(cfg.events, name, None)


def _episode_termination_value(extras: dict, name: str) -> float:
    value = extras.get("log", {}).get(f"Episode_Termination/{name}", 0.0)
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _termination_reason(extras: dict, success: bool, truncated: bool) -> str:
    if success:
        return "success"
    for name in ("cargo_spill", "tote_drop", "moving_robot_contact"):
        if _episode_termination_value(extras, name) > 0.5:
            return name
    return "time_limit" if truncated else "terminated"


def _default_metrics_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return default_artifacts_dir() / "eval" / f"groot_eval_{stamp}.json"


def main() -> None:
    try:
        default_camera_map = (
            RWH_KUAVO_V2_CAMERA_MAP
            if args_cli.policy_profile == RWH_KUAVO_V2_S56_PROFILE
            else None
        )
        camera_map = parse_camera_map(args_cli.camera_map, default_map=default_camera_map)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = KuavoRobustWorkcellEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.curriculum = None
    cfg.scene.robot.init_state.joint_pos.pop("zhead_.*_joint", None)
    cfg.scene.robot.init_state.joint_pos.update(
        {
            "zhead_1_joint": 0.0,
            "zhead_2_joint": math.radians(args_cli.initial_head_pitch_deg),
        }
    )
    ready_state: list[float] | None = None
    initial_state_source = args_cli.initial_pose
    if args_cli.policy_profile == RWH_KUAVO_V2_S56_PROFILE:
        cfg.actions.left_gripper = build_gripper_action_cfg(
            GRIPPER_SETTINGS, "left", continuous=True
        )
        cfg.actions.right_gripper = build_gripper_action_cfg(
            GRIPPER_SETTINGS, "right", continuous=True
        )
        if args_cli.initial_pose == "checkpoint-q50":
            try:
                ready_state = _configure_rwh_ready_pose(cfg, args_cli.checkpoint)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                parser.error(str(exc))
        elif args_cli.initial_pose == "dataset-medoid":
            ready_state = _configure_dataset_medoid_pose(cfg)
            print(f"[INFO] Dataset medoid source: {DATASET_MEDOID_SOURCE}")
    if args_cli.control_hz is not None:
        try:
            cfg.decimation = control_decimation(cfg.sim.dt, args_cli.control_hz)
        except ValueError as exc:
            parser.error(str(exc))
        cfg.sim.render_interval = cfg.decimation
    if args_cli.max_steps:
        # An explicit evaluator horizon must not be cut short by the task's
        # default TimeOutTerm (24 s in the current workcell configuration).
        requested_episode_s = args_cli.max_steps * cfg.sim.dt * cfg.decimation
        cfg.episode_length_s = max(cfg.episode_length_s, requested_episode_s)
    if args_cli.camera_width is not None:
        for scene_camera in camera_map.values():
            camera_cfg = getattr(cfg.scene, scene_camera, None)
            if camera_cfg is not None:
                camera_cfg.width = args_cli.camera_width
    if args_cli.camera_height is not None:
        for scene_camera in camera_map.values():
            camera_cfg = getattr(cfg.scene, scene_camera, None)
            if camera_cfg is not None:
                camera_cfg.height = args_cli.camera_height
    if args_cli.video_out is not None and args_cli.video_scene_view:
        setattr(
            cfg.scene,
            EVAL_SCENE_CAMERA_NAME,
            CameraCfg(
                prim_path="{ENV_REGEX_NS}/EvalSceneCamera",
                update_period=0.0,
                height=args_cli.camera_height or 480,
                width=args_cli.camera_width or 848,
                data_types=["rgb"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=20.0,
                    focus_distance=3.0,
                    horizontal_aperture=24.0,
                    clipping_range=(0.05, 20.0),
                ),
            ),
        )
    if not args_cli.domain_randomization:
        _disable_domain_randomization(cfg)

    env = ManagerBasedRLEnv(cfg=cfg)
    scene_video_camera = (
        env.scene[EVAL_SCENE_CAMERA_NAME]
        if args_cli.video_out is not None and args_cli.video_scene_view
        else None
    )
    if scene_video_camera is not None:
        scene_video_camera.set_world_poses_from_view(
            eyes=torch.tensor([args_cli.scene_camera_eye], device=env.device),
            targets=torch.tensor([args_cli.scene_camera_target], device=env.device),
        )
    bridge = KuavoLeRobotBridge(
        env,
        camera_map=camera_map,
        state_mode=args_cli.state_mode,
        action_mode=args_cli.action_mode,
        action_clip=None if args_cli.action_clip == 0 else args_cli.action_clip,
        policy_profile=args_cli.policy_profile,
        gripper_settings=GRIPPER_SETTINGS,
    )
    if args_cli.camera_preview:
        open_camera_viewports(
            env.scene,
            list(dict.fromkeys(camera_map.values())),
            headless=args_cli.headless,
        )

    policy_device = args_cli.policy_device or args_cli.device
    try:
        if args_cli.mock_policy:
            runner = ZeroLeRobotPolicyRunner(action_dim=bridge.action_dim)
            checkpoint_label = "mock:zero"
        else:
            print(
                f"[INFO] Loading GR00T checkpoint {args_cli.checkpoint!r} "
                f"on {policy_device}."
            )
            if args_cli.lerobot_python is not None:
                runner = SubprocessLeRobotGrootRunner(
                    python_executable=str(args_cli.lerobot_python),
                    checkpoint=args_cli.checkpoint,
                    device=policy_device,
                    actions_per_inference=args_cli.actions_per_inference,
                    local_files_only=args_cli.local_files_only,
                    expected_action_dim=bridge.action_dim,
                    base_model_path=args_cli.base_model_path,
                    strict=not args_cli.allow_checkpoint_key_mismatch,
                )
                print(
                    f"[INFO] Policy worker: {args_cli.lerobot_python} "
                    f"(LeRobot {runner.lerobot_version})."
                )
            else:
                runner = LeRobotGrootRunner.from_pretrained(
                    args_cli.checkpoint,
                    device=policy_device,
                    actions_per_inference=args_cli.actions_per_inference,
                    local_files_only=args_cli.local_files_only,
                    expected_action_dim=bridge.action_dim,
                    base_model_path=args_cli.base_model_path,
                    strict=not args_cli.allow_checkpoint_key_mismatch,
                )
            checkpoint_label = args_cli.checkpoint
    except Exception:
        env.close()
        raise

    max_steps = args_cli.max_steps or math.ceil(cfg.episode_length_s / env.step_dt)
    print(
        f"[INFO] Eval ready: episodes={args_cli.episodes}, max_steps={max_steps}, "
        f"control_hz={1.0 / env.step_dt:.1f}, "
        f"task_boxes={ACTIVE_RACK_BOX_SCENE_KEYS or 'legacy_totes'}"
    )
    print(f"[INFO] State schema ({len(bridge.state_names)}): {', '.join(bridge.state_names)}")
    print(f"[INFO] Policy action schema ({bridge.action_dim}): {', '.join(bridge.action_names)}")
    print(f"[INFO] Gripper preset: {GRIPPER_SETTINGS.name}")
    print(
        f"[INFO] Initial pose: {args_cli.initial_pose}; "
        "gripper action: continuous signed interpolation"
        if args_cli.policy_profile == RWH_KUAVO_V2_S56_PROFILE
        else f"[INFO] Initial pose: {args_cli.initial_pose}"
    )
    print(f"[INFO] Initial head pitch: {args_cli.initial_head_pitch_deg:.1f} deg")
    print(f"[INFO] Camera map: {camera_map}")
    video_fps = args_cli.video_fps or (1.0 / env.step_dt)
    video_camera_keys = tuple(camera_map) + (
        (EVAL_SCENE_VIDEO_KEY,) if scene_video_camera is not None else ()
    )
    if args_cli.video_out is not None:
        print(
            f"[INFO] Video recording: {args_cli.video_out} at {video_fps:.3f} FPS; "
            f"tiles={video_camera_keys}"
        )

    episode_results: list[dict[str, object]] = []
    all_inference_latencies: list[float] = []
    recorded_video_paths: list[str] = []
    rollout_trace: list[dict[str, object]] = []
    try:
        for episode_index in range(args_cli.episodes):
            if not simulation_app.is_running():
                break
            env.reset(seed=args_cli.seed + episode_index)
            runner.reset()
            reward_sum = 0.0
            max_progress = 0.0
            saturation_sum = 0.0
            inference_latencies: list[float] = []
            success = False
            reason = "max_steps"
            completed_steps = 0
            video_writer: FfmpegVideoWriter | None = None
            video_path = (
                episode_video_path(args_cli.video_out, episode_index, args_cli.episodes)
                if args_cli.video_out is not None
                else None
            )
            written_video: str | None = None
            try:
                for step_index in range(max_steps):
                    if not simulation_app.is_running():
                        reason = "simulation_closed"
                        break
                    progress = workcell_mdp.task_progress(env, **TASK_OBJECT_PARAMS)
                    max_progress = max(max_progress, float(progress[0].item()))
                    observation = bridge.observation(args_cli.task)
                    if video_path is not None:
                        video_observation = observation
                        if scene_video_camera is not None:
                            video_observation = {
                                **observation,
                                EVAL_SCENE_VIDEO_KEY: scene_video_camera.data.output["rgb"],
                            }
                        frame = camera_mosaic_rgb(
                            video_observation,
                            video_camera_keys,
                            output_height=args_cli.video_height,
                        )
                        if video_writer is None:
                            video_writer = FfmpegVideoWriter(
                                video_path,
                                width=int(frame.shape[1]),
                                height=int(frame.shape[0]),
                                fps=video_fps,
                                overwrite=args_cli.overwrite_video,
                            )
                        video_writer.write(frame)
                    sample = runner.select_action(observation)
                    if sample.inferred_new_chunk:
                        inference_latencies.append(sample.inference_ms)
                    adapted = bridge.action(sample.action)
                    saturation_sum += adapted.saturation_fraction
                    _, reward, terminated, truncated, extras = env.step(adapted.action)
                    completed_steps = step_index + 1
                    reward_sum += float(reward[0].item())
                    done = bool((terminated[0] | truncated[0]).item())
                    if args_cli.trace_out is not None:
                        rollout_trace.append(
                            {
                                "episode": episode_index,
                                "step": step_index,
                                "inferred_new_chunk": sample.inferred_new_chunk,
                                "policy_action": sample.action.detach().cpu().reshape(-1).tolist(),
                                "state_before": observation["observation.state"]
                                .detach()
                                .cpu()
                                .reshape(-1)
                                .tolist(),
                                # ManagerBasedRLEnv auto-resets a finished environment, so
                                # this is the reset state when done=True. Consumers must omit
                                # it from controller tracking-error calculations.
                                "state_after": bridge.state().detach().cpu().reshape(-1).tolist(),
                                "manager_action": adapted.action.detach()
                                .cpu()
                                .reshape(-1)
                                .tolist(),
                                "saturation_fraction": adapted.saturation_fraction,
                                "terminated": bool(terminated[0].item()),
                                "truncated": bool(truncated[0].item()),
                                "state_after_is_auto_reset": done,
                            }
                        )
                    success = done and _episode_termination_value(extras, "success") > 0.5
                    if success:
                        max_progress = 1.0
                    if done:
                        reason = _termination_reason(
                            extras, success, bool(truncated[0].item())
                        )
                        break
            finally:
                if video_writer is not None:
                    video_writer.close()
                    written_video = str(video_path)
                    recorded_video_paths.append(written_video)
                    print(f"[VIDEO] Episode {episode_index:03d}: {written_video}")

            elapsed_s = completed_steps * env.step_dt
            episode_result: dict[str, object] = {
                "episode": episode_index,
                "seed": args_cli.seed + episode_index,
                "success": success,
                "reason": reason,
                "steps": completed_steps,
                "sim_time_s": elapsed_s,
                "reward_sum": reward_sum,
                "max_task_progress": max_progress,
                "mean_action_saturation": (
                    saturation_sum / completed_steps if completed_steps else 0.0
                ),
                "policy_inferences": len(inference_latencies),
                "mean_inference_ms": (
                    statistics.fmean(inference_latencies) if inference_latencies else 0.0
                ),
                "p95_inference_ms": percentile_nearest_rank(inference_latencies, 95.0),
            }
            if written_video is not None:
                episode_result["video_path"] = written_video
            episode_results.append(episode_result)
            all_inference_latencies.extend(inference_latencies)
            print(
                f"[EPISODE {episode_index:03d}] success={success}, reason={reason}, "
                f"steps={completed_steps}, sim_time={elapsed_s:.2f}s, "
                f"progress={max_progress:.3f}, reward={reward_sum:.3f}"
            )
    finally:
        runner.close()
        env.close()

    successes = sum(bool(item["success"]) for item in episode_results)
    confidence_low, confidence_high = wilson_score_interval(successes, len(episode_results))
    success_times = [
        float(item["sim_time_s"]) for item in episode_results if bool(item["success"])
    ]
    summary = {
        "episodes_requested": args_cli.episodes,
        "episodes_completed": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results) if episode_results else 0.0,
        "success_rate_ci95": {
            "low": confidence_low,
            "high": confidence_high,
            "method": "wilson",
        },
        "termination_reasons": termination_reason_counts(
            str(item["reason"]) for item in episode_results
        ),
        "mean_success_time_s": statistics.fmean(success_times) if success_times else None,
        "mean_reward": (
            statistics.fmean(float(item["reward_sum"]) for item in episode_results)
            if episode_results
            else 0.0
        ),
        "mean_max_task_progress": (
            statistics.fmean(float(item["max_task_progress"]) for item in episode_results)
            if episode_results
            else 0.0
        ),
        "policy_inferences": len(all_inference_latencies),
        "mean_inference_ms": (
            statistics.fmean(all_inference_latencies) if all_inference_latencies else 0.0
        ),
        "p95_inference_ms": percentile_nearest_rank(all_inference_latencies, 95.0),
    }
    payload = {
        "format": "kuavo_groot_n1_7_eval",
        "format_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_label,
        "robot_model": ACTIVE_ROBOT_MODEL.name,
        "task": args_cli.task,
        "state_mode": args_cli.state_mode,
        "action_mode": args_cli.action_mode,
        "action_clip": None if args_cli.action_clip == 0 else args_cli.action_clip,
        "joint_limit_clamp": args_cli.policy_profile == RWH_KUAVO_V2_S56_PROFILE,
        "initial_pose": args_cli.initial_pose,
        "initial_state_16d": ready_state,
        "initial_state_source": initial_state_source,
        "initial_state_dataset": (
            DATASET_MEDOID_SOURCE if args_cli.initial_pose == "dataset-medoid" else None
        ),
        "initial_state_q50": ready_state if args_cli.initial_pose == "checkpoint-q50" else None,
        "initial_head_pitch_deg": args_cli.initial_head_pitch_deg,
        "policy_profile": args_cli.policy_profile,
        "controlled_joint_names": list(CONTROLLED_JOINT_NAMES),
        "state_names": list(bridge.state_names),
        "gripper_preset": GRIPPER_SETTINGS.name,
        "gripper_sides": list(GRIPPER_SETTINGS.active_sides),
        "camera_map": camera_map,
        "domain_randomization": bool(args_cli.domain_randomization),
        "physics_hz": 1.0 / cfg.sim.dt,
        "control_hz": 1.0 / env.step_dt,
        "video": {
            "camera_keys": list(video_camera_keys),
            "layout": "horizontal",
            "scene_view": bool(scene_video_camera is not None),
            "scene_camera_eye": (
                list(args_cli.scene_camera_eye) if scene_video_camera is not None else None
            ),
            "scene_camera_target": (
                list(args_cli.scene_camera_target) if scene_video_camera is not None else None
            ),
            "fps": video_fps,
            "tile_height": args_cli.video_height,
            "paths": recorded_video_paths,
        },
        "summary": summary,
        "episodes": episode_results,
    }
    if args_cli.trace_out is not None:
        trace_path = args_cli.trace_out.expanduser().resolve()
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {
                    "format": "kuavo_groot_rollout_trace",
                    "format_version": 1,
                    "state_names": list(bridge.state_names),
                    "policy_action_names": list(bridge.action_names),
                    "manager_action_names": [
                        "waist_yaw",
                        *(f"zarm_l{index}" for index in range(1, 8)),
                        *(f"zarm_r{index}" for index in range(1, 8)),
                        "left_gripper",
                        "right_gripper",
                    ],
                    "steps": rollout_trace,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
        payload["trace_path"] = str(trace_path)
        print(f"[TRACE] Per-step rollout trace: {trace_path}")
    metrics_path = (args_cli.metrics_out or _default_metrics_path()).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[RESULT] success_rate={summary['success_rate']:.3f} "
        f"({successes}/{len(episode_results)}), metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
