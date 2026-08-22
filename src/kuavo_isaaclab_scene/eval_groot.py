#!/usr/bin/env python3
"""Evaluate a LeRobot GR00T N1.7 policy in the Kuavo Isaac Lab workcell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import sys

from isaaclab.app import AppLauncher

from .gripper_config import add_gripper_cli_args, export_gripper_cli, resolve_gripper_settings
from .rack_box_layout import resolve_rack_box_pose_path
from .robot_material_config import (
    add_robot_material_cli_args,
    export_robot_material_cli,
    resolve_robot_material_settings,
)


parser = argparse.ArgumentParser(
    description="Run serial GR00T N1.7 rollouts in the Kuavo rack-to-conveyor task."
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Local LeRobot pretrained_model directory or Hugging Face repository ID.",
)
parser.add_argument("--episodes", type=int, default=10)
parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Maximum control steps per episode; 0 uses the environment time limit.",
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
    help="Decoded chunk actions to execute before re-planning (default: checkpoint n_action_steps).",
)
parser.add_argument(
    "--state-mode",
    choices=("manager", "joint_position"),
    default="manager",
    help="Representation used by observation.state; it must match the training dataset.",
)
parser.add_argument(
    "--action-mode",
    choices=("manager", "joint_position", "joint_delta"),
    default="manager",
    help="Representation of the decoded policy action; it must match the training dataset.",
)
parser.add_argument(
    "--action-clip",
    type=float,
    default=1.0,
    help="Clamp manager actions to +/-VALUE; use 0 to disable clipping.",
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

add_gripper_cli_args(parser)
add_robot_material_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
export_gripper_cli(args_cli)
export_robot_material_cli(args_cli)

if not args_cli.mock_policy and not args_cli.checkpoint:
    parser.error("--checkpoint is required unless --mock-policy is selected.")
if args_cli.episodes <= 0:
    parser.error("--episodes must be positive.")
if args_cli.max_steps < 0:
    parser.error("--max-steps must be zero or positive.")
if args_cli.actions_per_inference is not None and args_cli.actions_per_inference <= 0:
    parser.error("--actions-per-inference must be positive.")
if args_cli.action_clip < 0:
    parser.error("--action-clip must be zero or positive.")
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
    GRIPPER_SETTINGS = resolve_gripper_settings()
    ROBOT_MATERIAL_SETTINGS = resolve_robot_material_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))
if captured_pose_path is not None:
    os.environ["KUAVO_RACK_BOX_POSES"] = str(captured_pose_path)

args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

from isaaclab.envs import ManagerBasedRLEnv

from . import manager_mdp as workcell_mdp
from .camera_viewports import open_camera_viewports
from .groot_lerobot_bridge import (
    CONTROLLED_JOINT_NAMES,
    KuavoLeRobotBridge,
    LeRobotGrootRunner,
    ZeroLeRobotPolicyRunner,
    parse_camera_map,
)
from .manager_env import (
    ACTIVE_RACK_BOX_SCENE_KEYS,
    KuavoRobustWorkcellEnvCfg,
    TASK_OBJECT_PARAMS,
)


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
    for name in ("cargo_spill", "tote_drop", "human_or_robot_contact"):
        if _episode_termination_value(extras, name) > 0.5:
            return name
    return "time_limit" if truncated else "terminated"


def _percentile_95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _default_metrics_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "artifacts" / "eval" / f"groot_eval_{stamp}.json"


def main() -> None:
    try:
        camera_map = parse_camera_map(args_cli.camera_map)
    except ValueError as exc:
        parser.error(str(exc))

    cfg = KuavoRobustWorkcellEnvCfg()
    cfg.scene.num_envs = 1
    cfg.seed = args_cli.seed
    cfg.curriculum = None
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
    if not args_cli.domain_randomization:
        _disable_domain_randomization(cfg)

    env = ManagerBasedRLEnv(cfg=cfg)
    bridge = KuavoLeRobotBridge(
        env,
        camera_map=camera_map,
        state_mode=args_cli.state_mode,
        action_mode=args_cli.action_mode,
        action_clip=None if args_cli.action_clip == 0 else args_cli.action_clip,
    )
    if args_cli.camera_preview:
        open_camera_viewports(
            env.scene,
            list(dict.fromkeys(camera_map.values())),
            headless=args_cli.headless,
        )

    policy_device = args_cli.policy_device or args_cli.device
    if args_cli.mock_policy:
        runner = ZeroLeRobotPolicyRunner(action_dim=bridge.action_dim)
        checkpoint_label = "mock:zero"
    else:
        print(f"[INFO] Loading GR00T N1.7 checkpoint {args_cli.checkpoint!r} on {policy_device}.")
        runner = LeRobotGrootRunner.from_pretrained(
            args_cli.checkpoint,
            device=policy_device,
            actions_per_inference=args_cli.actions_per_inference,
            local_files_only=args_cli.local_files_only,
            expected_action_dim=bridge.action_dim,
        )
        checkpoint_label = args_cli.checkpoint

    max_steps = args_cli.max_steps or math.ceil(cfg.episode_length_s / env.step_dt)
    print(
        f"[INFO] Eval ready: episodes={args_cli.episodes}, max_steps={max_steps}, "
        f"control_hz={1.0 / env.step_dt:.1f}, task_boxes={ACTIVE_RACK_BOX_SCENE_KEYS or 'legacy_totes'}"
    )
    manager_action_names = (*CONTROLLED_JOINT_NAMES, *(f"{side}_gripper" for side in GRIPPER_SETTINGS.active_sides))
    print(f"[INFO] State schema ({len(bridge.state_names)}): {', '.join(bridge.state_names)}")
    print(f"[INFO] Policy action schema ({bridge.action_dim}): {', '.join(manager_action_names[:bridge.action_dim])}")
    print(f"[INFO] Gripper preset: {GRIPPER_SETTINGS.name}")
    print(f"[INFO] Camera map: {camera_map}")

    episode_results: list[dict[str, object]] = []
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

            for step_index in range(max_steps):
                if not simulation_app.is_running():
                    reason = "simulation_closed"
                    break
                progress = workcell_mdp.task_progress(env, **TASK_OBJECT_PARAMS)
                max_progress = max(max_progress, float(progress[0].item()))
                observation = bridge.observation(args_cli.task)
                sample = runner.select_action(observation)
                if sample.inferred_new_chunk:
                    inference_latencies.append(sample.inference_ms)
                adapted = bridge.action(sample.action)
                saturation_sum += adapted.saturation_fraction
                _, reward, terminated, truncated, extras = env.step(adapted.action)
                completed_steps = step_index + 1
                reward_sum += float(reward[0].item())
                done = bool((terminated[0] | truncated[0]).item())
                success = done and _episode_termination_value(extras, "success") > 0.5
                if success:
                    max_progress = 1.0
                if done:
                    reason = _termination_reason(extras, success, bool(truncated[0].item()))
                    break

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
                "p95_inference_ms": _percentile_95(inference_latencies),
            }
            episode_results.append(episode_result)
            print(
                f"[EPISODE {episode_index:03d}] success={success}, reason={reason}, "
                f"steps={completed_steps}, sim_time={elapsed_s:.2f}s, "
                f"progress={max_progress:.3f}, reward={reward_sum:.3f}"
            )
    finally:
        env.close()

    successes = sum(bool(item["success"]) for item in episode_results)
    success_times = [
        float(item["sim_time_s"]) for item in episode_results if bool(item["success"])
    ]
    summary = {
        "episodes_requested": args_cli.episodes,
        "episodes_completed": len(episode_results),
        "successes": successes,
        "success_rate": successes / len(episode_results) if episode_results else 0.0,
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
    }
    payload = {
        "format": "kuavo_groot_n1_7_eval",
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint_label,
        "task": args_cli.task,
        "state_mode": args_cli.state_mode,
        "action_mode": args_cli.action_mode,
        "action_clip": None if args_cli.action_clip == 0 else args_cli.action_clip,
        "controlled_joint_names": list(CONTROLLED_JOINT_NAMES),
        "state_names": list(bridge.state_names),
        "gripper_preset": GRIPPER_SETTINGS.name,
        "gripper_sides": list(GRIPPER_SETTINGS.active_sides),
        "camera_map": camera_map,
        "domain_randomization": bool(args_cli.domain_randomization),
        "summary": summary,
        "episodes": episode_results,
    }
    metrics_path = (args_cli.metrics_out or _default_metrics_path()).expanduser().resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"[RESULT] success_rate={summary['success_rate']:.3f} "
        f"({successes}/{len(episode_results)}), metrics={metrics_path}"
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
