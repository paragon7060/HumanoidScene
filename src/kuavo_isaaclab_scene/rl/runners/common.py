"""CLI/bootstrap helpers. No Isaac-dependent environment imports at module scope."""

import argparse
from dataclasses import asdict
from datetime import datetime
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
from uuid import uuid4

from ..tasks.specs import TASKS, PREDECESSOR, REQUIRES_RESET_BANK, task_spec
from ...robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
from ...robots.gripper_config import add_gripper_cli_args, export_gripper_cli
from ...core.paths import default_artifacts_dir


def parse_args(mode):
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description=f"Kuavo manager-based subtask PPO {mode}")
    parser.add_argument("--task", choices=TASKS, default="approach_rack")
    parser.add_argument("--boxes", default="small_box_0", help="Ordered comma-separated scene keys, or all captured rack boxes")
    parser.add_argument("--num-envs", type=int, default=8 if mode == "train" else 1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--reset-bank", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--max-snapshots", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, required=mode == "play")
    parser.add_argument("--config", type=Path, help="Trusted Python customization: configure_task(spec), configure(env_cfg, agent_cfg)")
    parser.add_argument("--prefill", type=int, default=1)
    parser.add_argument("--slots", type=int, default=4)
    parser.add_argument("--slot-pitch", type=float, default=0.52)
    parser.add_argument("--cargo-per-box", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--no-randomization", action="store_true")
    parser.add_argument("--workcell-layout", type=Path)
    parser.add_argument("--rack-box-poses", type=Path)
    parser.add_argument("--rack-boxes")
    parser.add_argument("--ignore-captured-box-poses", action="store_true")
    parser.add_argument("--log-dir", type=Path)
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if min(args.num_envs, args.max_iterations, args.episodes, args.max_snapshots) < 1:
        parser.error("Environment/iteration/episode/snapshot counts must be positive.")
    if args.task in REQUIRES_RESET_BANK and not args.reset_bank:
        parser.error(f"{args.task} needs --reset-bank with successful {PREDECESSOR[args.task]} states.")
    for name in ("checkpoint", "reset_bank", "config", "workcell_layout", "rack_box_poses"):
        value = getattr(args, name)
        if value and not value.expanduser().exists():
            parser.error(f"Missing --{name.replace('_', '-')}: {value}")
        if value:
            setattr(args, name, value.expanduser().resolve())
    export_robot_model_cli(args)
    export_gripper_cli(args)
    for flag, key in (("workcell_layout", "KUAVO_WORKCELL_LAYOUT"), ("rack_box_poses", "KUAVO_RACK_BOX_POSES")):
        if getattr(args, flag):
            os.environ[key] = str(getattr(args, flag).expanduser().resolve())
    if args.rack_boxes:
        os.environ["KUAVO_RACK_BOXES"] = args.rack_boxes
        os.environ.pop("KUAVO_RACK_BOX_LAYOUT", None)
    if args.ignore_captured_box_poses:
        os.environ["KUAVO_IGNORE_RACK_BOX_POSES"] = "1"
    elif args.rack_box_poses:
        os.environ.pop("KUAVO_IGNORE_RACK_BOX_POSES", None)
    try:
        version = importlib.metadata.version("rsl-rl-lib")
    except importlib.metadata.PackageNotFoundError:
        parser.error("Install the RL extra in the Isaac Lab environment: python -m pip install -e '.[rl]'")
    if version != "3.1.2":
        parser.error(f"Expected rsl-rl-lib==3.1.2 for this runner, found {version}; install '.[rl]'.")
    return args


def build_configs(args):
    from ..tasks.env_cfg import WorkcellRLEnvCfg
    from ..agents.ppo_cfg import WorkcellPPOCfg
    from ...envs.manager_env import ACTIVE_RACK_BOX_SCENE_KEYS
    box_names = ACTIVE_RACK_BOX_SCENE_KEYS if args.boxes == "all" else tuple(s.strip() for s in args.boxes.split(",") if s.strip())
    spec = task_spec(args.task, box_names=box_names,
        reset_bank=str(args.reset_bank.expanduser().resolve()) if args.reset_bank else None,
        snapshot_dir=str(args.snapshot_dir.expanduser().resolve()) if args.snapshot_dir else None,
        max_snapshots=args.max_snapshots, prefill_count=args.prefill, cargo_per_box=args.cargo_per_box,
        slot_count=args.slots, slot_pitch=args.slot_pitch,
        randomization=not args.no_randomization)
    customization = runpy.run_path(str(args.config.expanduser().resolve())) if args.config else {}
    if "configure_task" in customization:
        spec = customization["configure_task"](spec)
    cfg = WorkcellRLEnvCfg(task=spec, num_envs=args.num_envs, cameras=args.enable_cameras)
    cfg.seed = args.seed
    cfg.sim.device = args.device or "cuda:0"
    agent = WorkcellPPOCfg(seed=args.seed, device=cfg.sim.device, max_iterations=args.max_iterations,
                           experiment_name=f"kuavo_{args.task}")
    if "configure" in customization:
        customization["configure"](cfg, agent)
    if cfg.commands.workcell.task != cfg.task:
        raise ValueError("Change task fields in configure_task(), not after environment assembly.")
    if cfg.scene.num_envs * agent.num_steps_per_env < agent.algorithm.num_mini_batches:
        raise ValueError("PPO has more mini-batches than rollout samples.")
    return cfg, agent


def run_directory(args, mode):
    parent = args.log_dir.expanduser().resolve() if args.log_dir else default_artifacts_dir() / "rl" / args.task
    path = parent / f"{mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_run_config(path, cfg, agent, env):
    from isaaclab.utils.io import dump_yaml
    from ..mdp.reset_bank import contract, signature
    metadata = contract(env)
    action_schema = {n: env.action_manager.get_term(n).action_dim for n in env.action_manager.active_terms}
    manifest = {"version": 1, "task": asdict(cfg.task), "contract": metadata,
                "contract_hash": signature(metadata), "actions": action_schema,
                "action_config": cfg.actions.to_dict(),
                "observations": env.observation_manager.group_obs_dim,
                "robot_model": metadata["robot"], "gripper": metadata["gripper"]}
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False))
    dump_yaml(str(path / "env.yaml"), cfg)
    dump_yaml(str(path / "agent.yaml"), agent)
    return manifest


def check_checkpoint(checkpoint, manifest):
    metadata_path = checkpoint.parent / "manifest.json"
    if not metadata_path.is_file():
        raise ValueError(f"Checkpoint needs its manifest.json beside it: {metadata_path}")
    source = json.loads(metadata_path.read_text())
    for key in ("contract_hash", "actions", "action_config", "observations"):
        # JSON-normalize tuple/list representations before comparison.
        expected = json.loads(json.dumps(manifest[key]))
        if source.get(key) != expected:
            raise ValueError(f"Checkpoint {key} differs from this environment. Reuse its robot/box/config settings.")
    if source["task"]["name"] != manifest["task"]["name"]:
        raise ValueError("Checkpoint belongs to another skill. Train each skill separately; do not silently resume it.")
