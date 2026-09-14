"""PPO launch/evaluation shared by opt-in staged and end-to-end experiments."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import runpy
import traceback
from uuid import uuid4
from .spec import MultiBoxSpec, SKILLS
from ..action_spaces import add_action_space_argument


def canonical(value):
    if isinstance(value, dict):
        return {k: canonical(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [canonical(v) for v in value]
    if callable(value):
        return value.__module__ + ":" + value.__qualname__
    if isinstance(value, Path):
        return str(value)
    return value


def digest(value):
    return hashlib.sha256(json.dumps(canonical(value), sort_keys=True, default=str).encode()).hexdigest()


def main(strategy=None, mode="train"):
    from isaaclab.app import AppLauncher
    from ...robots.robot_model import add_robot_model_cli_args, export_robot_model_cli
    from ...robots.gripper_config import add_gripper_cli_args, export_gripper_cli
    from ...robots.initial_states import add_initial_state_args
    from ...workcell.rack_rollers import add_rack_roller_cli_args, export_rack_roller_cli
    from ...core.paths import CONFIG_DIR

    parser = argparse.ArgumentParser(description="Four-box whole-body PPO", allow_abbrev=False)
    add_action_space_argument(parser)
    parser.add_argument("--strategy", choices=("staged", "end-to-end"), default=strategy or "end-to-end")
    parser.add_argument("--skill", choices=(*SKILLS, "full"), default="pick" if strategy == "staged" else "full")
    parser.add_argument("--num-envs", type=int, default=256 if mode == "train" else 1)
    parser.add_argument("--env-spacing", type=float, default=8.)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-bank", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--skill-checkpoints", type=Path, help="JSON mapping pick/extract/carry/place to PPO checkpoints")
    parser.add_argument("--config", type=Path, help="Trusted Python: configure_spec(spec), configure(env, agent)")
    parser.add_argument("--workcell-layout", type=Path, default=CONFIG_DIR / "workcell_layout.json")
    parser.add_argument("--rack-box-poses", type=Path, default=CONFIG_DIR / "rack_box_poses.json")
    parser.add_argument("--log-dir", type=Path, default=Path("artifacts/rl/multi_box"))
    add_robot_model_cli_args(parser)
    add_gripper_cli_args(parser)
    add_initial_state_args(parser)
    add_rack_roller_cli_args(parser)
    parser.set_defaults(robot_model="s200062", gripper="s200062_integrated")
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    for field in ("workcell_layout", "rack_box_poses", "config", "checkpoint", "skill_checkpoints", "initial_states_file"):
        path = getattr(args, field)
        if path is not None and not path.expanduser().is_file():
            parser.error(f"Missing --{field.replace('_', '-')}: {path}")
        if path is not None:
            setattr(args, field, path.expanduser().resolve())
    if strategy and args.strategy != strategy:
        parser.error("Use the matching strategy entrypoint.")
    if min(args.num_envs, args.iterations, args.episodes) < 1 or args.env_spacing < 5:
        parser.error("Counts must be positive and env spacing >= 5 m.")
    if args.enable_cameras:
        parser.error("Four-box baseline uses state observations; omit --enable_cameras.")
    if mode == "train" and args.strategy == "staged" and args.skill == "full":
        parser.error("Train each skill separately, then evaluate with --skill full --skill-checkpoints.")
    if mode == "eval" and not (args.checkpoint or args.skill_checkpoints):
        parser.error("Evaluation needs --checkpoint or --skill-checkpoints.")
    if args.skill_checkpoints and (mode != "eval" or args.strategy != "staged" or args.skill != "full" or args.checkpoint):
        parser.error("--skill-checkpoints requires staged full evaluation without --checkpoint.")
    if mode == "eval" and args.strategy == "staged" and args.skill == "full" and not args.skill_checkpoints:
        parser.error("Full staged evaluation requires all four skill checkpoints.")
    if args.reset_bank and args.initial_state:
        parser.error("Reset bank already supplies the robot pose; omit --initial-state.")
    export_robot_model_cli(args)
    export_gripper_cli(args)
    export_rack_roller_cli(args)
    os.environ["KUAVO_WORKCELL_LAYOUT"] = str(args.workcell_layout.resolve())
    os.environ["KUAVO_RACK_BOX_POSES"] = str(args.rack_box_poses.resolve())
    customization = runpy.run_path(str(args.config)) if args.config else {}
    spec = MultiBoxSpec(strategy=args.strategy, skill=args.skill,
        reset_bank=str(args.reset_bank.resolve()) if args.reset_bank else None,
        snapshot_dir=str(args.snapshot_dir.resolve()) if args.snapshot_dir else None,
        episode_seconds=120. if args.skill == "full" else 30.)
    spec = customization.get("configure_spec", lambda s: s)(spec)
    if args.action_space is not None:
        spec = replace(spec, action_space=args.action_space)
    spec.validate()
    if (spec.strategy, spec.skill) != (args.strategy, args.skill):
        parser.error("Set strategy/skill via CLI, not configure_spec.")
    # Numeric mask maps physical CUDA device to logical cuda:0. Renderer indices remain physical.
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask:
        if not mask.isdigit() or args.device not in (None, "cuda:0"):
            parser.error("Use one numeric CUDA_VISIBLE_DEVICES index with --device cuda:0.")
        if "renderer/" in (args.kit_args or ""):
            parser.error("Runner manages renderer GPU flags when CUDA_VISIBLE_DEVICES is set.")
        args.kit_args = (args.kit_args or "") + f" --/renderer/activeGpu={mask} --/renderer/multiGpu/enabled=false"
    directory = args.log_dir.resolve() / f"{args.strategy}_{args.skill}_{mode}_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:6]}"
    directory.mkdir(parents=True)
    app = AppLauncher(args).app
    env = None
    raw = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab.utils.io import dump_yaml
        from ...robots.initial_states import configure_initial_state
        from ...robots.end_effector import calibration_definition
        from ...robots.gripper_config import resolve_gripper_settings
        from ..agents.ppo_cfg import WorkcellPPOCfg
        from .env_cfg import MultiBoxEnvCfg
        from .reset_states import read_bank
        cfg = MultiBoxEnvCfg(multi_box=spec, num_envs=args.num_envs, env_spacing=args.env_spacing)
        cfg.sim.device = args.device or "cuda:0"
        cfg.seed = args.seed
        cfg.log_dir = str(directory)
        agent = WorkcellPPOCfg(seed=args.seed, device=cfg.sim.device, max_iterations=args.iterations,
                              experiment_name="multi_box", save_interval=100)
        agent.algorithm.gamma = spec.discount
        configure_initial_state(cfg, args)
        if "configure" in customization:
            customization["configure"](cfg, agent)
        if agent.algorithm.gamma != spec.discount:
            raise ValueError("PPO gamma must equal multi_box.discount for potential shaping.")
        # Strategy, stage, horizon and reset source may differ; physics and schema may not.
        physical_spec = asdict(spec)
        for key in ("strategy", "skill", "reset_bank", "snapshot_dir", "max_snapshots", "episode_seconds"):
            physical_spec.pop(key)
        contract = dict(version=1, spec=physical_spec, robot=args.robot_model, gripper=args.gripper,
            actions=cfg.actions.to_dict(), observations=cfg.observations.to_dict(),
            layout=json.loads(args.workcell_layout.read_text()), boxes=json.loads(args.rack_box_poses.read_text()),
            robot_spawn=cfg.scene.robot.spawn.to_dict(), robot_actuators=cfg.scene.robot.actuators,
            geometry={n: asdict(g) for n, g in cfg.commands.workcell.geometry.items()},
            tcp=calibration_definition(), gripper_config=json.loads(resolve_gripper_settings().config_path.read_text()),
            sim_dt=cfg.sim.dt, decimation=cfg.decimation)
        # Serialize config objects canonically before hashing.
        contract["robot_actuators"] = {k: v.to_dict() for k, v in cfg.scene.robot.actuators.items()}
        cfg.experiment_contract = digest(contract)
        bank = read_bank(spec.reset_bank, cfg.experiment_contract, spec.skill) if spec.reset_bank else None
        raw = ManagerBasedRLEnv(cfg)
        raw._multi_box_bank = bank
        env = RslRlVecEnvWrapper(raw, clip_actions=agent.clip_actions)
        manifest = dict(contract=cfg.experiment_contract, strategy=args.strategy, skill=args.skill,
                        spec=asdict(spec), arguments=vars(args), compatibility=contract)
        (directory / "manifest.json").write_text(json.dumps(canonical(manifest), indent=2, default=str))
        dump_yaml(str(directory / "env.yaml"), cfg)
        dump_yaml(str(directory / "agent.yaml"), agent)

        def load(path, expected_skill):
            path = Path(path).expanduser().resolve()
            metadata = json.loads((path.parent / "manifest.json").read_text())
            if (metadata["contract"] != cfg.experiment_contract or metadata["skill"] != expected_skill
                    or metadata["strategy"] != args.strategy):
                raise ValueError(f"Checkpoint contract/strategy/skill mismatch: {path}")
            runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(directory) if mode == "train" else None,
                                    device=agent.device)
            runner.load(str(path), load_optimizer=mode == "train", map_location=agent.device)
            return runner

        print(f"[MULTI BOX] {args.strategy}/{args.skill}: {directory}", flush=True)
        if mode == "train":
            runner = load(args.checkpoint, args.skill) if args.checkpoint else OnPolicyRunner(
                env, agent.to_dict(), log_dir=str(directory), device=agent.device)
            runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        else:
            if args.skill_checkpoints:
                mapping = json.loads(args.skill_checkpoints.read_text())
                if set(mapping) != set(SKILLS):
                    raise ValueError(f"skill-checkpoints must contain exactly {SKILLS}")
                paths = {k: (args.skill_checkpoints.parent / v).resolve() for k, v in mapping.items()}
                policies = [load(paths[k], k).get_inference_policy(device=raw.device) for k in SKILLS]
            else:
                policies = [load(args.checkpoint, args.skill).get_inference_policy(device=raw.device)]
            obs = env.get_observations()
            outcomes = []
            with torch.inference_mode():
                while len(outcomes) < args.episodes and app.is_running():
                    if len(policies) == 1:
                        action = policies[0](obs)
                    else:
                        candidates = torch.stack([p(obs) for p in policies], 1)
                        t = raw.command_manager.get_term("workcell")
                        action = candidates[t.ids, t.route]
                    obs, _, _, _ = env.step(action)
                    latest = getattr(raw, "_multi_box_outcomes", {})
                    if latest.get("step") == raw.common_step_counter:
                        outcomes.extend(latest["episodes"])
            outcomes = outcomes[:args.episodes]
            report = dict(episodes=len(outcomes), requested_episodes=args.episodes,
                success_rate=sum(x["success"] for x in outcomes) / len(outcomes) if outcomes else None,
                outcomes=outcomes)
            (directory / "metrics.json").write_text(json.dumps(report, indent=2))
            if len(outcomes) < args.episodes:
                raise RuntimeError("Simulation closed before evaluation completed.")
            print(json.dumps(report), flush=True)
        (directory / "status.json").write_text(json.dumps({"status": "complete"}))
    except BaseException:
        diagnostic = traceback.format_exc()
        (directory / "status.json").write_text(json.dumps({"status": "failed", "traceback": diagnostic}))
        print(diagnostic, flush=True)  # Kit close can suppress the outer traceback.
        raise
    finally:
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        app.close()
