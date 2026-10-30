"""Isaac bootstrap and routing for SAC, diffusion data collection and DPPO."""

import json
import os
import re
import traceback
from .common import parse_args, build_configs, run_directory, write_run_config, check_checkpoint, install_stop_handlers


def add_arguments(parser):
    parser.add_argument("--method", choices=("sac", "dppo", "collect", "play"), required=True)
    parser.add_argument("--rollout-steps", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--replay-capacity", type=int, default=1000000, help="Total transitions, not per-env capacity")
    parser.add_argument("--replay-device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--learning-starts", type=int, default=100000, help="Fresh transitions, repeated after SAC resume")
    parser.add_argument("--updates-per-step", type=int, default=8, help="SAC minibatches per vector environment step")
    parser.add_argument("--epochs", type=int, default=5, help="DPPO optimization epochs")
    parser.add_argument("--critic-warmup", type=int, default=5, help="DPPO iterations updating only value function")
    parser.add_argument("--keep-checkpoints", type=int, default=2)
    parser.add_argument("--collect-max-steps", type=int, default=10000, help="Bound collection duration in vector steps")
    parser.add_argument("--stochastic-eval", action="store_true", help="Sample diffusion chains instead of deterministic mean paths")


def validate_args(args):
    for key in ("rollout_steps", "batch_size", "replay_capacity", "updates_per_step", "epochs",
                "keep_checkpoints", "collect_max_steps"):
        if getattr(args, key) < 1:
            raise ValueError(f"--{key.replace('_', '-')} must be positive")
    if min(args.learning_starts, args.critic_warmup) < 0:
        raise ValueError("Warmup counts must be nonnegative")
    if args.method in ("dppo", "collect", "play") and args.checkpoint is None:
        raise ValueError(f"--method {args.method} requires --checkpoint")
    if args.method == "collect" and args.num_envs > 64:
        raise ValueError("Collect demonstrations with <=64 envs")
    mask = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not re.fullmatch(r"[0-9]+", mask) or args.device not in (None, "cuda:0"):
        raise ValueError("Set CUDA_VISIBLE_DEVICES to ONE physical GPU index and use --device cuda:0")
    if "renderer/" in (args.kit_args or ""):
        raise ValueError("Renderer GPU flags are managed by this runner; remove them from --kit_args")
    args.device = "cuda:0"
    args.kit_args = (args.kit_args or "") + (f" --/renderer/activeGpu={mask}"
        " --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false")
    args.save_interval = args.save_interval or 1000


def main():
    args = parse_args("train", add_arguments)
    validate_args(args)
    # Validate checkpoint kind before paying the simulator startup cost.
    state = None
    if args.checkpoint and args.method != "collect":
        from .storage import load_checkpoint
        state = load_checkpoint(args.checkpoint)
        allowed = {"sac": ("sac",), "dppo": ("diffusion_bc", "dppo"), "play": ("sac", "diffusion_bc", "dppo")}
        if state.get("algorithm") not in allowed[args.method]:
            raise ValueError(f"Checkpoint {state.get('algorithm')} cannot be used for {args.method}")
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    install_stop_handlers()
    env = None
    directory = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from ..envs.terminal_observation import TerminalObservationMixin
        torch.manual_seed(args.seed)
        torch.set_num_threads(min(8, torch.get_num_threads()))
        class TransitionEnv(TerminalObservationMixin, ManagerBasedRLEnv):
            pass
        cfg, ppo_cfg = build_configs(args)
        if cfg.is_finite_horizon:
            raise ValueError("Alternative runners currently require the shared infinite-horizon timeout convention")
        directory = run_directory(args, args.method)
        cfg.log_dir = str(directory)
        env = TransitionEnv(cfg)
        settings = json.loads(json.dumps(vars(args), default=str))
        manifest = write_run_config(directory, cfg, settings, env)
        if args.checkpoint:
            check_checkpoint(args.checkpoint, manifest)
        print(f"[RL] method={args.method}; artifacts={directory}; CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}", flush=True)
        if args.method == "sac":
            from .train_sac import train
            train(env, args, directory, state)
        elif args.method == "dppo":
            from .train_dppo import train
            train(env, args, directory, state)
        elif args.method == "collect":
            from .collect_diffusion import collect
            collect(env, args, directory, manifest, ppo_cfg)
        else:
            from .evaluate_alternative import evaluate
            evaluate(env, args, directory, state)
        (directory / "status.json").write_text(json.dumps({"status": "complete", "method": args.method}))
    except BaseException as error:
        if directory is not None:
            (directory / "status.json").write_text(json.dumps(
                {"status": "failed", "method": args.method, "error": str(error), "traceback": traceback.format_exc()}, indent=2))
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
