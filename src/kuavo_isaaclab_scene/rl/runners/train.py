"""Train or resume one selected skill with RSL-RL PPO."""

from .common import parse_args, build_configs, run_directory, write_run_config, check_checkpoint, install_stop_handlers
from .checkpoints import AtomicCheckpointMixin
import json


def main():
    args = parse_args("train")
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    install_stop_handlers()
    env = None
    runner = None
    directory = None
    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        class DriveSafeRunner(AtomicCheckpointMixin, OnPolicyRunner):
            pass
        cfg, agent = build_configs(args)
        directory = run_directory(args, "train")
        cfg.log_dir = str(directory)
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent.clip_actions)
        manifest = write_run_config(directory, cfg, agent, env.unwrapped)
        if args.checkpoint:
            check_checkpoint(args.checkpoint, manifest)
        runner = DriveSafeRunner(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        if args.checkpoint:
            # Load only your own trusted RSL-RL checkpoints (PyTorch serialization).
            runner.load(str(args.checkpoint.expanduser().resolve()))
        print(f"[RL] task={args.task}; log/checkpoints={directory}", flush=True)
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
        (directory / "status.json").write_text(json.dumps({"status": "complete",
            "last_iteration": runner.current_learning_iteration}))
    except BaseException as error:
        if directory is not None:
            (directory / "status.json").write_text(json.dumps({"status": "failed",
                "error": repr(error)}))
        raise
    finally:
        try:
            if runner is not None and runner.writer is not None:
                runner.writer.close()
        finally:
            try:
                if env is not None:
                    env.close()
            finally:
                app.close()


if __name__ == "__main__":
    main()
