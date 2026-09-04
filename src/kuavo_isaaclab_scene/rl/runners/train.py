"""Train or resume one selected skill with RSL-RL PPO."""

from .common import parse_args, build_configs, run_directory, write_run_config, check_checkpoint


def main():
    args = parse_args("train")
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        cfg, agent = build_configs(args)
        directory = run_directory(args, "train")
        cfg.log_dir = str(directory)
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent.clip_actions)
        manifest = write_run_config(directory, cfg, agent, env.unwrapped)
        if args.checkpoint:
            check_checkpoint(args.checkpoint, manifest)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=str(directory), device=agent.device)
        if args.checkpoint:
            # Load only your own trusted RSL-RL checkpoints (PyTorch serialization).
            runner.load(str(args.checkpoint.expanduser().resolve()))
        print(f"[RL] task={args.task}; log/checkpoints={directory}", flush=True)
        runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=False)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
