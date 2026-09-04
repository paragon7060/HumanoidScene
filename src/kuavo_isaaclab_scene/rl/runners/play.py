"""Evaluate a skill and optionally collect successful states for the next skill."""

import json
from .common import parse_args, build_configs, run_directory, write_run_config, check_checkpoint


def main():
    args = parse_args("play")
    from isaaclab.app import AppLauncher
    app = AppLauncher(args).app
    env = None
    try:
        import torch
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner
        cfg, agent = build_configs(args)
        directory = run_directory(args, "play")
        env = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg), clip_actions=agent.clip_actions)
        manifest = write_run_config(directory, cfg, agent, env.unwrapped)
        check_checkpoint(args.checkpoint, manifest)
        runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=agent.device)
        runner.load(str(args.checkpoint.expanduser().resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=env.device)
        observations = env.get_observations()
        outcomes = []
        while app.is_running() and len(outcomes) < args.episodes:
            with torch.inference_mode():
                observations, _, _, _ = env.step(policy(observations))
            latest = getattr(env.unwrapped, "_rl_last_outcomes", {})
            if latest.get("step") == env.unwrapped.common_step_counter:
                outcomes.extend(latest["episodes"])
        outcomes = outcomes[:args.episodes]
        successes = sum(o["success"] for o in outcomes)
        report = {"task": args.task, "requested_episodes": args.episodes, "episodes": len(outcomes),
                  "successes": successes, "success_rate": successes / len(outcomes) if outcomes else None,
                  "outcomes": outcomes}
        (directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
        print(f"[RL] {successes}/{len(outcomes)} succeeded; metrics={directory / 'metrics.json'}", flush=True)
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == "__main__":
    main()
