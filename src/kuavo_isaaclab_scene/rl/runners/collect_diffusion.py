"""Collect complete successful RSL-PPO trajectories for native diffusion BC."""

import json
import numpy as np
import torch
from ..data.episodes import EpisodeWriter


def collect(env, args, directory, manifest, ppo_cfg):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    from tensordict import TensorDict
    from ..mdp.settling import gate_actions
    # Limit RAM to complete episodes from a small evaluation batch.
    if env.num_envs > 64:
        raise ValueError("Demonstration collection supports at most 64 envs; use 2-16")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=1.0)
    runner = OnPolicyRunner(wrapped, ppo_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(str(args.checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)
    obs = wrapped.reset()[0]
    histories = [[] for _ in range(env.num_envs)]
    writer = EpisodeWriter(directory / "episodes.hdf5", manifest)
    attempted = 0
    try:
        for _ in range(args.collect_max_steps):
            with torch.no_grad():
                action = gate_actions(env, policy(obs).clamp(-1, 1))
                before, sent = obs["policy"].cpu().numpy().copy(), action.cpu().numpy().copy()
                next_dict, _, terminated, truncated, _ = env.step(action)
                obs = TensorDict(next_dict, batch_size=[env.num_envs])
                for index in range(env.num_envs):
                    histories[index].append((before[index], sent[index]))
                latest = getattr(env, "_rl_last_outcomes", {})
                successes = ({o["env_id"]: o["success"] for o in latest.get("episodes", [])}
                             if latest.get("step") == env.common_step_counter else {})
                for index in (terminated | truncated).nonzero().flatten().tolist():
                    attempted += 1
                    if successes.get(index, False) and writer.count < args.episodes:
                        sequence = histories[index]
                        writer.add(np.stack([x[0] for x in sequence]), np.stack([x[1] for x in sequence]), True)
                        print(f"[BC] Successful episodes: {writer.count}/{args.episodes}", flush=True)
                    histories[index].clear()
            if writer.count >= args.episodes:
                break
        report = dict(successful_episodes=writer.count, attempted_episodes=attempted,
                      requested_episodes=args.episodes, complete=writer.count >= args.episodes)
        (directory / "collection.json").write_text(json.dumps(report, indent=2))
        if writer.count < args.episodes:
            raise RuntimeError(f"Collected {writer.count}/{args.episodes} successes within --collect-max-steps; "
                               "partial data retained. Improve the teacher or increase the limit.")
    finally:
        writer.close()
