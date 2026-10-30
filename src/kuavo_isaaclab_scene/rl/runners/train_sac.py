"""Vectorized SAC loop; replay is RAM/GPU storage, never a disk artifact."""

import time
import torch
from ..algorithms.sac import SAC, SACConfig, ReplayBuffer
from .storage import save_checkpoint, log_metrics, EpisodeMetrics


def train(env, args, directory, state=None):
    obs = env.reset()[0]["policy"]
    config = SACConfig(**state["config"]) if state else SACConfig()
    agent = SAC(obs.shape[-1], env.action_manager.total_action_dim, config, env.device)
    start = 0
    if state:
        agent.restore(state)
        start = state["iteration"]
        print("[SAC] Restored model/optimizers; replay is empty and exploration warmup restarts.", flush=True)
    if args.replay_capacity < env.num_envs:
        raise ValueError("Replay capacity must hold at least one complete vector step")
    replay = ReplayBuffer(args.replay_capacity, agent.obs_dim, agent.action_dim, args.replay_device)
    print(f"[SAC] Replay={replay.bytes / 2**30:.3f} GiB on {args.replay_device}; "
          f"sample reuse={args.updates_per_step * args.batch_size / env.num_envs:.2f} per transition", flush=True)
    transitions = 0
    for iteration in range(start + 1, start + args.max_iterations + 1):
        tick = time.monotonic()
        metrics, reward_sum, dones = {}, 0.0, 0
        episodes = EpisodeMetrics()
        for _ in range(args.rollout_steps):
            with torch.no_grad():
                agent.normalizer.update(obs)
                action = (torch.rand((env.num_envs, agent.action_dim), device=env.device) * 2 - 1
                          if transitions < args.learning_starts else agent.act(obs))
                next_dict, reward, terminated, truncated, info = env.step(action)
                episodes.record(env)
                replay.add(obs=obs, action=action, reward=reward,
                           next_obs=info["transition_next_obs"], terminated=terminated)
                obs = next_dict["policy"]
                reward_sum += reward.mean().item()
                dones += (terminated | truncated).sum().item()
                transitions += env.num_envs
            if transitions >= args.learning_starts:
                for _ in range(args.updates_per_step):
                    metrics = agent.update(replay.sample(args.batch_size, env.device))
        metrics.update(**episodes.report(), reward_per_step=reward_sum / args.rollout_steps, completed_episodes=dones,
                       transitions=transitions, replay_size=replay.size,
                       transitions_per_second=args.rollout_steps * env.num_envs / (time.monotonic() - tick))
        log_metrics(directory, iteration, metrics)
        if iteration % args.save_interval == 0 or iteration == start + args.max_iterations:
            save_checkpoint(directory, agent.checkpoint(), iteration, args.keep_checkpoints)
