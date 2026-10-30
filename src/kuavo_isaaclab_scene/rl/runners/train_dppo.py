"""On-policy diffusion fine-tuning on the exact shared manager-based task."""

import time
import torch
from ..algorithms.diffusion import load_diffusion
from ..algorithms.dppo import DPPO, DPPOConfig
from ..algorithms.common import generalized_advantage
from .storage import save_checkpoint, log_metrics, EpisodeMetrics


def train(env, args, directory, state):
    policy = load_diffusion(state, env.device)
    config = DPPOConfig(**state["dppo_config"]) if state["algorithm"] == "dppo" else DPPOConfig(
        minibatch_size=args.batch_size, epochs=args.epochs, critic_warmup=args.critic_warmup)
    agent = DPPO(policy, config)
    start = 0
    if state["algorithm"] == "dppo":
        agent.restore(state)
        start = state["iteration"]
    obs = env.reset()[0]["policy"]
    if (obs.shape[-1], env.action_manager.total_action_dim) != (policy.obs_dim, policy.action_dim):
        raise ValueError("Diffusion checkpoint dimensions do not match the environment")
    steps, count = args.rollout_steps, env.num_envs
    denoise, horizon = policy.config.denoising_steps, policy.config.horizon
    # Allocate once: no list+stack doubling of multi-GiB rollout storage.
    observations = torch.empty((steps, count, policy.obs_dim), device=env.device)
    chains = torch.empty((steps, count, denoise + 1, horizon, policy.action_dim), device=env.device)
    logps = torch.empty((steps, count, denoise), device=env.device)
    rewards = torch.empty((steps, count), device=env.device)
    values, next_values = torch.empty_like(rewards), torch.empty_like(rewards)
    terminated = torch.empty((steps, count), device=env.device, dtype=torch.bool)
    done = torch.empty_like(terminated)
    storage = (observations, chains, logps, rewards, values, next_values, terminated, done)
    print(f"[DPPO] Rollout={sum(x.numel() * x.element_size() for x in storage) / 2**30:.3f} GiB; "
          f"predict {horizon} actions, execute 1, replan every control step.", flush=True)
    for iteration in range(start + 1, start + args.max_iterations + 1):
        tick = time.monotonic()
        episodes = EpisodeMetrics()
        with torch.no_grad():
            for step in range(steps):
                observations[step] = obs
                actions, chain, logp = policy.sample(obs)
                chains[step], logps[step], values[step] = chain, logp, agent.value(obs)
                next_dict, reward, term, trunc, info = env.step(actions[:, 0])
                episodes.record(env)
                rewards[step], terminated[step], done[step] = reward, term, term | trunc
                next_values[step] = agent.value(info["transition_next_obs"])
                obs = next_dict["policy"]
            advantage, returns = generalized_advantage(
                rewards, values, next_values, terminated, done, config.gamma, config.gae_lambda)
        metrics = agent.update(observations.flatten(0, 1), chains.flatten(0, 1), logps.flatten(0, 1),
                               advantage.flatten(), returns.flatten(), iteration > config.critic_warmup)
        metrics.update(**episodes.report(), reward_per_step=rewards.mean().item(), completed_episodes=done.sum().item(),
                       critic_warmup=iteration <= config.critic_warmup,
                       transitions_per_second=steps * count / (time.monotonic() - tick))
        log_metrics(directory, iteration, metrics)
        if iteration % args.save_interval == 0 or iteration == start + args.max_iterations:
            save_checkpoint(directory, agent.checkpoint(), iteration, args.keep_checkpoints)
