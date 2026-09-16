"""Vectorized SAC loop; replay is RAM/GPU storage, never a disk artifact."""

import time
import torch
from ..algorithms.sac import SAC, SACConfig, ReplayBuffer
from .storage import save_checkpoint, log_metrics, EpisodeMetrics


def train(env, args, directory, state=None):
    obs = env.reset()[0]["policy"]
    env._capture_task_metrics = True
    config = SACConfig(**state["config"]) if state else SACConfig()
    agent = SAC(obs.shape[-1], env.action_manager.total_action_dim, config, env.device)
    start = 0
    if state:
        agent.restore(state)
        start = state["iteration"]
        print("[SAC] Restored model/optimizers; replay is empty and exploration warmup restarts.", flush=True)
    if args.replay_capacity < env.num_envs:
        raise ValueError("Replay capacity must hold at least one complete vector step")
    warmup_target = max(args.learning_starts,
                        getattr(args, "warmup_vector_steps", 450) * env.num_envs)
    replay = ReplayBuffer(args.replay_capacity, agent.obs_dim, agent.action_dim, args.replay_device)
    print(f"[SAC] Replay={replay.bytes / 2**30:.3f} GiB on {args.replay_device}; "
          f"sample reuse={args.updates_per_step * args.batch_size / env.num_envs:.2f} per transition", flush=True)
    print(f"[SAC] Warmup requires {warmup_target} action-enabled transitions; "
          "settling observations/transitions are excluded from normalization/replay.", flush=True)
    transitions = valid_transitions = optimizer_updates = 0
    skipped_nonfinite = 0
    settling_transitions = 0
    update_credit = 0.0
    command = env.command_manager.get_term("workcell") if getattr(env, "command_manager", None) else None
    reward_names = env.reward_manager.active_terms if getattr(env, "reward_manager", None) else []
    for iteration in range(start + 1, start + args.max_iterations + 1):
        tick = time.monotonic()
        metrics, reward_sum, dones = {}, 0.0, 0
        episodes = EpisodeMetrics()
        task_sums = {}
        reward_terms = torch.zeros(len(reward_names), device=env.device)
        valid_reward_sum = torch.zeros((), device=env.device)
        iteration_valid = iteration_warmup = settling_dones = 0
        for _ in range(args.rollout_steps):
            with torch.no_grad():
                # Snapshot BEFORE step: readiness can change at the end of this
                # step and is cleared on auto-reset. An enabled terminal action
                # must be retained; the last settling step must not be retained.
                enabled = (command.settling.ready.clone() if command is not None and command.settling is not None
                           else torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
                settling_transitions += int((~enabled).sum().item())
                finite_obs = torch.isfinite(obs).all(-1)
                enabled &= finite_obs
                agent.normalizer.update(obs[enabled])
                warming_up = valid_transitions < warmup_target
                action = (torch.rand((env.num_envs, agent.action_dim), device=env.device) * 2 - 1
                          if warming_up else agent.act(torch.where(torch.isfinite(obs), obs, 0.)))
                next_dict, reward, terminated, truncated, info = env.step(action)
                episodes.record(env)
                finite_transition = (torch.isfinite(info["transition_next_obs"]).all(-1)
                                     & torch.isfinite(reward) & torch.isfinite(action).all(-1))
                skipped_nonfinite += int((~finite_transition | ~finite_obs).sum().item())
                enabled &= finite_transition
                count = int(enabled.sum().item())
                if count:
                    replay.add(obs=obs[enabled], action=action[enabled], reward=reward[enabled],
                               next_obs=info["transition_next_obs"][enabled], terminated=terminated[enabled])
                    valid_reward_sum += reward[enabled].sum()
                    # Isaac Lab 2.3.2 stores weighted rates here; multiply by dt
                    # to log actual reward contributions per enabled transition.
                    if reward_names:
                        reward_terms += env.reward_manager._step_reward[enabled].sum(0) * env.step_dt
                    for name, value in info.get("transition_task_metrics", {}).items():
                        total = value[enabled].float().sum()
                        if name not in task_sums:
                            task_sums[name] = total
                        else:
                            task_sums[name] += total
                valid_transitions += count
                iteration_valid += count
                iteration_warmup += count if warming_up else 0
                settling_dones += int(((terminated | truncated) & ~enabled).sum().item())
                obs = next_dict["policy"]
                reward_sum += reward.mean().item()
                dones += (terminated | truncated).sum().item()
                transitions += env.num_envs
            if valid_transitions >= warmup_target and replay.size >= args.batch_size and count:
                # Partial-ready vector steps earn proportionally fewer updates.
                # Do not accrue update debt during settling or warmup.
                update_credit += args.updates_per_step * count / env.num_envs
                updates = int(update_credit)
                update_credit -= updates
                for _ in range(updates):
                    metrics = agent.update(replay.sample(args.batch_size, env.device))
                    optimizer_updates += 1
        metrics.update(**episodes.report(), reward_per_step=reward_sum / args.rollout_steps, completed_episodes=dones,
                       transitions=transitions, replay_size=replay.size,
                       valid_transitions=valid_transitions,
                       skipped_settling_transitions=settling_transitions,
                       nonfinite_transitions=skipped_nonfinite,
                       valid_vector_steps=valid_transitions / env.num_envs,
                       warmup_target=warmup_target, warmup_complete=valid_transitions >= warmup_target,
                       warmup_transitions_this_iteration=iteration_warmup,
                       valid_transitions_this_iteration=iteration_valid,
                       settling_completed_episodes=settling_dones,
                       optimizer_updates=optimizer_updates,
                       replay_history_seconds=replay.size / env.num_envs * env.step_dt
                       if hasattr(env, "step_dt") else None,
                       transitions_per_second=args.rollout_steps * env.num_envs / (time.monotonic() - tick))
        # Aggregate EVERY enabled step, including pre-reset terminal metrics.
        # None distinguishes "no controllable data" from zero contact/reward.
        metrics["reward_per_valid_step"] = valid_reward_sum.item() / iteration_valid if iteration_valid else None
        metrics.update({f"reward/{name}": value / iteration_valid if iteration_valid else None
                        for name, value in zip(reward_names, reward_terms.tolist())})
        metrics.update({f"workcell/{name}": task_sums[name].item() / iteration_valid
                        if iteration_valid else None for name in (command.metrics if command is not None else {})})
        if str(env.device).startswith("cuda"):
            metrics.update(torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                           torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        log_metrics(directory, iteration, metrics)
        if iteration % args.save_interval == 0 or iteration == start + args.max_iterations:
            keep = None if getattr(args, "external_checkpoint_retention", False) else args.keep_checkpoints
            save_checkpoint(directory, agent.checkpoint(), iteration, keep)
