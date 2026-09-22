"""Vectorized asymmetric SAC loop for actor/privileged-critic observations."""

from __future__ import annotations

import math
import time

import torch

from ..algorithms.asymmetric_sac import AsymmetricReplayBuffer, AsymmetricSAC
from ..algorithms.sac import SACConfig
from ..multi_box.rewards import MultiBoxRewardWeights
from .storage import log_metrics, save_checkpoint


def _critic_state(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """RSL parity: critic receives deployable policy plus privileged features."""
    return torch.cat((observations["policy"], observations["critic"]), dim=-1)


def _reward_breakdown(env) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return the v2 per-step reward terms and verify their composition."""
    breakdown = getattr(env, "_multi_box_grasp_reward_breakdown", None)
    if breakdown is None:
        raise RuntimeError("V2 asymmetric SAC requires a grasp reward breakdown")
    terms = dict(breakdown.terms)
    if not terms:
        raise RuntimeError("V2 grasp reward breakdown has no terms")
    expected_shape = (env.num_envs,)
    if breakdown.total.shape != expected_shape or any(
        value.shape != expected_shape for value in terms.values()
    ):
        raise RuntimeError("V2 grasp reward breakdown has an unexpected shape")
    reconstructed = torch.stack(tuple(terms.values())).sum(0)
    if not torch.allclose(reconstructed, breakdown.total, atol=1e-6, rtol=1e-5):
        error = (reconstructed - breakdown.total).abs().max().item()
        raise RuntimeError(f"V2 grasp reward terms do not sum to total (max error {error})")
    return terms, breakdown.total


def _termination_snapshot(env) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Read the just-computed manager terms and reconstruct terminal masks."""
    manager = env.termination_manager
    terms = {
        name: manager.get_term(name).clone()
        for name in manager.active_terms
    }
    terminated = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    truncated = torch.zeros_like(terminated)
    for name, cfg in zip(manager.active_terms, manager._term_cfgs, strict=True):
        target = truncated if cfg.time_out else terminated
        target |= terms[name]
    return terms, terminated, truncated


def _reset_settling_metrics(env) -> dict[str, int]:
    """Expose cumulative reset rejection causes without coupling SAC to Isaac types."""
    settling = getattr(env, "_multi_box_reset_settling", None)
    if settling is None:
        return {}
    result = {
        "reset_ready_envs": int(settling.ready.sum().item()),
        "reset_settling_envs": int((~settling.ready & ~settling.invalid).sum().item()),
    }
    for metric, attribute in (
        ("reset_invalid_total", "invalid_count"),
        ("reset_region_invalid_total", "region_invalid_count"),
        ("reset_footprint_invalid_total", "footprint_invalid_count"),
        ("reset_shelf_invalid_total", "shelf_invalid_count"),
        ("reset_timeout_invalid_total", "timeout_invalid_count"),
        ("reset_nonfinite_invalid_total", "nonfinite_invalid_count"),
    ):
        value = getattr(settling, attribute, None)
        if value is not None:
            result[metric] = int(value.sum().item())
    return result


def _settle_initial_resets(env, observations):
    """Finish startup roller/contact settling before the first SAC iteration."""
    settling = getattr(env, "_multi_box_reset_settling", None)
    if settling is None:
        raise RuntimeError("V2 asymmetric SAC requires reset-settling state")
    timeout = float(env.cfg.multi_box.reset_settle_timeout_seconds)
    # The first articulated-box teleport can be rejected even without rollers.
    # Two additional timeout windows cover its partial respawn and stable-hold
    # interval without allowing an unbounded startup loop.
    max_steps = max(1, math.ceil(3.0 * timeout / float(env.step_dt)))
    zero_action = torch.zeros_like(env.action_manager.action)
    for step in range(max_steps + 1):
        if bool(settling.ready.all()):
            return observations, step
        if step == max_steps:
            break
        with torch.no_grad():
            observations, _, _, _, _ = env.step(zero_action)
    metrics = _reset_settling_metrics(env)
    # A few spawn regions may need further respawns.  The rollout already
    # excludes unready rows from replay and normalizers, so do not block the
    # ready majority on the slowest reset.  Still fail if startup is broadly
    # unhealthy, which would otherwise waste a long run on invalid samples.
    minimum_ready = math.ceil(0.9 * env.num_envs)
    if metrics["reset_ready_envs"] < minimum_ready:
        raise RuntimeError(
            f"Initial v2 reset settling left too few ready environments "
            f"after {max_steps} steps (minimum {minimum_ready}): {metrics}")
    print(
        f"[V2 SAC] Initial settling reached {metrics['reset_ready_envs']}/"
        f"{env.num_envs} ready envs after {max_steps} steps; remaining rows "
        "will be excluded until their reset is accepted.",
        flush=True,
    )
    return observations, max_steps


def train(env, args, directory, state=None):
    observations, _ = env.reset(seed=args.seed)
    observations, initial_settling_steps = _settle_initial_resets(env, observations)
    actor_obs = observations["policy"]
    critic_obs = _critic_state(observations)
    config = (
        SACConfig(**state["config"])
        if state else SACConfig(
            hidden=args.hidden,
            gamma=MultiBoxRewardWeights().discount,
        )
    )
    agent = AsymmetricSAC(
        actor_obs.shape[-1], critic_obs.shape[-1],
        env.action_manager.total_action_dim, config, env.device)
    start = 0
    if state:
        agent.restore(state)
        start = int(state["iteration"])
        print(
            "[V2 SAC] Restored model/optimizers; replay is empty and warmup restarts.",
            flush=True,
        )
    if args.replay_capacity < env.num_envs:
        raise ValueError("Replay capacity must hold one complete vector step")
    warmup_target = max(
        args.learning_starts, args.warmup_vector_steps * env.num_envs)
    replay = AsymmetricReplayBuffer(
        args.replay_capacity, agent.actor_obs_dim, agent.critic_obs_dim,
        agent.action_dim, args.replay_device)
    print(
        f"[V2 SAC] actor={agent.actor_obs_dim} critic={agent.critic_obs_dim} "
        f"actions={agent.action_dim} replay={replay.bytes / 2**30:.3f} GiB "
        f"on {args.replay_device}; warmup={warmup_target}; "
        f"initial_settling_steps={initial_settling_steps}",
        flush=True,
    )

    transitions = valid_transitions = optimizer_updates = skipped_nonfinite = 0
    skipped_settling = invalid_resets = 0
    update_credit = 0.0
    reward_names = env.reward_manager.active_terms
    for iteration in range(start + 1, start + args.max_iterations + 1):
        tick = time.monotonic()
        metrics = {}
        reward_terms = torch.zeros(len(reward_names), device=env.device)
        reward_sum = torch.zeros((), device=env.device)
        breakdown_sums: dict[str, torch.Tensor] = {}
        breakdown_nonzero: dict[str, int] = {}
        breakdown_max_abs_error = 0.0
        iteration_valid = iteration_warmup = 0
        terminated_episodes = timeout_episodes = 0
        termination_counts: dict[str, int] = {}
        for _ in range(args.rollout_steps):
            with torch.no_grad():
                reset_settling = getattr(env, "_multi_box_reset_settling", None)
                ready_before = (
                    reset_settling.ready.clone() if reset_settling is not None
                    else torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                )
                finite_current = torch.isfinite(actor_obs).all(-1) \
                    & torch.isfinite(critic_obs).all(-1)
                normalizer_mask = finite_current & ready_before
                if normalizer_mask.any():
                    agent.update_normalizers(
                        actor_obs[normalizer_mask], critic_obs[normalizer_mask])
                warming_up = valid_transitions < warmup_target
                safe_actor_obs = torch.where(
                    torch.isfinite(actor_obs), actor_obs, torch.zeros_like(actor_obs))
                action = (
                    torch.rand(
                        (env.num_envs, agent.action_dim), device=env.device) * 2 - 1
                    if warming_up else agent.act(safe_actor_obs)
                )
                next_observations, reward, terminated, truncated, info = env.step(action)
                breakdown_terms, breakdown_total = _reward_breakdown(env)
                termination_terms, expected_terminated, expected_truncated = \
                    _termination_snapshot(env)
                if not torch.equal(terminated, expected_terminated) \
                        or not torch.equal(truncated, expected_truncated):
                    raise RuntimeError(
                        "Environment terminal masks differ from their manager terms")
                success = termination_terms.get("success")
                if success is not None and bool(success.any()):
                    success_reward = breakdown_terms.get("success_event")
                    if success_reward is None or not bool((success_reward[success] > 0).all()):
                        raise RuntimeError(
                            "A success terminal transition is missing its success reward")
                for name, value in termination_terms.items():
                    termination_counts[name] = termination_counts.get(name, 0) \
                        + int(value.sum().item())
                terminal = info.get("transition_next_observations")
                if terminal is None or "policy" not in terminal or "critic" not in terminal:
                    raise RuntimeError(
                        "Asymmetric SAC requires terminal policy and critic observations")
                next_actor = terminal["policy"]
                next_critic = _critic_state(terminal)
                finite_transition = finite_current \
                    & torch.isfinite(next_actor).all(-1) \
                    & torch.isfinite(next_critic).all(-1) \
                    & torch.isfinite(reward) \
                    & torch.isfinite(action).all(-1) \
                    & torch.isfinite(breakdown_total)
                for value in breakdown_terms.values():
                    finite_transition &= torch.isfinite(value)
                skipped_nonfinite += int((~finite_transition).sum().item())
                # Reset settling is outside the task MDP.  Its zero-action,
                # zero-reward frames and invalid partial respawns must never
                # enter replay or observation normalizers.
                finite_transition &= ready_before
                skipped_settling += int((~ready_before).sum().item())
                invalid = termination_terms.get("invalid_reset")
                if invalid is not None:
                    invalid_resets += int(invalid.sum().item())
                count = int(finite_transition.sum().item())
                if count:
                    error = (
                        reward[finite_transition] - breakdown_total[finite_transition]
                    ).abs().max().item()
                    breakdown_max_abs_error = max(breakdown_max_abs_error, error)
                    if error > 1e-5:
                        raise RuntimeError(
                            "Environment reward differs from v2 grasp reward "
                            f"breakdown (max error {error})")
                    replay.add(
                        actor_obs=actor_obs[finite_transition],
                        critic_obs=critic_obs[finite_transition],
                        action=action[finite_transition],
                        reward=reward[finite_transition],
                        next_actor_obs=next_actor[finite_transition],
                        next_critic_obs=next_critic[finite_transition],
                        terminated=terminated[finite_transition],
                    )
                    reward_sum += reward[finite_transition].sum()
                    reward_terms += (
                        env.reward_manager._step_reward[finite_transition].sum(0)
                        * env.step_dt
                    )
                    for name, value in breakdown_terms.items():
                        selected = value[finite_transition]
                        if name not in breakdown_sums:
                            breakdown_sums[name] = selected.sum()
                            breakdown_nonzero[name] = int((selected != 0).sum().item())
                        else:
                            breakdown_sums[name] += selected.sum()
                            breakdown_nonzero[name] += int((selected != 0).sum().item())
                valid_transitions += count
                iteration_valid += count
                iteration_warmup += count if warming_up else 0
                transitions += env.num_envs
                terminated_episodes += int(terminated.sum().item())
                timeout_episodes += int(truncated.sum().item())
                actor_obs = next_observations["policy"]
                critic_obs = _critic_state(next_observations)

            if valid_transitions >= warmup_target and replay.size >= args.batch_size and count:
                update_credit += args.updates_per_step * count / env.num_envs
                updates = int(update_credit)
                update_credit -= updates
                for _ in range(updates):
                    metrics = agent.update(replay.sample(args.batch_size, env.device))
                    optimizer_updates += 1

        metrics.update(
            reward_per_valid_step=(
                reward_sum.item() / iteration_valid if iteration_valid else None),
            transitions=transitions,
            valid_transitions=valid_transitions,
            valid_transitions_this_iteration=iteration_valid,
            warmup_target=warmup_target,
            warmup_complete=valid_transitions >= warmup_target,
            warmup_transitions_this_iteration=iteration_warmup,
            replay_size=replay.size,
            replay_gib=replay.bytes / 2**30,
            nonfinite_transitions=skipped_nonfinite,
            settling_transitions_skipped=skipped_settling,
            invalid_resets=invalid_resets,
            optimizer_updates=optimizer_updates,
            terminated_episodes=terminated_episodes,
            timeout_episodes=timeout_episodes,
            initial_settling_steps=initial_settling_steps,
            reward_breakdown_max_abs_error=breakdown_max_abs_error,
            transitions_per_second=(
                args.rollout_steps * env.num_envs / (time.monotonic() - tick)),
        )
        metrics.update({
            f"reward/{name}": value / iteration_valid if iteration_valid else None
            for name, value in zip(reward_names, reward_terms.tolist())
        })
        metrics.update({
            f"reward_term/{name}": value.item() / iteration_valid
            if iteration_valid else None
            for name, value in breakdown_sums.items()
        })
        metrics.update({
            f"reward_term_nonzero/{name}": count / iteration_valid
            if iteration_valid else None
            for name, count in breakdown_nonzero.items()
        })
        metrics.update({
            f"termination/{name}": count
            for name, count in termination_counts.items()
        })
        metrics.update(_reset_settling_metrics(env))
        if str(env.device).startswith("cuda"):
            metrics.update(
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        log_metrics(directory, iteration, metrics)
        if iteration % args.save_interval == 0 or iteration == start + args.max_iterations:
            keep = None if getattr(args, "external_checkpoint_retention", False) \
                else args.keep_checkpoints
            save_checkpoint(directory, agent.checkpoint(), iteration, keep)
    return agent
