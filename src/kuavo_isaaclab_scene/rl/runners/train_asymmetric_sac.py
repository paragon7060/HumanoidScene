"""Vectorized asymmetric SAC loop for actor/privileged-critic observations."""

from __future__ import annotations

import math
import time

import torch

from ..algorithms.asymmetric_sac import ActorImitationBuffer, AsymmetricReplayBuffer, AsymmetricSAC
from ..algorithms.sac import SACConfig
from ..multi_box.rewards import MultiBoxRewardWeights
from ..multi_box.experiments.guided_exploration import (
    GraspActionProjector, GuidedDemoWarmup,
)
from .storage import log_metrics, save_checkpoint


_SAFETY_CAUSES = (
    "robot_rack_collision", "obstacle_collision", "workspace_limit",
    "box_drop", "box_lift_limit", "box_speed_limit", "self_collision",
)
_CONTACT_FORCE_LIMITS_N = (0.1, 5.0, 10.0, 20.0)
_MAX_REASONABLE_GRASP_DISTANCE_M = 5.0
_FRONT_DISTANCE_BINS_M = (0.25, 0.5, 1.0, 2.0)
_FRONT_DISTANCE_LABELS = ("le_0p25m", "0p25_to_0p5m", "0p5_to_1m", "1_to_2m", "over_2m")


class _SafetyDiagnostics:
    """Count exact failure predicates and contact-force bands per iteration."""

    def __init__(self, device, obstacle_names=(), obstacle_threshold=5.0):
        self.invalid_box_pose = torch.zeros((), dtype=torch.long, device=device)
        self.invalid_flap_pose = torch.zeros((), dtype=torch.long, device=device)
        self.causes = torch.zeros(len(_SAFETY_CAUSES), dtype=torch.long, device=device)
        self.overlap = torch.zeros((), dtype=torch.long, device=device)
        self.unattributed = torch.zeros((), dtype=torch.long, device=device)
        self.eligible = torch.zeros((), dtype=torch.long, device=device)
        self.force_limits = torch.tensor(_CONTACT_FORCE_LIMITS_N, device=device)
        self.force_bands = torch.zeros(2, len(_CONTACT_FORCE_LIMITS_N), dtype=torch.long, device=device)
        self.force_max = torch.zeros(2, device=device)
        self.obstacle_names = obstacle_names
        self.obstacle_threshold = obstacle_threshold
        self.obstacle_counts = torch.zeros(len(obstacle_names), dtype=torch.long, device=device)
        self.obstacle_max = torch.zeros(len(obstacle_names), device=device)

    def record(self, safety, unsafe):
        get = safety.__getitem__ if isinstance(safety, dict) else lambda name: getattr(safety, name)
        self.invalid_box_pose += get("invalid_box_pose").sum()
        self.invalid_flap_pose += get("invalid_flap_pose").sum()
        causes = torch.stack([get(name) & unsafe for name in _SAFETY_CAUSES])
        self.causes += causes.sum(-1)
        count = causes.sum(0)
        self.overlap += ((count > 1) & unsafe).sum()
        self.unattributed += ((count == 0) & unsafe).sum()
        eligible = get("contact_eligible")
        self.eligible += eligible.sum()
        if self.obstacle_names:
            force = get("obstacle_target_force_n")
            self.obstacle_counts += ((force > self.obstacle_threshold) & eligible[:, None]).sum(0)
            self.obstacle_max = torch.maximum(self.obstacle_max,
                torch.where(eligible[:, None], force, 0.0).amax(0))
        for index, force in enumerate((get("rack_force_n"), get("obstacle_force_n"))):
            self.force_bands[index] += ((force[:, None] > self.force_limits) & eligible[:, None]).sum(0)
            finite_force = torch.nan_to_num(force, nan=0.0, posinf=1e6, neginf=0.0).clamp(0, 1e6)
            self.force_max[index] = torch.maximum(
                self.force_max[index], torch.where(eligible, finite_force, 0.0).max())

    def report(self) -> dict[str, int | float]:
        metrics = {
            f"unsafe_cause/{name}": int(value)
            for name, value in zip(_SAFETY_CAUSES, self.causes.tolist(), strict=True)
        }
        metrics["reset_invalid_box_pose"] = int(self.invalid_box_pose.item())
        metrics["reset_invalid_flap_pose"] = int(self.invalid_flap_pose.item())
        metrics["unsafe_cause/overlap"] = int(self.overlap.item())
        metrics["unsafe_cause/unattributed"] = int(self.unattributed.item())
        metrics["contact_force/eligible_samples"] = int(self.eligible.item())
        for name, count, force in zip(self.obstacle_names, self.obstacle_counts.tolist(),
                                      self.obstacle_max.tolist(), strict=True):
            metrics[f"unsafe_obstacle/{name}"] = int(count)
            metrics[f"contact_force/obstacle_{name}_max_n"] = float(force)
        for index, family in enumerate(("rack", "obstacle")):
            metrics[f"contact_force/{family}_max_n"] = float(self.force_max[index].item())
            for threshold, count in zip(_CONTACT_FORCE_LIMITS_N, self.force_bands[index].tolist(), strict=True):
                label = str(threshold).replace(".", "p")
                metrics[f"contact_force/{family}_gt_{label}_n"] = int(count)
        return metrics


class _ApproachDiagnostics:
    """Relate eligible rack/obstacle failures to distance from a safe front lane."""

    def __init__(self, device):
        bins = len(_FRONT_DISTANCE_LABELS)
        self.boundaries = torch.tensor(_FRONT_DISTANCE_BINS_M, device=device)
        self.samples = torch.zeros(bins, dtype=torch.long, device=device)
        self.rack = torch.zeros_like(self.samples)
        self.obstacle = torch.zeros_like(self.samples)
        self.both_staged = torch.zeros((), dtype=torch.long, device=device)

    def record(self, front_distance, safety, valid):
        if front_distance.ndim != 2 or front_distance.shape[-1] != 2 or valid.ndim != 1:
            raise ValueError("Front distance must contain both hands for each valid transition")
        get = safety.__getitem__ if isinstance(safety, dict) else lambda name: getattr(safety, name)
        eligible = get("contact_eligible")[valid]
        nearest = front_distance.amin(-1)
        bucket = torch.bucketize(nearest.contiguous(), self.boundaries)
        selected = torch.nn.functional.one_hot(bucket, len(self.samples)).bool() & eligible[:, None]
        self.samples += selected.sum(0)
        self.rack += (selected & get("robot_rack_collision")[valid][:, None]).sum(0)
        self.obstacle += (selected & get("obstacle_collision")[valid][:, None]).sum(0)
        self.both_staged += (eligible & (front_distance <= 0.25).all(-1)).sum()

    def report(self):
        result = {"front_stage/both_under_0p25m": int(self.both_staged.item())}
        for name, samples, rack, obstacle in zip(
                _FRONT_DISTANCE_LABELS, self.samples.tolist(), self.rack.tolist(),
                self.obstacle.tolist(), strict=True):
            prefix = f"front_stage/{name}"
            result[f"{prefix}/samples"] = samples
            result[f"{prefix}/rack_unsafe"] = rack
            result[f"{prefix}/obstacle_unsafe"] = obstacle
            result[f"{prefix}/rack_rate"] = rack / samples if samples else None
        return result


def _grasp_distance_masks(geometry, num_envs):
    """Reject finite simulator explosions before they enter replay or averages."""
    finite = plausible = torch.ones(
        num_envs, dtype=torch.bool, device=geometry["matched_flap_distance_m"].device)
    for name in ("matched_flap_distance_m", "front_staging_distance_m"):
        values = geometry.get(name)
        if values is None or values.shape != (num_envs, 2):
            raise RuntimeError(f"V2 SAC requires [env,2] {name}")
        finite = finite & torch.isfinite(values).all(-1)
        plausible = plausible & (
            (values >= 0) & (values <= _MAX_REASONABLE_GRASP_DISTANCE_M)
        ).all(-1)
    return finite, plausible


def _critic_state(observations: dict[str, torch.Tensor]) -> torch.Tensor:
    """RSL parity: critic receives deployable policy plus privileged features."""
    return torch.cat((observations["policy"], observations["critic"]), dim=-1)


def _sample_warmup_action(env, continuous_scale: float) -> torch.Tensor:
    """Explore small joint increments while sampling binary grippers fully."""
    if not 0 < continuous_scale <= 1:
        raise ValueError("Warmup continuous scale must be in (0, 1]")
    action = torch.empty(
        env.num_envs, env.action_manager.total_action_dim, device=env.device)
    offset = 0
    for name in env.action_manager.active_terms:
        width = env.action_manager.get_term(name).action_dim
        selected = action[:, offset:offset + width]
        if "gripper" in name:
            selected.copy_(torch.where(
                torch.rand_like(selected) < 0.5, -1.0, 1.0))
        else:
            selected.uniform_(-continuous_scale, continuous_scale)
        offset += width
    if offset != action.shape[1]:
        raise RuntimeError("Warmup action terms do not cover the action space")
    return action


def _demo_fraction(initial: float, updates: int, decay_updates: int) -> float:
    """Linearly retire imitation after the first part of actual SAC learning."""
    if not 0 <= initial < 1 or updates < 0 or decay_updates < 1:
        raise ValueError("Invalid demonstration decay schedule")
    return initial * max(0.0, 1.0 - updates / decay_updates)


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


def train(env, args, directory, state=None, demonstration_batch=None):
    observations, _ = env.reset(seed=args.seed)
    observations, initial_settling_steps = _settle_initial_resets(env, observations)
    actor_obs = observations["policy"]
    critic_obs = _critic_state(observations)
    config = (
        SACConfig(**state["config"])
        if state else SACConfig(
            hidden=args.hidden,
            gamma=MultiBoxRewardWeights().discount,
            min_alpha=getattr(args, "min_alpha", 0.0),
            initial_alpha=getattr(args, "initial_alpha", 0.001),
            reward_scale=getattr(args, "reward_scale", 10.0),
            entropy_backup=getattr(args, "entropy_backup", False),
            actor_feature_mode=getattr(args, "actor_feature_mode", "grasp_target"),
            freeze_actor_normalizer=(demonstration_batch is not None
                                     and getattr(args, "freeze_actor_normalizer", True)),
            initial_policy_std=getattr(args, "initial_policy_std", 0.15),
            max_policy_std=getattr(args, "max_policy_std", 0.3),
            actor_lr=getattr(args, "actor_lr", 0.00003),
        )
    )
    projection = GraspActionProjector([
        (name, env.action_manager.get_term(name).action_dim)
        for name in env.action_manager.active_terms
    ])
    agent = AsymmetricSAC(
        actor_obs.shape[-1], critic_obs.shape[-1],
        env.action_manager.total_action_dim, config, env.device,
        action_projector=projection)
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
    demonstration = None
    if demonstration_batch is not None:
        demonstration = AsymmetricReplayBuffer(
            len(demonstration_batch["reward"]), agent.actor_obs_dim,
            agent.critic_obs_dim, agent.action_dim, "cpu")
        demonstration.add(**demonstration_batch)
    pretraining = {"steps": 0, "initial_mse": 0.0, "final_mse": 0.0}
    if demonstration is not None and state is None:
        ready = env._multi_box_reset_settling.ready
        agent.actor_normalizer.update(agent.actor_features(actor_obs[ready]))
        pretraining = agent.pretrain_actor(
            demonstration_batch["actor_obs"].to(env.device),
            demonstration_batch["action"].to(env.device),
            steps=getattr(args, "demo_pretrain_steps", 0),
            batch_size=getattr(args, "demo_pretrain_batch_size", 256),
        )
    goal_capacity = getattr(args, "goal_replay_capacity", 100_000)
    goal_replay = AsymmetricReplayBuffer(
        goal_capacity, agent.actor_obs_dim, agent.critic_obs_dim, agent.action_dim, "cpu")
    guided_warmup = (
        GuidedDemoWarmup(
            env, noise_scale=getattr(args, "demo_warmup_noise_scale", 0.12))
        if demonstration is not None
        and getattr(args, "demo_guided_warmup", True)
        and (state is not None or pretraining["steps"] > 0
             or getattr(args, "guided_warmup_mode", "bc") == "ik")
        else None
    )
    if guided_warmup is not None and getattr(args, "guided_warmup_mode", "bc") == "ik":
        from ..multi_box.experiments.kinematic_exploration import KinematicGraspExplorer
        guided_warmup = KinematicGraspExplorer(env, demonstration_batch)
    teacher_replay = ActorImitationBuffer(
        min(args.replay_capacity, max(goal_capacity, warmup_target + env.num_envs))
        if getattr(args, "guided_warmup_mode", "bc") == "ik" else goal_capacity,
        agent.actor_obs_dim, agent.action_dim)
    if state and state.get("teacher_imitation"):
        teacher_replay.add(**state["teacher_imitation"])
    success_replay = AsymmetricReplayBuffer(
        getattr(args, "success_replay_capacity", 10_000), agent.actor_obs_dim,
        agent.critic_obs_dim, agent.action_dim, "cpu")
    if state and state.get("success_replay"):
        success_replay.add(**state["success_replay"])
    teacher_pretraining = (state or {}).get("teacher_pretraining", {
        "steps": 0, "initial_mse": 0.0, "final_mse": 0.0})
    teacher_fitted = bool((state or {}).get("teacher_fitted", False))
    demo_decay_updates = (state or {}).get("demo_decay_updates", max(1, math.ceil(
        args.max_iterations * args.rollout_steps * args.updates_per_step
        * getattr(args, "demo_decay_fraction", 0.3))))
    print(
        f"[V2 SAC] actor={agent.actor_obs_dim} critic={agent.critic_obs_dim} "
        f"actions={agent.action_dim} replay={replay.bytes / 2**30:.3f} GiB "
        f"on {args.replay_device}; warmup={warmup_target}; "
        f"warmup_action_hold={getattr(args, 'warmup_action_hold_steps', 1)}; "
        f"warmup_continuous_scale={getattr(args, 'warmup_continuous_scale', 1.0)}; "
        f"demo_transitions={demonstration.size if demonstration else 0}; "
        f"demo_initial_fraction={getattr(args, 'demo_batch_fraction', 0.0)}; "
        f"demo_decay_updates={demo_decay_updates}; demo_usage=actor_bc_only; "
        f"demo_pretraining={pretraining}; guided_warmup={guided_warmup is not None}; "
        f"min_alpha={config.min_alpha}; "
        f"initial_settling_steps={initial_settling_steps}",
        flush=True,
    )

    transitions = valid_transitions = skipped_nonfinite = 0
    optimizer_updates = int((state or {}).get("optimizer_updates", 0))
    actor_updates = int((state or {}).get("actor_updates", 0))
    skipped_settling = invalid_resets = 0
    update_credit = 0.0
    warmup_action = None
    warmup_vector_step = 0
    warmup_action_hold = getattr(args, "warmup_action_hold_steps", 1)
    if warmup_action_hold < 1:
        raise ValueError("warmup_action_hold_steps must be positive")
    gripper_columns = {}
    action_offset = 0
    for name in env.action_manager.active_terms:
        if name in ("left_gripper", "right_gripper"):
            gripper_columns[name] = action_offset
        action_offset += env.action_manager.get_term(name).action_dim
    gripper_indices = [gripper_columns[name] for name in (
        "left_gripper", "right_gripper")]
    torso_term = env.action_manager.get_term("height")
    torso_diagnostics = all(hasattr(torso_term, name) for name in (
        "_joint_ids", "_joint_targets", "_pitch_reference"))
    reward_names = env.reward_manager.active_terms
    for iteration in range(start + 1, start + args.max_iterations + 1):
        tick = time.monotonic()
        metrics = {}
        reward_terms = torch.zeros(len(reward_names), device=env.device)
        reward_sum = torch.zeros((), device=env.device)
        breakdown_sums: dict[str, torch.Tensor] = {}
        breakdown_nonzero: dict[str, int] = {}
        progress_abs = {name: torch.zeros((), device=env.device) for name in (
            "approach_progress", "front_staging_progress")}
        progress_positive = {name: torch.zeros((), device=env.device) for name in progress_abs}
        flap_distance_sum = torch.zeros(2, device=env.device)
        front_distance_sum = torch.zeros(2, device=env.device)
        flap_near_count = torch.zeros(2, device=env.device)
        front_near_count = torch.zeros(2, device=env.device)
        both_flap_near_count = torch.zeros((), device=env.device)
        pinch_count = torch.zeros(2, device=env.device)
        bilateral_pinch_count = torch.zeros((), device=env.device)
        instantaneous_success_count = torch.zeros((), device=env.device)
        success_gate_counts = {name: torch.zeros((), device=env.device)
                               for name in ("opposing_flaps", "stable", "proof_lift")}
        max_hold_time = torch.zeros((), device=env.device)
        close_count = torch.zeros(2, device=env.device)
        premature_close_count = torch.zeros(2, device=env.device)
        torso_pitch_error_sum = torch.zeros((), device=env.device)
        torso_tracking_error_sum = torch.zeros((), device=env.device)
        torso_pitch_error_max = torch.zeros((), device=env.device)
        torso_diagnostic_count = 0
        breakdown_max_abs_error = 0.0
        iteration_valid = iteration_warmup = 0
        iteration_guided_warmup = 0
        warmup_successes = sac_successes = 0
        success_samples = 0
        demo_samples = 0
        online_teacher_labels = 0
        terminated_episodes = timeout_episodes = 0
        termination_counts: dict[str, int] = {}
        from ..multi_box.debug.contact_force import eligible_obstacle_targets
        obstacle_names = [path.rsplit("/", 1)[-1]
                          for path in eligible_obstacle_targets(env.cfg.scene)]
        safety_diagnostics = _SafetyDiagnostics(
            env.device, obstacle_names, float(env.cfg.task.obstacle_contact_force))
        approach_diagnostics = _ApproachDiagnostics(env.device)
        skipped_implausible = torch.zeros((), dtype=torch.long, device=env.device)
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
                teacher_action = None
                safe_actor_obs = torch.where(
                    torch.isfinite(actor_obs), actor_obs, torch.zeros_like(actor_obs))
                if warming_up:
                    if guided_warmup is not None:
                        action = (guided_warmup.act(safe_actor_obs)
                                  if getattr(args, "guided_warmup_mode", "bc") == "ik"
                                  else guided_warmup.act(agent, safe_actor_obs))
                    else:
                        if warmup_action is None or warmup_vector_step % warmup_action_hold == 0:
                            warmup_action = _sample_warmup_action(
                                env, getattr(args, "warmup_continuous_scale", 1.0))
                        action = warmup_action.clone()
                    warmup_vector_step += 1
                else:
                    action = agent.act(safe_actor_obs)
                    if (getattr(args, "online_teacher_labels", False)
                            and guided_warmup is not None
                            and getattr(args, "guided_warmup_mode", "bc") == "ik"
                            and _demo_fraction(getattr(args, "demo_batch_fraction", 0.0),
                                               actor_updates, demo_decay_updates) > 0):
                        teacher_action = projection(
                            safe_actor_obs, guided_warmup.act(safe_actor_obs))
                action = projection(safe_actor_obs, action)
                next_observations, reward, terminated, truncated, info = env.step(action)
                if guided_warmup is not None:
                    guided_warmup.reset(terminated | truncated)
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
                successful = int(termination_terms["success"].sum().item())
                warmup_successes += successful if warming_up else 0
                sac_successes += successful if not warming_up else 0
                safety = info.get("transition_safety")
                if safety is None:
                    raise RuntimeError("V2 SAC requires pre-reset grasp safety measurements")
                safety_diagnostics.record(safety, termination_terms["unsafe"])
                geometry = info.get("transition_grasp_geometry")
                if geometry is None:
                    raise RuntimeError("V2 SAC requires pre-reset hand/flap geometry")
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
                geometry_finite, geometry_plausible = _grasp_distance_masks(
                    geometry, env.num_envs)
                finite_transition &= geometry_finite
                hand_pinching = geometry.get("hand_pinching")
                instantaneous_success = geometry.get("instantaneous_success")
                if hand_pinching is None or hand_pinching.shape != (env.num_envs, 2) \
                        or hand_pinching.dtype != torch.bool \
                        or instantaneous_success is None \
                        or instantaneous_success.shape != (env.num_envs,) \
                        or instantaneous_success.dtype != torch.bool:
                    raise RuntimeError("V2 SAC requires pre-reset pinch milestone masks")
                skipped_nonfinite += int((~finite_transition).sum().item())
                skipped_implausible += (finite_transition & ~geometry_plausible).sum()
                finite_transition &= geometry_plausible
                # Reset settling is outside the task MDP.  Its zero-action,
                # zero-reward frames and invalid partial respawns must never
                # enter replay or observation normalizers.
                finite_transition &= ready_before
                skipped_settling += int((~ready_before).sum().item())
                invalid = termination_terms.get("invalid_reset")
                if invalid is not None:
                    invalid_resets += int(invalid.sum().item())
                    finite_transition &= ~invalid
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
                    transition_batch = dict(
                        actor_obs=actor_obs[finite_transition],
                        critic_obs=critic_obs[finite_transition],
                        action=action[finite_transition],
                        reward=reward[finite_transition],
                        next_actor_obs=next_actor[finite_transition],
                        next_critic_obs=next_critic[finite_transition],
                        terminated=terminated[finite_transition],
                    )
                    replay.add(**transition_batch)
                    success_rows = termination_terms["success"][finite_transition]
                    if success_rows.any():
                        success_replay.add(**{
                            key: value[success_rows] for key, value in transition_batch.items()})
                    if warming_up and getattr(args, "guided_warmup_mode", "bc") == "ik":
                        teacher_replay.add(actor_obs=transition_batch["actor_obs"],
                                           action=transition_batch["action"])
                    elif teacher_action is not None:
                        teacher_replay.add(actor_obs=transition_batch["actor_obs"],
                                           action=teacher_action[finite_transition])
                        online_teacher_labels += count
                    reached = geometry["matched_flap_distance_m"][finite_transition].amin(-1) <= 0.25
                    reached |= hand_pinching[finite_transition].any(-1)
                    if reached.any():
                        goal_replay.add(**{key: value[reached] for key, value in transition_batch.items()})
                    reward_sum += reward[finite_transition].sum()
                    flap_distance = geometry["matched_flap_distance_m"][finite_transition]
                    front_distance = geometry["front_staging_distance_m"][finite_transition]
                    flap_distance_sum += flap_distance.sum(0)
                    front_distance_sum += front_distance.sum(0)
                    approach_diagnostics.record(front_distance, safety, finite_transition)
                    flap_near_count += (flap_distance < 0.10).sum(0)
                    front_near_count += (front_distance < 0.10).sum(0)
                    both_flap_near_count += (flap_distance < 0.10).all(-1).sum()
                    pinching = hand_pinching[finite_transition]
                    pinch_count += pinching.sum(0)
                    bilateral_pinch_count += pinching.all(-1).sum()
                    instantaneous_success_count += instantaneous_success[finite_transition].sum()
                    for name, value in success_gate_counts.items():
                        if name in geometry:
                            value.add_(geometry[name][finite_transition].sum())
                    if "hold_time_s" in geometry:
                        max_hold_time = torch.maximum(
                            max_hold_time, geometry["hold_time_s"][finite_transition].max())
                    closing = action[finite_transition][:, gripper_indices] > 0
                    close_count += closing.sum(0)
                    premature_close_count += (closing & (flap_distance >= 0.12)).sum(0)
                    if torso_diagnostics:
                        torso_valid = finite_transition & ~(terminated | truncated)
                        if bool(torso_valid.any()):
                            actual = env.scene["robot"].data.joint_pos[
                                :, torso_term._joint_ids][torso_valid]
                            target = torso_term._joint_targets[torso_valid]
                            pitch_error = (
                                actual.sum(-1) - torso_term._pitch_reference[torso_valid]
                            ).abs()
                            torso_pitch_error_sum += pitch_error.sum()
                            torso_tracking_error_sum += (actual - target).abs().mean(-1).sum()
                            torso_pitch_error_max = torch.maximum(
                                torso_pitch_error_max, pitch_error.max())
                            torso_diagnostic_count += int(torso_valid.sum().item())
                    reward_terms += (
                        env.reward_manager._step_reward[finite_transition].sum(0)
                        * env.step_dt
                    )
                    for name, value in breakdown_terms.items():
                        selected = value[finite_transition]
                        if name in progress_abs:
                            progress_abs[name] += selected.abs().sum()
                            progress_positive[name] += (selected > 0).sum()
                        if name not in breakdown_sums:
                            breakdown_sums[name] = selected.sum()
                            breakdown_nonzero[name] = int((selected != 0).sum().item())
                        else:
                            breakdown_sums[name] += selected.sum()
                            breakdown_nonzero[name] += int((selected != 0).sum().item())
                valid_transitions += count
                iteration_valid += count
                iteration_warmup += count if warming_up else 0
                iteration_guided_warmup += count if warming_up and guided_warmup else 0
                transitions += env.num_envs
                terminated_episodes += int(terminated.sum().item())
                timeout_episodes += int(truncated.sum().item())
                actor_obs = next_observations["policy"]
                critic_obs = _critic_state(next_observations)

            if valid_transitions >= warmup_target and replay.size >= args.batch_size and count:
                # Fit before the first Q update and the first unguided action.
                # Warmup samples contain measured new-dynamics rewards.
                if teacher_replay.size and not teacher_fitted:
                    teacher_batch = teacher_replay.sample(
                        min(teacher_replay.size, 100_000), env.device)
                    if demonstration is not None:
                        legacy = demonstration.sample(
                            max(1, len(teacher_batch["action"]) // 4), env.device)
                        teacher_batch = {
                            key: torch.cat((value, legacy[key]))
                            for key, value in teacher_batch.items()}
                    teacher_pretraining = agent.pretrain_actor(
                        teacher_batch["actor_obs"], teacher_batch["action"],
                        steps=getattr(args, "teacher_pretrain_steps", 5000),
                        batch_size=getattr(args, "demo_pretrain_batch_size", 256))
                    teacher_fitted = True
                    print(f"[V2 SAC] Online entry imitation: {teacher_pretraining}", flush=True)
                update_credit += args.updates_per_step * count / env.num_envs
                updates = int(update_credit)
                update_credit -= updates
                for _ in range(updates):
                    fraction = _demo_fraction(
                        getattr(args, "demo_batch_fraction", 0.0),
                        actor_updates, demo_decay_updates)
                    demo_count = round(args.batch_size * fraction) if demonstration else 0
                    demo_batch = demonstration.sample(demo_count, env.device) \
                        if demo_count else None
                    if demo_count and teacher_replay.size:
                        teacher_count = round(demo_count * 0.8)
                        offline = demonstration.sample(demo_count - teacher_count, env.device)
                        teacher = teacher_replay.sample(teacher_count, env.device)
                        demo_batch = {key: torch.cat((offline[key], value))
                                      for key, value in teacher.items()}
                    goal_count = round(args.batch_size * getattr(args, "goal_batch_fraction", 0.25)) \
                        if goal_replay.size else 0
                    success_count = min(success_replay.size, round(
                        args.batch_size * getattr(args, "success_batch_fraction", 0.05)))
                    batch = replay.sample(args.batch_size - goal_count - success_count, env.device)
                    if goal_count:
                        goals = goal_replay.sample(goal_count, env.device)
                        batch = {key: torch.cat((value, goals[key])) for key, value in batch.items()}
                    if success_count:
                        successes = success_replay.sample(success_count, env.device)
                        batch = {key: torch.cat((value, successes[key])) for key, value in batch.items()}
                    metrics = agent.update(
                        batch,
                        demonstration=demo_batch, demonstration_weight=(
                            fraction * getattr(args, "demo_bc_strength", 10.0) if demo_count else 0.0),
                        update_actor=optimizer_updates >= getattr(args, "critic_warmup_updates", 500))
                    demo_samples += demo_count if metrics["actor_updated"] else 0
                    actor_updates += int(metrics["actor_updated"])
                    success_samples += success_count
                    optimizer_updates += 1

        metrics.update(
            goal_replay_size=goal_replay.size,
            teacher_replay_size=teacher_replay.size,
            online_teacher_labels_this_iteration=online_teacher_labels,
            success_replay_size=success_replay.size,
            success_samples_this_iteration=success_samples,
            successful_warmup_episodes=warmup_successes,
            successful_sac_episodes=sac_successes,
            teacher_pretrain_steps=teacher_pretraining["steps"],
            teacher_pretrain_initial_mse=teacher_pretraining["initial_mse"],
            teacher_pretrain_final_mse=teacher_pretraining["final_mse"],
            rollout_policy=("mixed_warmup_sac" if 0 < iteration_warmup < iteration_valid
                            else (getattr(args, "guided_warmup_mode", "bc") + "_warmup"
                                  if iteration_warmup else "sac")),
            actor_encoded_dim=agent.actor_features.output_dim,
            actor_normalizer_count=float(agent.actor_normalizer.count),
            actor_normalizer_frozen=agent.config.freeze_actor_normalizer,
            reward_per_valid_step=(
                reward_sum.item() / iteration_valid if iteration_valid else None),
            transitions=transitions,
            valid_transitions=valid_transitions,
            valid_transitions_this_iteration=iteration_valid,
            warmup_target=warmup_target,
            warmup_complete=valid_transitions >= warmup_target,
            warmup_transitions_this_iteration=iteration_warmup,
            guided_warmup_transitions_this_iteration=iteration_guided_warmup,
            demo_pretrain_initial_mse=pretraining["initial_mse"],
            demo_pretrain_final_mse=pretraining["final_mse"],
            replay_size=replay.size,
            demo_replay_size=demonstration.size if demonstration else 0,
            demo_bc_samples_this_iteration=demo_samples,
            demo_bc_fraction=_demo_fraction(
                getattr(args, "demo_batch_fraction", 0.0), actor_updates,
                demo_decay_updates) if demonstration else 0.0,
            replay_gib=replay.bytes / 2**30,
            nonfinite_transitions=skipped_nonfinite,
            settling_transitions_skipped=skipped_settling,
            invalid_resets=invalid_resets,
            optimizer_updates=optimizer_updates,
            actor_updates=actor_updates,
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
        for name in progress_abs:
            metrics[f"reward_term_abs/{name}"] = (
                progress_abs[name].item() / iteration_valid if iteration_valid else None)
            metrics[f"reward_term_positive/{name}"] = (
                progress_positive[name].item() / iteration_valid if iteration_valid else None)
        for hand, index in (("left", 0), ("right", 1)):
            for label, values in (("flap", flap_distance_sum), ("front", front_distance_sum)):
                metrics[f"distance/{hand}_{label}_mean_m"] = (
                    values[index].item() / iteration_valid if iteration_valid else None)
            metrics[f"distance/{hand}_flap_under_0p1m"] = (
                flap_near_count[index].item() / iteration_valid if iteration_valid else None)
            metrics[f"distance/{hand}_front_under_0p1m"] = (
                front_near_count[index].item() / iteration_valid if iteration_valid else None)
        metrics["distance/both_flaps_under_0p1m"] = (
            both_flap_near_count.item() / iteration_valid if iteration_valid else None)
        metrics["grasp/bilateral_pinch_fraction"] = (
            bilateral_pinch_count.item() / iteration_valid if iteration_valid else None)
        metrics["grasp/instantaneous_success_fraction"] = (
            instantaneous_success_count.item() / iteration_valid if iteration_valid else None)
        for name, value in success_gate_counts.items():
            metrics[f"grasp/{name}_fraction"] = (
                value.item() / iteration_valid if iteration_valid else None)
        metrics["grasp/max_hold_time_s"] = max_hold_time.item()
        for hand, index in (("left", 0), ("right", 1)):
            metrics[f"grasp/{hand}_pinch_fraction"] = (
                pinch_count[index].item() / iteration_valid if iteration_valid else None)
            metrics[f"grasp/{hand}_close_fraction"] = (
                close_count[index].item() / iteration_valid if iteration_valid else None)
            metrics[f"grasp/{hand}_premature_close_fraction"] = (
                premature_close_count[index].item() / iteration_valid if iteration_valid else None)
        if torso_diagnostics:
            metrics["torso/pitch_error_mean_rad"] = (
                torso_pitch_error_sum.item() / torso_diagnostic_count
                if torso_diagnostic_count else None)
            metrics["torso/pitch_error_max_rad"] = torso_pitch_error_max.item()
            metrics["torso/joint_tracking_error_mean_rad"] = (
                torso_tracking_error_sum.item() / torso_diagnostic_count
                if torso_diagnostic_count else None)
        metrics.update({
            f"termination/{name}": count
            for name, count in termination_counts.items()
        })
        metrics.update(safety_diagnostics.report())
        metrics.update(approach_diagnostics.report())
        metrics["implausible_distance_transitions"] = int(skipped_implausible.item())
        metrics.update(_reset_settling_metrics(env))
        if guided_warmup is not None and hasattr(guided_warmup, "phase"):
            for phase in range(3):
                metrics[f"guide/phase_{phase}_envs"] = int((guided_warmup.phase == phase).sum())
        if str(env.device).startswith("cuda"):
            metrics.update(
                torch_peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            )
        log_metrics(directory, iteration, metrics)
        if iteration % args.save_interval == 0 or iteration == start + args.max_iterations:
            keep = None if getattr(args, "external_checkpoint_retention", False) \
                else args.keep_checkpoints
            payload = agent.checkpoint() | {
                "optimizer_updates": optimizer_updates,
                "actor_updates": actor_updates,
                "demo_decay_updates": demo_decay_updates,
                "teacher_fitted": teacher_fitted,
                "teacher_pretraining": teacher_pretraining,
                "teacher_imitation": teacher_replay.snapshot(),
                "success_replay": {
                    key: value[:success_replay.size].clone()
                    for key, value in success_replay.data.items()},
            }
            save_checkpoint(directory, payload, iteration, keep)
    return agent
