"""Evaluate native SAC, diffusion BC or DPPO with common task outcome metrics."""

import json
import torch
from ..algorithms.sac import SAC, SACConfig
from ..algorithms.diffusion import load_diffusion


def evaluate(env, args, directory, state):
    if state["algorithm"] == "sac":
        agent = SAC(state["obs_dim"], state["action_dim"], SACConfig(**state["config"]), env.device)
        agent.restore(state, training=False)
        policy = lambda obs: agent.act(obs, deterministic=True)
    else:
        agent = load_diffusion(state, env.device)
        policy = lambda obs: agent.sample(obs, deterministic=not args.stochastic_eval)[0][:, 0]
    obs = env.reset()[0]["policy"]
    outcomes = []
    with torch.no_grad():
        while len(outcomes) < args.episodes:
            next_dict, _, _, _, _ = env.step(policy(obs))
            obs = next_dict["policy"]
            latest = getattr(env, "_rl_last_outcomes", {})
            if latest.get("step") == env.common_step_counter:
                outcomes.extend(latest["episodes"])
    outcomes = outcomes[:args.episodes]
    report = dict(algorithm=state["algorithm"], episodes=len(outcomes),
                  success_rate=sum(o["success"] for o in outcomes) / len(outcomes), outcomes=outcomes)
    (directory / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    print(f"[RL] Evaluation success rate={report['success_rate']:.3f}; {directory}", flush=True)
