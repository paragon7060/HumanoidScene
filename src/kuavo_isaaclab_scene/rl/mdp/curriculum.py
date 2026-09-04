"""Gradually expand reset uncertainty; does not relax success predicates."""

def reset_difficulty(env, env_ids, ramp_steps=300_000):
    del env_ids
    value = min(env.common_step_counter / max(1, ramp_steps), 1.0)
    env._rl_difficulty = value
    return value
