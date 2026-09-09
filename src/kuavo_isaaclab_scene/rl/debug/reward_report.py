"""Format cached RewardManager values without evaluating rewards twice."""


def step_contributions(terms, dt):
    # Isaac Lab 2.3.2 exposes weight * raw via get_active_iterable_terms.
    # Multiply by control dt exactly once, including discrete bonus terms.
    return {name: float(values[0]) * dt for name, values in terms}


def format_report(sample, status, episode_return):
    lines = [f"RL REWARD | {status}", "Actual weighted reward / control step"]
    if sample is None:
        lines.append("Waiting for first physics step")
    else:
        if "blocked_checks" in sample:
            lines.append("CHECK: " + (", ".join(sample["blocked_checks"]) or "pass - keep holding"))
            lines.append("FAIL: " + (", ".join(sample.get("failure_reasons", [])) or "none"))
            if "obstacle_force" in sample:
                lines.append(f"Obstacle force: {sample['obstacle_force']:.3f} N")
            lines.extend(sample.get("grasp_debug", []))
        lines += [f"{name}: {value:+.5f}" for name, value in sample["terms"].items()]
        lines += [f"TOTAL: {sample['total']:+.5f} | RETURN: {episode_return:+.3f}",
                  f"Lift: {sample['lift_cm']:.1f} cm | Hold: {sample['hold']:.2f} s",
                  f"Grasp L/R: {int(sample['left_grasp'])}/{int(sample['right_grasp'])}",
                  f"Flap distance L/R: {sample['left_distance_cm']:.1f}/{sample['right_distance_cm']:.1f} cm"]
    lines += ["A: run/pause   X: recenter   B: reset", "Y: panel   Index triggers: close/open"]
    return "\n".join(lines)
