"""Format cached RewardManager values without evaluating rewards twice."""


def step_contributions(terms, dt):
    # Isaac Lab 2.3.2 exposes weight * raw via get_active_iterable_terms.
    # Multiply by control dt exactly once, including discrete bonus terms.
    return {name: float(values[0]) * dt for name, values in terms}


FAILURE_LABELS = {
    "obstacle_collision": "OBSTACLE COLLISION", "floor_drop": "BOX DROPPED",
    "outside": "BASE OUT OF BOUNDS", "cargo_lost": "CARGO LOST", "settle_timeout": "SETTLE TIMEOUT",
}


def reward_summary(sample, status):
    """Large, fixed-size header derived from the retained PRE-reset snapshot."""
    if sample is None:
        return "READY - PRESS A", ["No physics sample yet"]
    reasons = sample.get("failure_reasons", [])
    if sample.get("failure"):
        label = FAILURE_LABELS.get(reasons[0], reasons[0]) if reasons else "UNKNOWN REASON"
        headline = "FAILED: " + label
    elif sample.get("success"):
        headline = "SUCCESS"
    elif sample.get("timeout"):
        headline = "TIME LIMIT REACHED"
    else:
        headline = "SUCCESS CONDITIONS" + (" (PAUSED)" if "PAUSED" in status else "")
    lines = ["FAIL: " + FAILURE_LABELS.get(reason, reason) for reason in reasons[1:]]
    if "obstacle_collision" in reasons:
        lines.append(f"Impact {sample['obstacle_force']:.2f} N > {sample.get('obstacle_limit', 20.):g} N")
    labels = {
        "grasp": f"Grasp {sample.get('required_hands', 'R')}",
        "height": f"Lift {sample['lift_cm']:.1f} / >{sample.get('required_lift_cm', 6.):g} cm",
        "tilt": f"Tilt {sample.get('tilt_deg', 0.):.1f} / <{sample.get('max_tilt_deg', 40.):g} deg",
        "hold": f"Hold {sample['hold']:.2f} / {sample.get('required_hold', .5):g} s",
        "initial_wait": "Initial wait", "cargo": "Cargo retained",
    }
    checks = sample.get("success_checks", {})
    # Show unmet checks first; missing dwell is never presented as success.
    for name, passed in sorted(checks.items(), key=lambda item: item[1]):
        if name == "initial_wait" and passed:
            continue
        lines.append(("OK: " if passed else "NEED: ") + labels.get(name, name))
    if sample.get("failure") or sample.get("success") or sample.get("timeout"):
        lines.append("LAST STEP - press B to reset")
    return headline, lines


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
                if not sample.get("collision_constraints_enabled", True):
                    lines.append("Collision termination/penalty: OFF (force display only)")
            lines.extend(sample.get("grasp_debug", []))
        lines += [f"{name}: {value:+.5f}" for name, value in sample["terms"].items()]
        if "reach_progress" in sample:
            lines.append("Reach new progress L/R: " + "/".join(f"{v:.5f}" for v in sample["reach_progress"]))
            lines.append("Reach best score L/R: " + "/".join(f"{v:.4f}" for v in sample["reach_best"]))
        lines += [f"TOTAL: {sample['total']:+.5f} | RETURN: {episode_return:+.3f}",
                  f"Lift: {sample['lift_cm']:.1f} cm | Hold: {sample['hold']:.2f} s",
                  f"Grasp L/R: {int(sample['left_grasp'])}/{int(sample['right_grasp'])}",
                  f"Flap distance L/R: {sample['left_distance_cm']:.1f}/{sample['right_distance_cm']:.1f} cm"]
    lines += ["A: run/pause   X: recenter   B: reset", "Y: panel   Index triggers: close/open"]
    return "\n".join(lines)
