"""Explain which v2 grasp safety predicate ended a demonstration attempt.

The termination manager exposes one boolean `unsafe` term for seven different
predicates, so an aborted attempt cannot be diagnosed from the episode label
alone.  These helpers format the pre-reset predicate snapshot the environment
already published; they never measure contacts, poses or velocities again.
"""

from __future__ import annotations

from dataclasses import dataclass


SAFETY_CAUSES = (
    "robot_rack_collision",
    "self_collision",
    "obstacle_collision",
    "workspace_limit",
    "box_drop",
    "box_lift_limit",
    "box_speed_limit",
)


@dataclass(frozen=True)
class BoxSafetyValues:
    """Target-box snapshot behind the drop, lift and speed predicates."""

    height_m: float
    lift_m: float
    linear_speed: float
    angular_speed: float


def unsafe_causes(safety) -> tuple[str, ...]:
    """Return every safety predicate that is true for the first environment.

    `safety` is the pre-reset predicate mapping published as
    `transition_safety`, so a true `unsafe` term always names at least one
    predicate here.
    """
    missing = set(SAFETY_CAUSES) - set(safety)
    if missing:
        raise KeyError(f"Safety snapshot is missing predicates: {sorted(missing)}")
    return tuple(name for name in SAFETY_CAUSES if bool(safety[name][0]))


def safety_measurements(safety, cfg, box: BoxSafetyValues | None = None) -> str:
    """Format measured safety values next to the limit that each one must obey."""
    values = [
        f"rack {float(safety['rack_force_n'][0]):.1f}/"
        f"{float(cfg.multi_box.rack_contact_force):.1f} N",
        f"obstacle {float(safety['obstacle_force_n'][0]):.1f}/"
        f"{float(cfg.task.obstacle_contact_force):.1f} N",
        f"base {float(safety['base_distance_m'][0]):.2f}/"
        f"{float(cfg.multi_box.workspace_radius):.2f} m",
    ]
    if cfg.multi_box.self_collision_enabled:
        values.append(
            f"self clearance {100 * float(safety['self_collision_distance_m'][0]):.2f}/"
            f"{100 * float(cfg.multi_box.self_collision_clearance):.2f} cm")
    else:
        values.append("self-collision off")
    if box is not None:
        values.append(f"box z {box.height_m:.2f} m")
        values.append(
            f"lift {box.lift_m:.2f}/{float(cfg.multi_box.max_box_lift_height):.2f} m")
        values.append(
            f"box speed {box.linear_speed:.2f}/"
            f"{float(cfg.multi_box.max_box_linear_speed):.1f} m/s, "
            f"{box.angular_speed:.2f}/"
            f"{float(cfg.multi_box.max_box_angular_speed):.1f} rad/s")
    return " | ".join(values)
