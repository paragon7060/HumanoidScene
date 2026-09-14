"""Pure helpers for selecting the physical Task1 grasping hand."""

import math


DEFAULT_KINEMATIC_APPROACH_RENDER_FRAMES = 73


def executor_target_keys(
    paired_boxes: list[str] | tuple[str, ...] | None,
    clear_same_shelf_boxes: bool,
) -> tuple[str, ...]:
    """Resolve the exact physical targets without silently removing pair mates."""
    if paired_boxes is None:
        return ("medium_box_0",)
    keys = tuple(paired_boxes)
    if len(keys) != 2 or len(set(keys)) != 2 or not all(keys):
        raise ValueError("paired_boxes must contain two distinct box keys")
    if clear_same_shelf_boxes:
        raise ValueError("both paired boxes must remain in the scene")
    return keys


def _distance(first, second) -> float:
    if len(first) != 3 or len(second) != 3:
        raise ValueError("positions must contain xyz")
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(first, second, strict=True)))


def _midpoint(first, second) -> list[float]:
    return [(float(a) + float(b)) * 0.5 for a, b in zip(first, second, strict=True)]


def paired_retention_metrics(
    samples: list[dict],
    paired_boxes: tuple[str, str],
    active_gripper: str,
    closed_box_positions: dict[str, list[float]],
) -> dict:
    """Measure whether two boxes follow one active TCP as a retained pair."""
    if active_gripper not in ("left", "right"):
        raise ValueError("paired retention requires one active gripper")
    keys = executor_target_keys(paired_boxes, False)
    if set(closed_box_positions) != set(keys):
        raise ValueError("closed_box_positions must contain exactly the paired boxes")
    rows = [row for row in samples if row.get("phase") in ("retreat", "final_hold")]
    if not rows:
        raise ValueError("paired retention requires retreat or final-hold samples")
    hand_index = {"left": 0, "right": 1}[active_gripper]
    closed_center = _midpoint(*(closed_box_positions[key] for key in keys))
    first_hand = rows[0]["eef_positions_b_m"][hand_index]
    closed_relative = [closed_center[i] - float(first_hand[i]) for i in range(3)]
    initial_separation = _distance(*(closed_box_positions[key] for key in keys))
    separation_drifts = []
    hand_center_drifts = []
    for row in rows:
        positions = row["box_body_positions_b_m"]
        if set(positions) != set(keys):
            raise ValueError("every pair sample must contain exactly both boxes")
        center = _midpoint(*(positions[key] for key in keys))
        hand = row["eef_positions_b_m"][hand_index]
        relative = [center[i] - float(hand[i]) for i in range(3)]
        separation_drifts.append(abs(_distance(*(positions[key] for key in keys)) - initial_separation))
        hand_center_drifts.append(_distance(relative, closed_relative))
    final_positions = rows[-1]["box_body_positions_b_m"]
    hold_rows = [row for row in rows if row["phase"] == "final_hold"]
    if not hold_rows:
        raise ValueError("paired retention requires final-hold samples")
    first_hold = hold_rows[0]["box_body_positions_b_m"]
    last_hold = hold_rows[-1]["box_body_positions_b_m"]
    return {
        "box_robotward_progress_m": {
            key: round(float(closed_box_positions[key][0]) - float(final_positions[key][0]), 12)
            for key in keys
        },
        "pair_separation_drift_max_m": max(separation_drifts),
        "hand_pair_center_drift_max_m": max(hand_center_drifts),
        "final_hold_box_motion_m": {
            key: _distance(first_hold[key], last_hold[key]) for key in keys
        },
    }


def paired_box_acceptance(
    metrics: dict,
    *,
    approach_box_motion_m: dict[str, float],
    approach_box_motion_max_m: float,
    progress_min_m: float,
    pair_separation_drift_max_m: float,
    hand_pair_center_drift_max_m: float,
    final_hold_box_motion_max_m: float,
    active_motor_obstruction: bool,
    tracking_passed: bool,
) -> tuple[bool, str]:
    """Apply the fixed two-box, one-hand physical acceptance contract."""
    progress = metrics["box_robotward_progress_m"]
    final_motion = metrics["final_hold_box_motion_m"]
    keys = set(progress)
    if set(approach_box_motion_m) != keys or set(final_motion) != keys or len(keys) != 2:
        raise ValueError("paired acceptance requires the same exact two boxes")
    passed = bool(
        all(float(approach_box_motion_m[key]) <= approach_box_motion_max_m for key in keys)
        and all(float(progress[key]) >= progress_min_m for key in keys)
        and float(metrics["pair_separation_drift_max_m"]) <= pair_separation_drift_max_m
        and float(metrics["hand_pair_center_drift_max_m"]) <= hand_pair_center_drift_max_m
        and all(float(final_motion[key]) <= final_hold_box_motion_max_m for key in keys)
        and active_motor_obstruction
        and tracking_passed
    )
    return passed, "paired_single_hand_partial_extraction"


def resolved_kinematic_capture_frame_count(
    waypoint_count: int, requested_frame_count: int | None
) -> int:
    """Resolve the normal-speed default without oversampling short paths."""
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2")
    if requested_frame_count is None:
        return min(DEFAULT_KINEMATIC_APPROACH_RENDER_FRAMES, waypoint_count)
    return requested_frame_count


def evenly_spaced_capture_indices(
    waypoint_count: int, frame_count: int
) -> tuple[int, ...]:
    """Select exact, monotonic video samples while retaining all replay waypoints."""
    if waypoint_count < 2:
        raise ValueError("waypoint_count must be at least 2")
    if frame_count < 2:
        raise ValueError("frame_count must be at least 2")
    if frame_count > waypoint_count:
        raise ValueError("frame_count cannot exceed waypoint_count")
    scale = (waypoint_count - 1) / (frame_count - 1)
    return tuple(round(index * scale) for index in range(frame_count))


def _box_x_extent(pose, box_size_m) -> float:
    """Return the world-X half extent of an oriented box pose in wxyz order."""
    if len(pose) != 7 or len(box_size_m) != 3:
        raise ValueError("expected xyz+wxyz pose and xyz box size")
    _, _, _, w, x, y, z = (float(value) for value in pose)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm == 0.0:
        raise ValueError("box quaternion must be nonzero")
    w, x, y, z = (value / norm for value in (w, x, y, z))
    half = [float(value) * 0.5 for value in box_size_m]
    if any(value <= 0.0 for value in half):
        raise ValueError("box size must be positive")
    rotation_x = (
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y - z * w),
        2.0 * (x * z + y * w),
    )
    return sum(abs(axis) * size for axis, size in zip(rotation_x, half, strict=True))


def box_extraction_metrics(
    closed_pose,
    final_pose,
    *,
    box_size_m,
    rack_front_x_b_m: float,
    front_progress_min_m: float,
    front_inside_max_m: float,
    final_hold_box_motion_m: float,
    final_hold_box_motion_max_m: float,
) -> dict:
    """Measure partial/full extraction along the rack's robotward (-X) direction."""
    closed_extent = _box_x_extent(closed_pose, box_size_m)
    final_extent = _box_x_extent(final_pose, box_size_m)
    closed_front = float(closed_pose[0]) - closed_extent
    final_front = float(final_pose[0]) - final_extent
    final_back = float(final_pose[0]) + final_extent
    front_progress = closed_front - final_front
    front_inside = final_front - float(rack_front_x_b_m)
    settled = float(final_hold_box_motion_m) <= float(final_hold_box_motion_max_m)
    partial = bool(
        front_progress >= float(front_progress_min_m)
        and front_inside <= float(front_inside_max_m)
        and settled
    )
    full = bool(final_back <= float(rack_front_x_b_m) and settled)
    return {
        "closed_box_front_x_b_m": closed_front,
        "final_box_front_x_b_m": final_front,
        "final_box_back_x_b_m": final_back,
        "box_robotward_front_progress_m": front_progress,
        "final_box_front_inside_rack_m": front_inside,
        "partial_extraction_success": partial,
        "full_extraction_success": full,
    }


def physical_acceptance(
    active_gripper: str,
    *,
    approach_contact_free: bool,
    partial_extraction_success: bool,
    stable_retention: bool,
) -> tuple[bool, str]:
    """Select the task outcome gate without weakening the bimanual contract."""
    active_hand_indices(active_gripper)
    if active_gripper == "both":
        return bool(approach_contact_free and stable_retention), "stable_bimanual_retention"
    return bool(approach_contact_free and partial_extraction_success), "partial_extraction"


def active_hand_indices(active_gripper: str) -> tuple[int, ...]:
    """Map the public gripper mode to left/right hand indices."""
    try:
        return {"both": (0, 1), "left": (0,), "right": (1,)}[active_gripper]
    except KeyError as error:
        raise ValueError(f"unsupported active gripper: {active_gripper!r}") from error


def closed_motor_targets(open_positions, motor_names, active_gripper: str) -> list[float]:
    """Close only motors belonging to the selected hand."""
    if len(open_positions) != len(motor_names):
        raise ValueError("motor positions and names must have the same length")
    active_hand_indices(active_gripper)
    result = []
    for position, name in zip(open_positions, motor_names, strict=True):
        if name.startswith("l_"):
            side = "left"
        elif name.startswith("r_"):
            side = "right"
        else:
            raise ValueError(f"cannot infer gripper side from motor name: {name!r}")
        result.append(0.0 if active_gripper in ("both", side) else float(position))
    return result


def retention_reference(eef_positions, active_gripper: str) -> list[float]:
    """Return the active TCP, or the midpoint used by the bimanual gate."""
    if len(eef_positions) != 2 or any(len(position) != 3 for position in eef_positions):
        raise ValueError("expected left/right xyz TCP positions")
    indices = active_hand_indices(active_gripper)
    return [
        sum(float(eef_positions[index][axis]) for index in indices) / len(indices)
        for axis in range(3)
    ]


def motor_obstruction_gates(
    hand_obstruction, active_gripper: str, threshold: float
) -> tuple[bool, bool | None]:
    """Evaluate required hands and suppress a bimanual claim in single-hand modes."""
    if len(hand_obstruction) != 2:
        raise ValueError("expected left/right motor obstruction values")
    indices = active_hand_indices(active_gripper)
    active = min(float(hand_obstruction[index]) for index in indices) >= threshold
    bilateral = (
        min(float(value) for value in hand_obstruction) >= threshold
        if active_gripper == "both"
        else None
    )
    return bool(active), None if bilateral is None else bool(bilateral)
