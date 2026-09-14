"""Pure helpers for selecting the physical Task1 grasping hand."""


DEFAULT_KINEMATIC_APPROACH_RENDER_FRAMES = 73


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
