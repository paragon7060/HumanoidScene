"""Validated Task1 scenario manifests and physical-run argument construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from data_collection.task1.contract import ARM_JOINT_NAMES, WAIST_ARM_JOINT_NAMES


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = PROJECT_DIR / "configs" / "task1" / "scenarios"

_EXECUTION_NUMBER_KEYS = (
    "seed",
    "settle_steps",
    "approach_steps_per_waypoint",
    "close_steps",
    "closed_hold_steps",
    "retreat_steps_per_waypoint",
    "final_hold_steps",
    "capture_stride",
    "waypoint_tracking_tolerance_rad",
    "approach_box_motion_max_m",
    "retreat_box_motion_min_m",
    "retention_drift_max_m",
    "motor_obstruction_min_rad",
)

_OPTIONAL_EXECUTION_NUMBER_KEYS = (
    "rack_front_x_b_m",
    "partial_extraction_front_progress_min_m",
    "partial_extraction_front_inside_max_m",
    "final_hold_box_motion_max_m",
)


def _require_mapping(value, name: str) -> Mapping:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _resolve_project_path(value: str, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty project-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay inside the project")
    resolved = (PROJECT_DIR / path).resolve()
    if not resolved.is_file():
        raise ValueError(f"{name} does not exist: {value}")
    return resolved


def _validate_gripper_contract(data: Mapping) -> None:
    robot = _require_mapping(data.get("robot"), "robot")
    preset_name = robot.get("gripper_preset")
    presets = json.loads((PROJECT_DIR / "configs/grippers.json").read_text(encoding="utf-8"))[
        "presets"
    ]
    if preset_name not in presets:
        raise ValueError(f"unknown gripper_preset: {preset_name!r}")
    expected = _require_mapping(data.get("gripper_contract"), "gripper_contract")
    actual = presets[preset_name]
    actual_contract = {
        "actuator": {
            key: actual["actuator"][key]
            for key in ("effort_limit_sim", "stiffness", "damping")
        },
        "finger_contact": {
            key: actual["finger_contact"][key]
            for key in ("static_friction", "dynamic_friction")
        },
    }
    if expected != actual_contract:
        raise ValueError("gripper_contract does not match configs/grippers.json")


def load_scenario_config(path: str | Path) -> dict:
    """Load and validate one Task1 scenario manifest."""
    source = Path(path).expanduser().resolve()
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load Task1 scenario: {source}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Task1 scenario schema_version must be 1")
    if not isinstance(data.get("scenario_id"), str) or not data["scenario_id"]:
        raise ValueError("scenario_id must be a non-empty string")

    robot = _require_mapping(data.get("robot"), "robot")
    if robot != {
        "model": "s200062",
        "gripper_preset": "s200062_integrated",
        "tcp_frame": "endeffector_center",
    }:
        raise ValueError("single-box Task1 robot contract must use S200062 calibrated TCP")

    scene = _require_mapping(data.get("scene"), "scene")
    _resolve_project_path(scene.get("rack_box_poses"), "scene.rack_box_poses")
    if scene.get("target_box") != "medium_box_0":
        raise ValueError("current single-box Task1 target_box must be medium_box_0")
    if scene.get("clear_same_shelf_boxes") is not True:
        raise ValueError("current single-box Task1 must clear same-shelf obstacles")

    planning = _require_mapping(data.get("planning"), "planning")
    include_waist = planning.get("include_waist")
    arm_only = planning.get("arm_only_baseline")
    expected_names = WAIST_ARM_JOINT_NAMES if include_waist else ARM_JOINT_NAMES
    expected_dof = len(expected_names)
    if not isinstance(include_waist, bool) or not isinstance(arm_only, bool):
        raise ValueError("planning waist flags must be booleans")
    if arm_only == include_waist:
        raise ValueError("arm_only_baseline must be the inverse of include_waist")
    if planning.get("cspace_dof") != expected_dof:
        raise ValueError(f"planning.cspace_dof must be {expected_dof}")
    if planning.get("cspace_joint_names") != expected_names:
        raise ValueError("planning.cspace_joint_names does not match the selected DoF")
    if planning.get("torso_height_m") != 0.25:
        raise ValueError("verified single-box torso_height_m must be 0.25")

    execution = _require_mapping(data.get("execution"), "execution")
    if execution.get("initial_state") != "meta_default":
        raise ValueError("verified single-box initial_state must be meta_default")
    for key in _EXECUTION_NUMBER_KEYS:
        if isinstance(execution.get(key), bool) or not isinstance(execution.get(key), (int, float)):
            raise ValueError(f"execution.{key} must be numeric")
    for key in _OPTIONAL_EXECUTION_NUMBER_KEYS:
        if key in execution and (
            isinstance(execution[key], bool)
            or not isinstance(execution[key], (int, float))
        ):
            raise ValueError(f"execution.{key} must be numeric")
    if execution.get("active_gripper", "both") not in ("both", "left", "right"):
        raise ValueError("execution.active_gripper must be both, left, or right")
    if "box_size_m" in execution and (
        not isinstance(execution["box_size_m"], list)
        or len(execution["box_size_m"]) != 3
        or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0
            for value in execution["box_size_m"]
        )
    ):
        raise ValueError("execution.box_size_m must contain three positive numbers")
    for key in ("direct_approach_replay", "kinematic_direct_approach_render", "overwrite_video"):
        if not isinstance(execution.get(key), bool):
            raise ValueError(f"execution.{key} must be a boolean")

    _validate_gripper_contract(data)
    verification = _require_mapping(data.get("verification"), "verification")
    verification_state = verification.get("state")
    expected_passed = verification.get("expected_passed")
    if verification_state == "verified_partial_extraction" and expected_passed is not True:
        raise ValueError("verified_partial_extraction must set expected_passed true")
    if verification_state == "verified_failure" and expected_passed is not False:
        raise ValueError("verified_failure must set expected_passed false")
    data["_scenario_path"] = str(source)
    return data


def validate_plan_files(scenario: Mapping, approach_plan: Path, retreat_plan: Path) -> None:
    """Reject unsuccessful or cross-DoF plans before starting Isaac Sim."""
    expected = scenario["planning"]["cspace_joint_names"]
    expected_dof = scenario["planning"]["cspace_dof"]
    for label, path in (("approach", approach_plan), ("retreat", retreat_plan)):
        source = Path(path).expanduser().resolve()
        try:
            plan = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load {label} plan: {source}") from exc
        if plan.get("status") != "SUCCESS":
            raise ValueError(f"{label} plan status is not SUCCESS")
        if plan.get("joint_names") != expected:
            raise ValueError(
                f"{label} plan does not match scenario requires {expected_dof}DoF"
            )
        rows = plan.get("waypoint_q_rad")
        if not isinstance(rows, list) or not rows or any(
            not isinstance(row, list) or len(row) != expected_dof for row in rows
        ):
            raise ValueError(f"{label} plan has invalid {expected_dof}DoF waypoints")


def build_physical_executor_argv(
    scenario: Mapping,
    *,
    approach_plan: Path,
    retreat_plan: Path,
    output: Path,
    video_out: Path,
) -> list[str]:
    """Translate the resolved physical section into the canonical executor CLI."""
    execution = scenario["execution"]
    result = [
        "--approach-plan", str(Path(approach_plan)),
        "--retreat-plan", str(Path(retreat_plan)),
        "--output", str(Path(output)),
        "--video-out", str(Path(video_out)),
        "--rack-box-poses", str(
            _resolve_project_path(
                scenario["scene"]["rack_box_poses"], "scene.rack_box_poses"
            )
        ),
        "--initial-state", execution["initial_state"],
    ]
    flag_names = {
        "seed": "seed",
        "torso_height_m": "torso-height-m",
        "settle_steps": "settle-steps",
        "approach_steps_per_waypoint": "approach-steps-per-waypoint",
        "close_steps": "close-steps",
        "closed_hold_steps": "closed-hold-steps",
        "retreat_steps_per_waypoint": "retreat-steps-per-waypoint",
        "final_hold_steps": "final-hold-steps",
        "capture_stride": "capture-stride",
        "waypoint_tracking_tolerance_rad": "waypoint-tracking-tolerance-rad",
        "approach_box_motion_max_m": "approach-box-motion-max-m",
        "retreat_box_motion_min_m": "retreat-box-motion-min-m",
        "retention_drift_max_m": "retention-drift-max-m",
        "motor_obstruction_min_rad": "motor-obstruction-min-rad",
    }
    for key, cli_name in flag_names.items():
        result.extend((f"--{cli_name}", str(execution[key])))
    if "active_gripper" in execution:
        result.extend(("--active-gripper", execution["active_gripper"]))
    if "box_size_m" in execution:
        result.append("--box-size-m")
        result.extend(str(value) for value in execution["box_size_m"])
    for key, cli_name in (
        ("rack_front_x_b_m", "rack-front-x-b-m"),
        (
            "partial_extraction_front_progress_min_m",
            "partial-extraction-front-progress-min-m",
        ),
        (
            "partial_extraction_front_inside_max_m",
            "partial-extraction-front-inside-max-m",
        ),
        ("final_hold_box_motion_max_m", "final-hold-box-motion-max-m"),
    ):
        if key in execution:
            result.extend((f"--{cli_name}", str(execution[key])))
    for key, cli_name in (
        ("direct_approach_replay", "direct-approach-replay"),
        ("kinematic_direct_approach_render", "kinematic-direct-approach-render"),
        ("overwrite_video", "overwrite-video"),
    ):
        if execution[key]:
            result.append(f"--{cli_name}")
    return result


def scenario_digest(scenario: Mapping) -> str:
    """Return the SHA256 of the exact checked-in manifest."""
    return hashlib.sha256(Path(scenario["_scenario_path"]).read_bytes()).hexdigest()
