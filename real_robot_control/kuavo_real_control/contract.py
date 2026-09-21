"""Load and validate the pinned real S63 control contract."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Tuple


ARM_JOINT_NAMES = tuple(
    "zarm_{}{}_joint".format(side, index)
    for side in "lr"
    for index in range(1, 8)
)


def _finite_tuple(raw: Any, name: str, length: int) -> Tuple[float, ...]:
    if not isinstance(raw, list) or len(raw) != length:
        raise ValueError("{} must contain {} values".format(name, length))
    values = tuple(float(value) for value in raw)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("{} must contain only finite values".format(name))
    return values


@dataclass(frozen=True)
class ControlConfig:
    robot_model: str
    operational_limits_provenance: str
    operational_limits_evidence: Dict[str, Any]
    arm_joint_names: Tuple[str, ...]
    sensor_joint_indices: Tuple[int, ...]
    joint_lower_rad: Tuple[float, ...]
    joint_upper_rad: Tuple[float, ...]
    position_limit_margin_rad: float
    max_command_velocity_rad_s: float
    max_command_acceleration_rad_s2: float
    max_source_step_rad: float
    max_start_error_rad: float
    max_tracking_error_rad: float
    tracking_error_timeout_s: float
    state_timeout_s: float
    external_command_quiet_s: float
    publish_hz: float
    handover_hold_s: float
    approach_settle_s: float
    max_approach_error_rad: float
    finish_hold_s: float
    rl_delta_scale_rad: float
    ros: Dict[str, str]

    @property
    def safe_lower_rad(self) -> Tuple[float, ...]:
        return tuple(value + self.position_limit_margin_rad for value in self.joint_lower_rad)

    @property
    def safe_upper_rad(self) -> Tuple[float, ...]:
        return tuple(value - self.position_limit_margin_rad for value in self.joint_upper_rad)


def load_config(path: Path) -> ControlConfig:
    raw = json.loads(Path(path).read_text())
    if raw.get("schema_version") != 1 or raw.get("robot_model") != "s63":
        raise ValueError("Expected schema_version=1 for robot_model=s63")
    provenance = raw.get("operational_limits_provenance")
    if provenance != "observed_real_vr_p99_operational_cap_not_hardware_rating":
        raise ValueError("S63 operational limits must declare their measured operational provenance")
    evidence = raw.get("operational_limits_evidence")
    evidence_fields = {
        "source_log",
        "joint_command_abs_velocity_p99_rad_s",
        "joint_command_abs_acceleration_p99_rad_s2",
        "controller_mode_transition_velocity_rad_s",
    }
    if not isinstance(evidence, dict) or set(evidence) != evidence_fields:
        raise ValueError("operational_limits_evidence must contain exactly {}".format(sorted(evidence_fields)))
    if not isinstance(evidence["source_log"], str) or not evidence["source_log"]:
        raise ValueError("operational_limits_evidence.source_log must be a non-empty string")
    for name in evidence_fields - {"source_log"}:
        value = float(evidence[name])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("operational_limits_evidence.{} must be finite and positive".format(name))
        evidence[name] = value
    names = tuple(raw.get("arm_joint_names", ()))
    if names != ARM_JOINT_NAMES:
        raise ValueError("S63 arm joint order differs from the verified left7,right7 contract")
    indices = tuple(raw.get("sensor_joint_indices", ()))
    if indices != tuple(range(4, 18)):
        raise ValueError("S63 sensor order must be body4,left7,right7,head2")
    lower = _finite_tuple(raw.get("joint_lower_rad"), "joint_lower_rad", 14)
    upper = _finite_tuple(raw.get("joint_upper_rad"), "joint_upper_rad", 14)
    if any(low >= high for low, high in zip(lower, upper)):
        raise ValueError("Every joint lower limit must be below its upper limit")

    scalar_names = (
        "position_limit_margin_rad",
        "max_command_velocity_rad_s",
        "max_command_acceleration_rad_s2",
        "max_source_step_rad",
        "max_start_error_rad",
        "max_tracking_error_rad",
        "tracking_error_timeout_s",
        "state_timeout_s",
        "external_command_quiet_s",
        "publish_hz",
        "handover_hold_s",
        "approach_settle_s",
        "max_approach_error_rad",
        "finish_hold_s",
        "rl_delta_scale_rad",
    )
    scalars = {}
    for name in scalar_names:
        value = float(raw.get(name, float("nan")))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("{} must be finite and positive".format(name))
        scalars[name] = value
    if any(high - low <= 2.0 * scalars["position_limit_margin_rad"] for low, high in zip(lower, upper)):
        raise ValueError("position limit margin removes a joint's usable range")
    if not 10.0 <= scalars["publish_hz"] <= 100.0:
        raise ValueError("publish_hz must be in [10, 100]")
    if scalars["rl_delta_scale_rad"] > scalars["max_source_step_rad"]:
        raise ValueError("One normalized RL action could exceed max_source_step_rad")
    evidence_pairs = (
        ("max_command_velocity_rad_s", "joint_command_abs_velocity_p99_rad_s"),
        ("max_command_acceleration_rad_s2", "joint_command_abs_acceleration_p99_rad_s2"),
    )
    for limit_name, evidence_name in evidence_pairs:
        observed = evidence[evidence_name]
        configured = scalars[limit_name]
        if configured < observed or configured > observed * 1.05:
            raise ValueError(
                "{} must be the observed p99 rounded up by no more than 5%".format(limit_name)
            )

    ros = raw.get("ros")
    required_ros = {
        "sensor_topic",
        "arm_command_topic",
        "arm_mode_service",
        "enable_topic",
        "disabled_topic",
        "claw_service",
    }
    if not isinstance(ros, dict) or set(ros) != required_ros:
        raise ValueError("ros must contain exactly {}".format(sorted(required_ros)))
    if any(not isinstance(value, str) or not value.startswith("/") for value in ros.values()):
        raise ValueError("Every ROS name must be an absolute topic/service name")

    return ControlConfig(
        robot_model="s63",
        operational_limits_provenance=provenance,
        operational_limits_evidence=dict(evidence),
        arm_joint_names=names,
        sensor_joint_indices=indices,
        joint_lower_rad=lower,
        joint_upper_rad=upper,
        ros=dict(ros),
        **scalars
    )
