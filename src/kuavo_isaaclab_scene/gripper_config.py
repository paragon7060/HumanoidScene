"""Pure-Python gripper preset loading shared by all launchers.

This module intentionally does not import Isaac Lab.  Launchers can therefore
resolve ``--gripper`` before :class:`isaaclab.app.AppLauncher` starts Kit.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

from .paths import ASSET_DIR, CONFIG_DIR, PACKAGE_CONFIG_DIR


GRIPPER_ENV = "KUAVO_GRIPPER"
GRIPPER_CONFIG_ENV = "KUAVO_GRIPPER_CONFIG"
DEFAULT_GRIPPER_CONFIG = PACKAGE_CONFIG_DIR / "grippers.json"
SIDES = ("left", "right")


@dataclass(frozen=True)
class GripperActuatorSettings:
    effort_limit_sim: float
    stiffness: float
    damping: float
    friction: float


@dataclass(frozen=True)
class GripperSideSettings:
    enabled: bool
    robot_mount_body: str
    robot_mount_pos: tuple[float, float, float]
    robot_mount_rot: tuple[float, float, float, float]
    usd_path: str | None = None
    attachment_mount_body: str | None = None


@dataclass(frozen=True)
class GripperSettings:
    name: str
    enabled: bool
    usd_path: str
    attachment_mount_body: str
    joint_names: tuple[str, ...]
    default_joint_pos: dict[str, float]
    open_command: dict[str, float]
    close_command: dict[str, float]
    pinch_close_threshold_m: float
    actuator: GripperActuatorSettings
    sides: dict[str, GripperSideSettings]
    config_path: Path
    integrated: bool = False

    @property
    def active_sides(self) -> tuple[str, ...]:
        if not self.enabled:
            return ()
        return tuple(side for side in SIDES if self.sides[side].enabled)

    def usd_path_for(self, side: str) -> str:
        side_cfg = self.sides[side]
        return side_cfg.usd_path or self.usd_path

    def attachment_mount_body_for(self, side: str) -> str:
        side_cfg = self.sides[side]
        return side_cfg.attachment_mount_body or self.attachment_mount_body

    def asset_name_for(self, side: str) -> str:
        return "robot" if self.integrated else f"{side}_gripper"

    def joint_names_for(self, side: str) -> tuple[str, ...]:
        return tuple(name.replace("{side}", side[0]) for name in self.joint_names)

    @property
    def joint_name_exprs_for_robot(self) -> tuple[str, ...]:
        """Return regexes covering both hands of an integrated preset."""
        return tuple(name.replace("{side}", "[lr]") for name in self.joint_names)

    def command_for(self, side: str, command: dict[str, float]) -> dict[str, float]:
        return {name.replace("{side}", side[0]): value for name, value in command.items()}

    def command_for_all_sides(self, command: dict[str, float]) -> dict[str, float]:
        result: dict[str, float] = {}
        for side in self.active_sides:
            result.update(self.command_for(side, command))
        return result

    def body_names_for(self, side: str) -> str:
        if self.integrated:
            if self.name == "s56_qiangnao":
                return rf"{side[0]}_(palm|thumb_.*|index_.*|middle_.*|ring_.*|little_.*)"
            return rf"{side[0]}_(twofinger_base|[fb]_(bar_[1-4]|finger))"
        return ".*"


def add_gripper_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--gripper",
        default=os.environ.get(GRIPPER_ENV),
        metavar="PRESET",
        help="Gripper preset from grippers.json (default: file's 'default'; use 'none' to disable).",
    )
    parser.add_argument(
        "--gripper-config",
        type=Path,
        default=Path(os.environ[GRIPPER_CONFIG_ENV]) if os.environ.get(GRIPPER_CONFIG_ENV) else None,
        metavar="JSON",
        help="Alternative gripper preset JSON. Relative USD paths are resolved beside this file.",
    )


def export_gripper_cli(args: argparse.Namespace) -> None:
    if getattr(args, "gripper", None):
        os.environ[GRIPPER_ENV] = str(args.gripper)
    else:
        os.environ.pop(GRIPPER_ENV, None)
    config_path = getattr(args, "gripper_config", None)
    if config_path is not None:
        os.environ[GRIPPER_CONFIG_ENV] = str(Path(config_path).expanduser().resolve())
    else:
        os.environ.pop(GRIPPER_CONFIG_ENV, None)


def _config_path(path: str | Path | None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
    elif os.environ.get(GRIPPER_CONFIG_ENV):
        resolved = Path(os.environ[GRIPPER_CONFIG_ENV]).expanduser().resolve()
    else:
        working = CONFIG_DIR / "grippers.json"
        resolved = working if working.is_file() else DEFAULT_GRIPPER_CONFIG
    if not resolved.is_file():
        raise FileNotFoundError(f"Gripper configuration does not exist: {resolved}")
    return resolved


def _number(value: Any, label: str, *, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result) or (non_negative and result < 0.0):
        qualifier = "a finite non-negative number" if non_negative else "finite"
        raise ValueError(f"{label} must be {qualifier}.")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f"{label} must contain exactly {size} numbers.")
    result = tuple(_number(item, f"{label}[{index}]") for index, item in enumerate(value))
    if size == 4:
        norm = math.sqrt(sum(item * item for item in result))
        if norm < 1.0e-8:
            raise ValueError(f"{label} quaternion must be non-zero.")
        result = tuple(item / norm for item in result)
    return result


def _command(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{label} must be a non-empty regex-to-number object.")
    return {str(key): _number(item, f"{label}.{key}") for key, item in value.items()}


def _resolve_usd_path(value: str, config_path: Path) -> str:
    package_asset_token = "${KUAVO_PACKAGE_ASSET_DIR}"
    if value.startswith(package_asset_token):
        value = str(ASSET_DIR) + value[len(package_asset_token) :]
    value = os.path.expandvars(value)
    # Isaac tokens are expanded after Kit starts by gripper_runtime.py.
    if value.startswith("${ISAAC_NUCLEUS_DIR}") or value.startswith("${NUCLEUS_ASSET_ROOT_DIR}"):
        return value
    if "://" in value or value.startswith("omniverse:"):
        return value
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = config_path.parent / candidate
    return str(candidate.resolve())


def load_gripper_settings(
    preset: str | None = None,
    config_path: str | Path | None = None,
) -> GripperSettings:
    path = _config_path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid gripper JSON {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("presets"), dict):
        raise ValueError(f"{path} must contain a 'presets' object.")
    selected = preset or os.environ.get(GRIPPER_ENV) or payload.get("default")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"{path} must define a non-empty 'default' gripper preset.")
    if selected not in payload["presets"]:
        choices = ", ".join(sorted(payload["presets"]))
        raise ValueError(f"Unknown gripper preset {selected!r}; available presets: {choices}.")
    raw = payload["presets"][selected]
    if not isinstance(raw, dict):
        raise ValueError(f"Gripper preset {selected!r} must be an object.")
    enabled = bool(raw.get("enabled", True))
    integrated = bool(raw.get("integrated", False))
    if not enabled:
        disabled_side = GripperSideSettings(False, "", (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        return GripperSettings(
            selected, False, "", "", (), {}, {}, {}, 0.035,
            GripperActuatorSettings(0.0, 0.0, 0.0, 0.0),
            {side: disabled_side for side in SIDES}, path,
        )

    for key in ("usd_path", "attachment_mount_body", "joint_names", "default_joint_pos", "open_command", "close_command", "actuator", "sides"):
        if key not in raw:
            raise ValueError(f"Enabled gripper preset {selected!r} is missing {key!r}.")
    if not isinstance(raw["usd_path"], str) or not raw["usd_path"]:
        raise ValueError(f"Gripper preset {selected!r} usd_path must be non-empty.")
    if not isinstance(raw["joint_names"], list) or not raw["joint_names"]:
        raise ValueError(f"Gripper preset {selected!r} joint_names must be a non-empty list.")
    actuator = raw["actuator"]
    if not isinstance(actuator, dict):
        raise ValueError(f"Gripper preset {selected!r} actuator must be an object.")
    actuator_cfg = GripperActuatorSettings(
        effort_limit_sim=_number(actuator.get("effort_limit_sim"), "actuator.effort_limit_sim", non_negative=True),
        stiffness=_number(actuator.get("stiffness"), "actuator.stiffness", non_negative=True),
        damping=_number(actuator.get("damping"), "actuator.damping", non_negative=True),
        friction=_number(actuator.get("friction", 0.0), "actuator.friction", non_negative=True),
    )
    raw_sides = raw["sides"]
    if not isinstance(raw_sides, dict):
        raise ValueError(f"Gripper preset {selected!r} sides must be an object.")
    sides: dict[str, GripperSideSettings] = {}
    for side in SIDES:
        item = raw_sides.get(side, {"enabled": False})
        if not isinstance(item, dict):
            raise ValueError(f"sides.{side} must be an object.")
        side_enabled = bool(item.get("enabled", True))
        if side_enabled and not isinstance(item.get("robot_mount_body"), str):
            raise ValueError(f"sides.{side}.robot_mount_body must be a string.")
        override = item.get("usd_path")
        if override is not None and (not isinstance(override, str) or not override):
            raise ValueError(f"sides.{side}.usd_path must be a non-empty string when set.")
        mount_override = item.get("attachment_mount_body")
        if mount_override is not None and (not isinstance(mount_override, str) or not mount_override):
            raise ValueError(f"sides.{side}.attachment_mount_body must be non-empty when set.")
        sides[side] = GripperSideSettings(
            enabled=side_enabled,
            robot_mount_body=str(item.get("robot_mount_body", "")),
            robot_mount_pos=_vector(item.get("robot_mount_pos", [0.0, 0.0, 0.0]), 3, f"sides.{side}.robot_mount_pos"),
            robot_mount_rot=_vector(item.get("robot_mount_rot", [1.0, 0.0, 0.0, 0.0]), 4, f"sides.{side}.robot_mount_rot"),
            usd_path=_resolve_usd_path(override, path) if override else None,
            attachment_mount_body=mount_override,
        )
    threshold = _number(raw.get("pinch_close_threshold_m", 0.035), "pinch_close_threshold_m", non_negative=True)
    if threshold <= 0.0:
        raise ValueError("pinch_close_threshold_m must be positive.")
    return GripperSettings(
        name=selected,
        enabled=True,
        usd_path=_resolve_usd_path(raw["usd_path"], path),
        attachment_mount_body=str(raw["attachment_mount_body"]),
        joint_names=tuple(str(name) for name in raw["joint_names"]),
        default_joint_pos=_command(raw["default_joint_pos"], "default_joint_pos"),
        open_command=_command(raw["open_command"], "open_command"),
        close_command=_command(raw["close_command"], "close_command"),
        pinch_close_threshold_m=threshold,
        actuator=actuator_cfg,
        sides=sides,
        config_path=path,
        integrated=integrated,
    )


def resolve_gripper_settings() -> GripperSettings:
    # The full S200062 articulation already contains both two-finger grippers.
    # Select their integrated control preset unless the caller explicitly
    # disables gripper actions with KUAVO_GRIPPER=none.
    from .robot_model import default_gripper_for_model, resolve_robot_model, validate_robot_gripper

    model = resolve_robot_model()
    selected = os.environ.get(GRIPPER_ENV)
    if selected is None:
        selected = default_gripper_for_model(model)
    settings = load_gripper_settings(selected)
    validate_robot_gripper(model, settings.name)
    return settings


BASE_TELEOP_ACTION_NAMES = (
    "left_dx", "left_dy", "left_dz", "left_droll", "left_dpitch", "left_dyaw",
    "right_dx", "right_dy", "right_dz", "right_droll", "right_dpitch", "right_dyaw",
    "head_yaw", "head_pitch",
)


def teleop_action_names(settings: GripperSettings) -> tuple[str, ...]:
    return BASE_TELEOP_ACTION_NAMES + tuple(f"{side}_gripper" for side in settings.active_sides)


def gripper_teleop_action(
    settings: GripperSettings,
    left_pinch_m: float,
    right_pinch_m: float,
) -> tuple[float, ...]:
    """Return binary manager inputs: positive=open, negative=close."""
    pinches = {"left": float(left_pinch_m), "right": float(right_pinch_m)}
    return tuple(
        -1.0
        if math.isfinite(pinches[side]) and pinches[side] <= settings.pinch_close_threshold_m
        else 1.0
        for side in settings.active_sides
    )
