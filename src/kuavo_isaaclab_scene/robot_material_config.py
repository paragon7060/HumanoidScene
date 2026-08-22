"""Pure-Python Kuavo visual-material preset loading.

The source Kuavo URDF intentionally assigns white to most links.  These
presets provide a configurable rendering override without changing the robot
articulation, collision meshes, or instanceable USD payload.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
from typing import Any

from .paths import CONFIG_DIR, PACKAGE_CONFIG_DIR


ROBOT_MATERIAL_ENV = "KUAVO_ROBOT_MATERIAL"
ROBOT_MATERIAL_CONFIG_ENV = "KUAVO_ROBOT_MATERIAL_CONFIG"
DEFAULT_ROBOT_MATERIAL_CONFIG = PACKAGE_CONFIG_DIR / "robot_materials.json"


@dataclass(frozen=True)
class VisualMaterialSpec:
    diffuse_color: tuple[float, float, float]
    roughness: float
    metallic: float


@dataclass(frozen=True)
class LinkMaterialRule:
    pattern: str
    material: str


@dataclass(frozen=True)
class RobotMaterialSettings:
    name: str
    enabled: bool
    materials: dict[str, VisualMaterialSpec]
    link_rules: tuple[LinkMaterialRule, ...]
    config_path: Path

    def material_for_link(self, link_name: str) -> str | None:
        for rule in self.link_rules:
            if re.fullmatch(rule.pattern, link_name):
                return rule.material
        return None


def add_robot_material_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robot-material",
        default=os.environ.get(ROBOT_MATERIAL_ENV),
        metavar="PRESET",
        help=(
            "Kuavo visual palette from robot_materials.json "
            "(default: industrial_blue; use original for the source white palette)."
        ),
    )
    parser.add_argument(
        "--robot-material-config",
        type=Path,
        default=(
            Path(os.environ[ROBOT_MATERIAL_CONFIG_ENV])
            if os.environ.get(ROBOT_MATERIAL_CONFIG_ENV)
            else None
        ),
        metavar="JSON",
        help="Alternative Kuavo visual-material preset JSON.",
    )


def export_robot_material_cli(args: argparse.Namespace) -> None:
    preset = getattr(args, "robot_material", None)
    if preset:
        os.environ[ROBOT_MATERIAL_ENV] = str(preset)
    else:
        os.environ.pop(ROBOT_MATERIAL_ENV, None)
    config_path = getattr(args, "robot_material_config", None)
    if config_path is not None:
        os.environ[ROBOT_MATERIAL_CONFIG_ENV] = str(Path(config_path).expanduser().resolve())
    else:
        os.environ.pop(ROBOT_MATERIAL_CONFIG_ENV, None)


def _config_path(path: str | Path | None) -> Path:
    if path is not None:
        resolved = Path(path).expanduser().resolve()
    elif os.environ.get(ROBOT_MATERIAL_CONFIG_ENV):
        resolved = Path(os.environ[ROBOT_MATERIAL_CONFIG_ENV]).expanduser().resolve()
    else:
        working = CONFIG_DIR / "robot_materials.json"
        resolved = working if working.is_file() else DEFAULT_ROBOT_MATERIAL_CONFIG
    if not resolved.is_file():
        raise FileNotFoundError(f"Robot material configuration does not exist: {resolved}")
    return resolved


def _unit_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number in [0, 1].")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be a finite number in [0, 1].")
    return result


def _color(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly three RGB values.")
    return tuple(_unit_number(item, f"{label}[{index}]") for index, item in enumerate(value))


def load_robot_material_settings(
    preset: str | None = None,
    config_path: str | Path | None = None,
) -> RobotMaterialSettings:
    path = _config_path(config_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid robot material JSON {path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("presets"), dict):
        raise ValueError(f"{path} must contain a 'presets' object.")
    selected = preset or os.environ.get(ROBOT_MATERIAL_ENV) or payload.get("default")
    if not isinstance(selected, str) or not selected:
        raise ValueError(f"{path} must define a non-empty default material preset.")
    if selected not in payload["presets"]:
        choices = ", ".join(sorted(payload["presets"]))
        raise ValueError(f"Unknown robot material preset {selected!r}; available presets: {choices}.")
    raw = payload["presets"][selected]
    if not isinstance(raw, dict):
        raise ValueError(f"Robot material preset {selected!r} must be an object.")
    enabled = bool(raw.get("enabled", True))
    if not enabled:
        return RobotMaterialSettings(selected, False, {}, (), path)

    raw_materials = raw.get("materials")
    raw_rules = raw.get("link_rules")
    if not isinstance(raw_materials, dict) or not raw_materials:
        raise ValueError(f"Enabled material preset {selected!r} needs a non-empty materials object.")
    if not isinstance(raw_rules, list) or not raw_rules:
        raise ValueError(f"Enabled material preset {selected!r} needs non-empty link_rules.")
    materials: dict[str, VisualMaterialSpec] = {}
    for name, item in raw_materials.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid material identifier {name!r}.")
        if not isinstance(item, dict):
            raise ValueError(f"materials.{name} must be an object.")
        materials[name] = VisualMaterialSpec(
            diffuse_color=_color(item.get("diffuse_color"), f"materials.{name}.diffuse_color"),
            roughness=_unit_number(item.get("roughness"), f"materials.{name}.roughness"),
            metallic=_unit_number(item.get("metallic"), f"materials.{name}.metallic"),
        )

    rules: list[LinkMaterialRule] = []
    for index, item in enumerate(raw_rules):
        if not isinstance(item, dict):
            raise ValueError(f"link_rules[{index}] must be an object.")
        pattern = item.get("pattern")
        material = item.get("material")
        if not isinstance(pattern, str) or not pattern:
            raise ValueError(f"link_rules[{index}].pattern must be a non-empty string.")
        try:
            re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid link regex {pattern!r}: {exc}") from exc
        if material not in materials:
            raise ValueError(f"link_rules[{index}] references unknown material {material!r}.")
        rules.append(LinkMaterialRule(pattern, material))
    return RobotMaterialSettings(selected, True, materials, tuple(rules), path)


def resolve_robot_material_settings() -> RobotMaterialSettings:
    return load_robot_material_settings()
