"""Shared, simulator-independent rack-box layout configuration.

Shelf numbers exposed to users are one-based: 1 is the bottom shelf and 3 is
the top shelf.  The resulting spawn plan is consumed by both ``scene.py`` and
``manager_env.py`` so the two runtime variants cannot silently diverge.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

from .paths import CONFIG_DIR
from .workcell_layout import local_point_to_world, local_quat_to_world
from .workcell_layout import rack_tier_surface_z, scale as layout_scale


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
RackBoxLayout = dict[int, list[str]]
DEFAULT_RACK_BOX_POSE_PATH = CONFIG_DIR / "rack_box_poses.json"

BOX_TYPE_LABELS: dict[str, str] = {
    "small": "SmallBox",
    "medium": "MediumBox",
    "large": "LargeBox",
    "xlarge": "XLargeBox",
}
BOX_TYPE_SCENE_KEYS: dict[str, str] = {
    "small": "small_box",
    "medium": "medium_box",
    "large": "large_box",
    "xlarge": "xlarge_box",
}
BOX_TYPE_ALIASES: dict[str, str] = {
    "s": "small",
    "small": "small",
    "smallbox": "small",
    "m": "medium",
    "medium": "medium",
    "mediumbox": "medium",
    "l": "large",
    "large": "large",
    "largebox": "large",
    "xl": "xlarge",
    "xlarge": "xlarge",
    "x-large": "xlarge",
    "xlargebox": "xlarge",
}

# Edit this dictionary for a persistent code-level default.  A shelf value can
# be either an ordered list (for example ["small", "medium"]) or a count map
# (for example {"small": 2, "medium": 1}).  CLI/JSON values override it.
DEFAULT_RACK_BOX_LAYOUT: Mapping[int, Sequence[str] | Mapping[str, int]] = {
    1: [],
    2: [],
    3: [],
}

# All four supplied USD files currently have the same authored bounding box.
# These editable target dimensions make the named variants useful while
# retaining their original flap articulation.  Order is native (X, Y, Z):
# rack-width, rack-depth, height after the rack-aligned spawn rotation.
BOX_NATIVE_SIZE_M: Vec3 = (1.01, 1.01, 1.515)
BOX_DIMENSIONS_M: dict[str, Vec3] = {
    "small": (0.26, 0.22, 0.18),
    "medium": (0.32, 0.25, 0.22),
    "large": (0.37, 0.29, 0.26),
    "xlarge": (0.42, 0.33, 0.30),
}

# Two instances of each supplied USD are always spawned.  Instances omitted
# from a shelf layout stay in these floor staging slots.
STAGING_BOX_POSITIONS: dict[str, Vec3] = {
    "SmallBox_0": (2.00, 1.60, 0.02),
    "SmallBox_1": (2.00, 1.94, 0.02),
    "MediumBox_0": (2.40, 1.60, 0.02),
    "MediumBox_1": (2.40, 1.94, 0.02),
    "LargeBox_0": (2.82, 1.60, 0.02),
    "LargeBox_1": (2.82, 1.94, 0.02),
    "XLargeBox_0": (3.28, 1.60, 0.02),
    "XLargeBox_1": (3.28, 1.94, 0.02),
}

MAX_INSTANCES_PER_TYPE = 2
MAX_BOXES_PER_SHELF = 4
# Measured directly in the authored Rack.usd root Xform. Local X runs across
# the shelf, local Y runs along its depth, and local Z points upward.
RACK_SHELF_CENTER_LOCAL_X_RAW = -0.50
RACK_SHELF_USABLE_WIDTH_RAW = 0.92
RACK_FRONT_ROW_DEPTH_RAW = 0.16
RACK_BACK_ROW_DEPTH_RAW = 0.52
RACK_RAMP_BACK_DEPTH_RAW = 0.85102
RACK_ROW_GAP_M = 0.04
RACK_SURFACE_CLEARANCE_M = 0.008


@dataclass(frozen=True)
class BoxSpawnSpec:
    """Pose/scale for one of the eight fixed box articulation instances."""

    instance_name: str
    scene_key: str
    box_type: str
    position: Vec3
    rotation: Quat
    scale: Vec3
    shelf: int | None
    row: int | None
    column: int | None

    @property
    def on_rack(self) -> bool:
        return self.shelf is not None


@dataclass(frozen=True)
class CapturedBoxPose:
    """One exact box pose expressed relative to the workcell Rack anchor."""

    instance_name: str
    local_pos: Vec3
    local_rot: Quat
    scale: Vec3
    shelf: int


def _canonical_box_type(value: object) -> str:
    token = str(value).strip().lower().replace("_", "").replace(" ", "")
    try:
        return BOX_TYPE_ALIASES[token]
    except KeyError as exc:
        allowed = ", ".join(BOX_TYPE_LABELS)
        raise ValueError(f"Unknown box type '{value}'. Use one of: {allowed}.") from exc


def _shelf_number(value: object) -> int:
    try:
        shelf = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Shelf '{value}' must be 1, 2, or 3.") from exc
    if shelf not in (1, 2, 3):
        raise ValueError(f"Shelf '{value}' must be 1, 2, or 3.")
    return shelf


def normalize_rack_box_layout(raw: Mapping[object, object]) -> RackBoxLayout:
    """Normalize list/count-map dictionary forms and validate capacity."""
    if "shelves" in raw:
        nested = raw["shelves"]
        if not isinstance(nested, Mapping):
            raise ValueError("The JSON 'shelves' value must be an object.")
        raw = nested

    result: RackBoxLayout = {1: [], 2: [], 3: []}
    seen: set[int] = set()
    for shelf_value, contents in raw.items():
        shelf = _shelf_number(shelf_value)
        if shelf in seen:
            raise ValueError(f"Shelf {shelf} is specified more than once.")
        seen.add(shelf)

        expanded: list[str] = []
        if isinstance(contents, Mapping):
            for box_type, count_value in contents.items():
                if isinstance(count_value, bool) or not isinstance(count_value, int):
                    raise ValueError(
                        f"Shelf {shelf} count for '{box_type}' must be an integer."
                    )
                if count_value < 0:
                    raise ValueError(
                        f"Shelf {shelf} count for '{box_type}' cannot be negative."
                    )
                expanded.extend([_canonical_box_type(box_type)] * count_value)
        elif isinstance(contents, Sequence) and not isinstance(contents, (str, bytes)):
            expanded = [_canonical_box_type(box_type) for box_type in contents]
        else:
            raise ValueError(
                f"Shelf {shelf} must contain a box list or a box-type/count object."
            )

        if len(expanded) > MAX_BOXES_PER_SHELF:
            raise ValueError(
                f"Shelf {shelf} requests {len(expanded)} boxes; the automatic layout supports "
                f"at most {MAX_BOXES_PER_SHELF}."
            )
        result[shelf] = expanded

    totals = Counter(box_type for boxes in result.values() for box_type in boxes)
    for box_type, count in totals.items():
        if count > MAX_INSTANCES_PER_TYPE:
            raise ValueError(
                f"Layout requests {count} '{box_type}' boxes, but only "
                f"{MAX_INSTANCES_PER_TYPE} instances are spawned."
            )
    return result


def parse_compact_rack_box_layout(value: str) -> RackBoxLayout:
    """Parse ``1:small*2,medium;2:large;3:xlarge`` CLI syntax."""
    value = value.strip()
    if not value or value.lower() in {"none", "empty"}:
        return {1: [], 2: [], 3: []}

    raw: dict[int, list[str]] = {}
    for shelf_clause in value.split(";"):
        clause = shelf_clause.strip()
        if not clause:
            continue
        separator = ":" if ":" in clause else "=" if "=" in clause else None
        if separator is None:
            raise ValueError(
                f"Invalid shelf clause '{clause}'. Expected SHELF:box,box syntax."
            )
        shelf_text, box_text = clause.split(separator, 1)
        shelf = _shelf_number(shelf_text.strip())
        if shelf in raw:
            raise ValueError(f"Shelf {shelf} is specified more than once.")

        boxes: list[str] = []
        for item in box_text.split(","):
            token = item.strip()
            if not token:
                continue
            if "*" in token:
                box_text_value, count_text = token.rsplit("*", 1)
                try:
                    count = int(count_text)
                except ValueError as exc:
                    raise ValueError(f"Invalid count in '{token}'. Use box*COUNT.") from exc
                if count < 0:
                    raise ValueError(f"Count in '{token}' cannot be negative.")
            else:
                box_text_value, count = token, 1
            boxes.extend([_canonical_box_type(box_text_value)] * count)
        raw[shelf] = boxes
    return normalize_rack_box_layout(raw)


def load_rack_box_layout(path: str | Path) -> RackBoxLayout:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping):
        raise ValueError(f"Rack-box layout file must contain a JSON object: {source}")
    return normalize_rack_box_layout(raw)


def resolve_rack_box_layout(
    compact: str | None = None,
    json_path: str | Path | None = None,
) -> RackBoxLayout:
    """Resolve CLI, environment, or code-default layout in that order."""
    if compact is not None and json_path is not None:
        raise ValueError("Use only one of --rack-boxes and --rack-box-layout.")
    if compact is not None:
        return parse_compact_rack_box_layout(compact)
    if json_path is not None:
        return load_rack_box_layout(json_path)

    env_compact = os.environ.get("KUAVO_RACK_BOXES")
    env_path = os.environ.get("KUAVO_RACK_BOX_LAYOUT")
    if env_compact is not None and env_path is not None:
        raise ValueError(
            "Set only one of KUAVO_RACK_BOXES and KUAVO_RACK_BOX_LAYOUT."
        )
    if env_compact is not None:
        return parse_compact_rack_box_layout(env_compact)
    if env_path is not None:
        return load_rack_box_layout(env_path)
    return normalize_rack_box_layout(DEFAULT_RACK_BOX_LAYOUT)


def rack_box_count(layout: RackBoxLayout) -> int:
    return sum(len(boxes) for boxes in layout.values())


def resolve_rack_box_pose_path(
    explicit_path: str | Path | None = None,
    *,
    ignore: bool = False,
) -> Path | None:
    """Resolve CLI/environment/default captured-pose file, if enabled."""
    if ignore or os.environ.get("KUAVO_IGNORE_RACK_BOX_POSES", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Captured rack-box pose file not found: {path}")
        return path
    env_path = os.environ.get("KUAVO_RACK_BOX_POSES")
    if env_path:
        path = Path(env_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Captured rack-box pose file not found: {path}")
        return path
    return DEFAULT_RACK_BOX_POSE_PATH if DEFAULT_RACK_BOX_POSE_PATH.is_file() else None


def _vec(value: object, length: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != length:
        raise ValueError(f"{label} must be a list with {length} numbers.")
    return tuple(float(component) for component in value)


def load_captured_box_poses(path: str | Path) -> dict[str, CapturedBoxPose]:
    """Load exact Rack-anchor-relative box poses written by the capture tool."""
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    if not isinstance(raw, Mapping) or not isinstance(raw.get("boxes"), Mapping):
        raise ValueError(f"Captured box pose file must contain a 'boxes' object: {source}")

    valid_names = {
        f"{label}_{index}"
        for label in BOX_TYPE_LABELS.values()
        for index in range(MAX_INSTANCES_PER_TYPE)
    }
    poses: dict[str, CapturedBoxPose] = {}
    for instance_name, value in raw["boxes"].items():
        if instance_name not in valid_names:
            raise ValueError(f"Unknown captured box instance '{instance_name}' in {source}.")
        if not isinstance(value, Mapping):
            raise ValueError(f"Captured box '{instance_name}' must be an object.")
        local_pos = _vec(value.get("local_pos"), 3, f"{instance_name}.local_pos")
        local_rot_raw = _vec(value.get("local_rot"), 4, f"{instance_name}.local_rot")
        norm = math.sqrt(sum(component * component for component in local_rot_raw))
        if norm < 1.0e-8:
            raise ValueError(f"{instance_name}.local_rot cannot be a zero quaternion.")
        local_rot = tuple(component / norm for component in local_rot_raw)
        box_scale = _vec(value.get("scale"), 3, f"{instance_name}.scale")
        if any(component <= 0.0 for component in box_scale):
            raise ValueError(f"{instance_name}.scale values must be positive.")
        shelf = _shelf_number(value.get("shelf"))
        poses[instance_name] = CapturedBoxPose(
            instance_name=instance_name,
            local_pos=local_pos,  # type: ignore[arg-type]
            local_rot=local_rot,  # type: ignore[arg-type]
            scale=box_scale,  # type: ignore[arg-type]
            shelf=shelf,
        )
    if not poses:
        raise ValueError(f"Captured box pose file contains no boxes: {source}")
    return poses


def format_rack_box_layout(layout: RackBoxLayout) -> str:
    parts = []
    for shelf in (1, 2, 3):
        labels = [BOX_TYPE_LABELS[box_type] for box_type in layout[shelf]]
        parts.append(f"shelf {shelf}=[{', '.join(labels)}]")
    return "; ".join(parts)


def _box_scale(box_type: str) -> Vec3:
    dimensions = BOX_DIMENSIONS_M[box_type]
    return tuple(
        dimensions[axis] / BOX_NATIVE_SIZE_M[axis] for axis in range(3)
    )  # type: ignore[return-value]


def _row_lateral_offsets(box_types: Sequence[str]) -> list[float]:
    """Return Rack.usd-local X offsets for one or two centered boxes."""
    rack_width_scale = layout_scale("rack")[0]
    center = RACK_SHELF_CENTER_LOCAL_X_RAW * rack_width_scale
    usable_width = RACK_SHELF_USABLE_WIDTH_RAW * rack_width_scale
    widths = [BOX_DIMENSIONS_M[box_type][0] for box_type in box_types]
    occupied = sum(widths) + RACK_ROW_GAP_M * max(0, len(widths) - 1)
    if occupied > usable_width + 1.0e-6:
        raise ValueError(
            f"A shelf row needs {occupied:.3f} m but only {usable_width:.3f} m is usable. "
            "Reduce box sizes in BOX_DIMENSIONS_M or change the shelf contents."
        )
    cursor = center - occupied / 2.0
    offsets: list[float] = []
    for width in widths:
        offsets.append((cursor + width / 2.0) / rack_width_scale)
        cursor += width + RACK_ROW_GAP_M
    return offsets


def rack_shelf_point(
    shelf: int,
    depth_raw: float,
    lateral_raw: float,
    clearance_m: float,
    rack_slope_rad: float,
) -> Vec3:
    """Transform a measured Rack.usd-local shelf point to world coordinates."""
    if shelf not in (1, 2, 3):
        raise ValueError(f"Shelf must be 1, 2, or 3, got {shelf}.")
    rack_scale = layout_scale("rack")
    surface_z_raw = rack_tier_surface_z(shelf - 1) / rack_scale[2] - math.tan(
        rack_slope_rad
    ) * (RACK_RAMP_BACK_DEPTH_RAW - depth_raw)
    return local_point_to_world(
        "rack",
        (
            lateral_raw,
            -depth_raw,
            surface_z_raw + clearance_m / rack_scale[2],
        ),
    )


def build_box_spawn_plan(
    layout: RackBoxLayout,
    rack_slope_rad: float,
    captured_pose_path: str | Path | None = None,
) -> dict[str, BoxSpawnSpec]:
    """Build all eight poses; omitted instances remain at floor staging."""
    instance_counts = Counter[str]()
    plan: dict[str, BoxSpawnSpec] = {}
    local_pitch = (
        math.cos(-rack_slope_rad / 2.0),
        math.sin(-rack_slope_rad / 2.0),
        0.0,
        0.0,
    )
    rack_aligned_rotation = local_quat_to_world("rack", local_pitch)

    for shelf in (1, 2, 3):
        shelf_types = layout[shelf]
        rows = (shelf_types[:2], shelf_types[2:4])
        for row_index, row_types in enumerate(rows):
            if not row_types:
                continue
            lateral_offsets = _row_lateral_offsets(row_types)
            depth_raw = RACK_FRONT_ROW_DEPTH_RAW if row_index == 0 else RACK_BACK_ROW_DEPTH_RAW
            for column, (box_type, lateral_offset) in enumerate(
                zip(row_types, lateral_offsets, strict=True)
            ):
                instance_index = instance_counts[box_type]
                instance_counts[box_type] += 1
                label = BOX_TYPE_LABELS[box_type]
                instance_name = f"{label}_{instance_index}"
                scale = _box_scale(box_type)
                bottom_offset = 0.005 * scale[2]
                position = rack_shelf_point(
                    shelf,
                    depth_raw,
                    lateral_offset,
                    bottom_offset + RACK_SURFACE_CLEARANCE_M,
                    rack_slope_rad,
                )
                plan[instance_name] = BoxSpawnSpec(
                    instance_name=instance_name,
                    scene_key=f"{BOX_TYPE_SCENE_KEYS[box_type]}_{instance_index}",
                    box_type=box_type,
                    position=position,
                    rotation=rack_aligned_rotation,
                    scale=scale,
                    shelf=shelf,
                    row=row_index,
                    column=column,
                )

    for box_type, label in BOX_TYPE_LABELS.items():
        for instance_index in range(MAX_INSTANCES_PER_TYPE):
            instance_name = f"{label}_{instance_index}"
            if instance_name in plan:
                continue
            plan[instance_name] = BoxSpawnSpec(
                instance_name=instance_name,
                scene_key=f"{BOX_TYPE_SCENE_KEYS[box_type]}_{instance_index}",
                box_type=box_type,
                position=STAGING_BOX_POSITIONS[instance_name],
                rotation=(1.0, 0.0, 0.0, 0.0),
                scale=_box_scale(box_type),
                shelf=None,
                row=None,
                column=None,
            )

    if captured_pose_path is not None:
        for instance_name, captured in load_captured_box_poses(captured_pose_path).items():
            existing = plan[instance_name]
            plan[instance_name] = BoxSpawnSpec(
                instance_name=instance_name,
                scene_key=existing.scene_key,
                box_type=existing.box_type,
                position=local_point_to_world("rack", captured.local_pos),
                rotation=local_quat_to_world("rack", captured.local_rot),
                scale=captured.scale,
                shelf=captured.shelf,
                row=None,
                column=None,
            )
    return plan


def rack_instance_names(plan: Mapping[str, BoxSpawnSpec]) -> tuple[str, ...]:
    return tuple(spec.instance_name for spec in plan.values() if spec.on_rack)
