"""Shared 6-DoF workcell anchors for both Isaac Lab scene variants."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

from .paths import CONFIG_DIR


Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # Isaac convention: (w, x, y, z)


@dataclass(frozen=True)
class AnchorPose:
    pos: Vec3
    rot: Quat
    scale: Vec3 = (1.0, 1.0, 1.0)


DEFAULT_LAYOUT_PATH = CONFIG_DIR / "workcell_layout.json"

# --- Real-world rack calibration -------------------------------------------
# `assets/Rack.usd` is the user-supplied local rack asset (replaces the old
# official Nucleus `RackLongEmpty_A2`).  It is already authored in real
# meters. The captured pose/scale is authored on the workcell Rack anchor;
# its child Visual always remains at an identity local transform.
#
#   native X (raw asset)  -> becomes world Y (rack width)  after the 90 deg
#                            yaw the code applies to align the rack's long
#                            axis with the gravity-feed depth direction.
#   native Y (raw asset)  -> becomes world X (rack depth, the direction Kuavo
#                            reaches into).
#   native Z (raw asset)  -> stays world Z (rack height).
#
# Measured directly from `assets/Rack.usd` (local bbox of `/Root/Rack`):
#   size = (1.051, 0.8809538, 2.165)  ->  (X=width, Y=depth, Z=height)
RACK_RAW_WIDTH = 1.051
RACK_RAW_DEPTH = 0.8809538024143262
RACK_RAW_HEIGHT = 2.165
# Raw (native meters) bottom/top of each physical shelf ramp mesh,
# bottom-to-top, measured in the `/Root/Rack`-local frame.
RACK_RAW_TIER_RANGES: tuple[tuple[float, float], ...] = (
    (0.370878, 0.499122),
    (0.975878, 1.104122),
    (1.585878, 1.714122),
)

# Requested real-world footprint for the single rack: 88 cm deep, 34.7 cm
# wide, 216.5 cm tall. `Rack.usd` is already close to these dimensions
# (88.1 cm deep, 105.1 cm wide, 216.5 cm tall natively), so the default
# scale is identity; width is intentionally left at the asset's native
# footprint rather than force-squeezed to 34.7 cm, since re-scaling only
# one axis of a rigid rack mesh would distort its geometry.
RACK_NOMINAL_DEPTH_M = RACK_RAW_DEPTH
RACK_NOMINAL_WIDTH_M = RACK_RAW_WIDTH
RACK_NOMINAL_HEIGHT_M = RACK_RAW_HEIGHT
RACK_DEFAULT_SCALE: Vec3 = (1.0, 1.0, 1.0)

DEFAULT_ANCHORS: dict[str, AnchorPose] = {
    "robot": AnchorPose((0.0, 0.0, 0.93), (1.0, 0.0, 0.0, 0.0)),
    "rack": AnchorPose(
        (0.96, 0.35, 0.0),
        (0.7071068, 0.0, 0.0, 0.7071068),
        RACK_DEFAULT_SCALE,
    ),
    "conveyor": AnchorPose((0.40, -0.95, 0.0), (0.7071068, 0.0, 0.0, 0.7071068)),
    "fence": AnchorPose((1.08, 0.93, 0.92), (1.0, 0.0, 0.0, 0.0)),
    "button_station": AnchorPose((0.45, 0.91, 0.89), (1.0, 0.0, 0.0, 0.0)),
}
REQUIRED_ANCHORS = tuple(DEFAULT_ANCHORS)


def _layout_path() -> Path:
    override = os.environ.get("KUAVO_WORKCELL_LAYOUT")
    return Path(override).expanduser().resolve() if override else DEFAULT_LAYOUT_PATH


def _normalize_quat(value: list[float], anchor: str) -> Quat:
    if len(value) != 4:
        raise ValueError(f"Layout anchor '{anchor}.rot' must be a [w, x, y, z] list.")
    norm = math.sqrt(sum(float(component) ** 2 for component in value))
    if norm < 1.0e-8:
        raise ValueError(f"Layout anchor '{anchor}.rot' cannot be a zero quaternion.")
    return tuple(float(component) / norm for component in value)  # type: ignore[return-value]


def load_layout() -> dict[str, AnchorPose]:
    path = _layout_path()
    with path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    missing = set(REQUIRED_ANCHORS).difference(raw)
    if missing:
        raise ValueError(f"Missing workcell layout anchors in {path}: {sorted(missing)}")

    layout: dict[str, AnchorPose] = {}
    for name in REQUIRED_ANCHORS:
        value = raw[name]
        # Backwards compatibility with the earlier translation-only layout.
        if isinstance(value, list):
            pos_value = value
            rot_value = list(DEFAULT_ANCHORS[name].rot)
            scale_value = list(DEFAULT_ANCHORS[name].scale)
        elif isinstance(value, dict):
            pos_value = value.get("pos")
            rot_value = value.get("rot")
            scale_value = value.get("scale", list(DEFAULT_ANCHORS[name].scale))
        else:
            raise ValueError(f"Layout anchor '{name}' must be an object with pos/rot.")
        if not isinstance(pos_value, list) or len(pos_value) != 3:
            raise ValueError(f"Layout anchor '{name}.pos' must be an [x, y, z] list.")
        if not isinstance(rot_value, list):
            raise ValueError(f"Layout anchor '{name}.rot' must be a [w, x, y, z] list.")
        if not isinstance(scale_value, list) or len(scale_value) != 3:
            raise ValueError(f"Layout anchor '{name}.scale' must be an [sx, sy, sz] list.")
        layout[name] = AnchorPose(
            tuple(float(component) for component in pos_value),  # type: ignore[arg-type]
            _normalize_quat(rot_value, name),
            tuple(float(component) for component in scale_value),  # type: ignore[arg-type]
        )
    return layout


def quat_conjugate(quat: Quat) -> Quat:
    return (quat[0], -quat[1], -quat[2], -quat[3])


def quat_multiply(left: Quat, right: Quat) -> Quat:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quat_rotate(quat: Quat, vector: Vec3) -> Vec3:
    vector_quat: Quat = (0.0, vector[0], vector[1], vector[2])
    rotated = quat_multiply(quat_multiply(quat, vector_quat), quat_conjugate(quat))
    return (rotated[1], rotated[2], rotated[3])


LAYOUT_PATH = _layout_path()
LAYOUT = load_layout()


def position(anchor: str) -> Vec3:
    return LAYOUT[anchor].pos


def rotation(anchor: str) -> Quat:
    return LAYOUT[anchor].rot


def scale(anchor: str) -> Vec3:
    return LAYOUT[anchor].scale


def local_point_to_world(anchor: str, local_point: Vec3) -> Vec3:
    """Transform an asset-local point by its captured scale, Xform, and pose."""
    anchor_pose = LAYOUT[anchor]
    scaled = tuple(local_point[index] * anchor_pose.scale[index] for index in range(3))
    rotated = quat_rotate(anchor_pose.rot, scaled)  # type: ignore[arg-type]
    return tuple(anchor_pose.pos[index] + rotated[index] for index in range(3))  # type: ignore[return-value]


def local_quat_to_world(anchor: str, local_quat: Quat) -> Quat:
    """Transform an asset-local orientation by the captured root Xform."""
    return quat_multiply(LAYOUT[anchor].rot, local_quat)


def rack_tier_surface_z(tier_index: int) -> float:
    """Top-surface height of one physical shelf, local to the rack anchor."""
    _, top_raw = RACK_RAW_TIER_RANGES[tier_index]
    return top_raw * scale("rack")[2]


def rack_half_extents() -> Vec3:
    """Half the rack world footprint (depth, width, height) at current scale."""
    rack_scale = scale("rack")
    return (
        RACK_RAW_DEPTH * rack_scale[1] / 2.0,
        RACK_RAW_WIDTH * rack_scale[0] / 2.0,
        RACK_RAW_HEIGHT * rack_scale[2] / 2.0,
    )


def rotation_delta(anchor: str) -> Quat:
    """Rotation taking default world vectors into the captured orientation."""
    return quat_multiply(LAYOUT[anchor].rot, quat_conjugate(DEFAULT_ANCHORS[anchor].rot))


def rotate_default_vector(anchor: str, vector: Vec3) -> Vec3:
    return quat_rotate(rotation_delta(anchor), vector)


def offset(anchor: str, default_world_delta: Vec3) -> Vec3:
    """Map a default world-space offset through an anchor's captured rotation."""
    rotated = rotate_default_vector(anchor, default_world_delta)
    origin = LAYOUT[anchor].pos
    return tuple(origin[index] + rotated[index] for index in range(3))  # type: ignore[return-value]


def remap_point(anchor: str, default_world_point: Vec3) -> Vec3:
    default_origin = DEFAULT_ANCHORS[anchor].pos
    delta = tuple(default_world_point[index] - default_origin[index] for index in range(3))
    return offset(anchor, delta)  # type: ignore[arg-type]


def remap_quat(anchor: str, default_world_quat: Quat) -> Quat:
    return quat_multiply(rotation_delta(anchor), default_world_quat)
