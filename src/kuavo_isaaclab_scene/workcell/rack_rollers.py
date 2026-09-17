"""Optional, physically modeled gravity-roller deck for the rack shelves.

Real gravity-flow racks use rows of small free-spinning rollers: motion
along the ramp (down-slope) rolls with almost no resistance because the
rollers turn, while motion across the rollers' own spin axis still has to
slide against ordinary contact friction. PhysX materials are isotropic
(one friction value per contact), so a single friction coefficient cannot
reproduce both behaviors at once, and a scripted "read the box velocity and
push back" force is not something a real robot's controller would ever
encounter on the physical rack. This module instead authors real roller
geometry (small cylinders on free revolute joints) so the resulting contact
dynamics stay close to what a robot will experience on the physical rack,
which matters when a policy trained here must also run on real hardware.

This is opt-in and additive: the default scene keeps using the plain
Rack.usd shelf surface exactly as before. Passing --rack-rollers (or
setting KUAVO_RACK_ROLLERS=1) spawns three extra per-tier articulations,
many small free-spinning roller cylinders each, without modifying
Rack.usd or any existing box/flap/gripper physics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile

from ..core.paths import ASSET_DIR, RACK_ROLLER_ASSET, RACK_ROLLER_RUNTIME_ASSET
from .rack_box_layout import RACK_RAMP_BACK_DEPTH_RAW
from .workcell_layout import (
    RACK_SHELF_CENTER_LOCAL_X_RAW,
    RACK_SHELF_WIDTH_RAW,
    rack_tier_surface_z,
    scale as layout_scale,
)

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]

# Locked down with the user for the resized Rack.usd: 7 rollers across the
# shelf width, 26 rows along its depth, 3 cm diameter, 6 cm roller length,
# 6 cm inter-roller gaps, and 5 cm margins at both ends.  The complete
# footprint exactly matches the resized shelf's measured 0.88 m width:
#   0.05*2 + 0.06*7 + 0.06*6 == 0.88
DEFAULT_ROLLER_DIAMETER_M = 0.03
DEFAULT_ROLLER_LENGTH_M = 0.06
DEFAULT_ROLLER_GAP_M = 0.06
DEFAULT_ROLLER_END_MARGIN_M = 0.05
DEFAULT_ROLLER_ROWS = 26
DEFAULT_ROLLER_COLUMNS = 7
DEFAULT_ROLLER_DEPTH_MARGIN_M = 0.02
DEFAULT_ROLLER_MASS_KG = 0.05
# Measured directly in Rack.usd: each shelf_0N Xform's own local Z offset
# from the /Root/Rack anchor (pure translate, no rotation/scale). Nesting
# each tier's rollers under that same shelf_0N prim means their own
# xformOp:translate must be given relative to this offset, not the bare
# Rack root.
RACK_SHELF_NAMES: dict[int, str] = {1: "shelf_01", 2: "shelf_02", 3: "shelf_03"}
RACK_ROLLER_TIERS = tuple(RACK_SHELF_NAMES)
RACK_SHELF_LOCAL_Z_OFFSETS: dict[int, float] = {1: 0.395, 2: 1.0, 3: 1.61}
# The source shelf_ramp cube is 5 cm thick. Recessing 2 cm out of its
# middle before the rollers sit down means boxes only rise by
# diameter_m - recess_m above the original surface instead of the full
# roller diameter. The roller asset thins and lowers the ramp collision so
# the recessed rollers remain physically clear while the shelf can still
# catch boxes between rollers. See generate_roller_deck_usda's docstring.
DEFAULT_ROLLER_RECESS_M = 0.02
# Light passive bearing damping only; stiffness stays 0 so the roller is a
# free joint, not a servo. This is authored solely on the joint's angular
# drive (see _tier_fragment), not on the roller's own PhysxRigidBodyAPI,
# so it is not double-counted.
#
# Sized from the roller's own moment of inertia rather than picked by feel:
# a solid cylinder I = 0.5*m*r^2 with the default 0.05 kg / 3.0 cm roller
# is ~5.6e-6 kg*m^2, so a damping torque tau = -b*omega decays free spin
# with time constant I/b. The previous 0.0008 gave I/b =~ 3-6 ms (it was
# also duplicated on PhysxRigidBodyAPI, which applies its own separate
# damping torque on top of the joint drive's) -- a spun-up roller with no
# box on it would visibly stop within a couple of physics steps, which is
# nothing like a real low-friction bearing and made the box's own rolling
# motion do more of the work fighting residual drag than it should. This
# value instead targets I/b =~ 0.25 s, a light bearing that coasts down
# gradually instead of nearly instantly, while still eventually settling
# (never a literal zero) so idle rollers do not spin forever.
DEFAULT_ROLLER_ANGULAR_DAMPING = 0.00002
# Rolling only has low resistance because static friction is high enough to
# spin the roller instead of letting the box skid across it (classic
# rolling-without-slipping): energy goes into roller rotation, not into
# resisting the box, so the box still glides down the ramp easily. Across
# the roller's own axis there is no rotation to absorb the motion, so the
# same coefficient shows up as ordinary sliding friction and blocks lateral
# drift. A low, near-zero coefficient (like the bare shelf uses) would
# instead let the box skid in every direction and defeat the whole point of
# adding rollers, so this must stay moderate/high, matching the other
# authoritative structural surfaces already used in this codebase (the
# conveyor deck and workcell floor both use 0.9/0.8).
DEFAULT_ROLLER_STATIC_FRICTION = 0.9
DEFAULT_ROLLER_DYNAMIC_FRICTION = 0.8


@dataclass(frozen=True)
class RackRollerSettings:
    enabled: bool
    diameter_m: float
    length_m: float
    gap_m: float
    end_margin_m: float
    rows: int
    columns: int
    depth_margin_m: float
    recess_m: float = DEFAULT_ROLLER_RECESS_M
    mass_kg: float = DEFAULT_ROLLER_MASS_KG
    angular_damping: float = DEFAULT_ROLLER_ANGULAR_DAMPING
    static_friction: float = DEFAULT_ROLLER_STATIC_FRICTION
    dynamic_friction: float = DEFAULT_ROLLER_DYNAMIC_FRICTION

    @property
    def usable_width_m(self) -> float:
        return (
            self.end_margin_m * 2.0
            + self.length_m * self.columns
            + self.gap_m * (self.columns - 1)
        )

    @property
    def box_clearance_m(self) -> float:
        """How much higher boxes rest above the original shelf surface."""
        return max(0.0, self.diameter_m - self.recess_m)


def rack_visual_asset(settings: RackRollerSettings) -> Path:
    """Select the plain or roller rack and validate roller dependencies once."""
    if not settings.enabled:
        return ASSET_DIR / "Rack.usd"
    missing = [path for path in (RACK_ROLLER_ASSET, RACK_ROLLER_RUNTIME_ASSET) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing rack roller asset(s): {', '.join(map(str, missing))}. "
            "Build rack_roller.usda with workcell/rack_rollers.write_roller_deck_usda; "
            "rack_roller_runtime.usda is its committed GPU-safe composition."
        )
    return RACK_ROLLER_RUNTIME_ASSET


def rack_roller_status(settings: RackRollerSettings, *, include_clearance: bool = False) -> str:
    """Format the shared startup message for a roller-enabled rack."""
    message = (
        f"Rack rollers enabled: {settings.rows}x{settings.columns} free-spinning cylinders per tier, "
        f"diameter {settings.diameter_m * 100:.1f} cm; "
        f"recessed {settings.recess_m * 100:.1f} cm into the shelf surface"
    )
    if include_clearance:
        message += f"; boxes raised by {settings.box_clearance_m * 100:.1f} cm to rest on top"
    return f"[INFO] {message}."


def rack_contact_body_paths(prim_path: str, usd_path: str | Path | None) -> list[str]:
    """Return exact rigid-body paths accepted by PhysX GPU contact filters."""
    if usd_path is None or Path(usd_path).name != RACK_ROLLER_RUNTIME_ASSET.name:
        return [prim_path]
    paths = [f"{prim_path}/RackBody"]
    paths.extend(
        f"{prim_path}/RollerDeck_0{tier}/Roller_r{row:02d}_c{column:02d}"
        for tier in RACK_ROLLER_TIERS
        for row in range(DEFAULT_ROLLER_ROWS)
        for column in range(DEFAULT_ROLLER_COLUMNS)
    )
    return paths


def _env_bool(name: str, fallback: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return fallback
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false or 1/0, got '{value}'.")


def _env_float(name: str, fallback: float) -> float:
    value = os.environ.get(name)
    return fallback if value is None else float(value)


def _env_int(name: str, fallback: int) -> int:
    value = os.environ.get(name)
    return fallback if value is None else int(value)


def add_rack_roller_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rack-rollers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Spawn a physically modeled gravity-roller deck (small free-spinning "
            "cylinders) under rack boxes instead of the bare Rack.usd shelf surface. "
            "Off by default (or set KUAVO_RACK_ROLLERS=1); existing scenes/"
            "checkpoints are unchanged."
        ),
    )


def export_rack_roller_cli(args: argparse.Namespace) -> None:
    value = getattr(args, "rack_rollers", None)
    if value is None:
        os.environ.pop("KUAVO_RACK_ROLLERS", None)
    else:
        os.environ["KUAVO_RACK_ROLLERS"] = "1" if value else "0"


def resolve_rack_roller_settings(
    *,
    enabled: bool | None = None,
    diameter_m: float | None = None,
    length_m: float | None = None,
    gap_m: float | None = None,
    end_margin_m: float | None = None,
    rows: int | None = None,
    columns: int | None = None,
    depth_margin_m: float | None = None,
    recess_m: float | None = None,
    static_friction: float | None = None,
    dynamic_friction: float | None = None,
) -> RackRollerSettings:
    """Resolve CLI, environment, and code defaults, then validate them."""
    settings = RackRollerSettings(
        enabled=(_env_bool("KUAVO_RACK_ROLLERS", False) if enabled is None else bool(enabled)),
        diameter_m=(
            _env_float("KUAVO_RACK_ROLLER_DIAMETER_M", DEFAULT_ROLLER_DIAMETER_M)
            if diameter_m is None else float(diameter_m)
        ),
        length_m=(
            _env_float("KUAVO_RACK_ROLLER_LENGTH_M", DEFAULT_ROLLER_LENGTH_M)
            if length_m is None else float(length_m)
        ),
        gap_m=(
            _env_float("KUAVO_RACK_ROLLER_GAP_M", DEFAULT_ROLLER_GAP_M)
            if gap_m is None else float(gap_m)
        ),
        end_margin_m=(
            _env_float("KUAVO_RACK_ROLLER_END_MARGIN_M", DEFAULT_ROLLER_END_MARGIN_M)
            if end_margin_m is None else float(end_margin_m)
        ),
        rows=(
            _env_int("KUAVO_RACK_ROLLER_ROWS", DEFAULT_ROLLER_ROWS)
            if rows is None else int(rows)
        ),
        columns=(
            _env_int("KUAVO_RACK_ROLLER_COLUMNS", DEFAULT_ROLLER_COLUMNS)
            if columns is None else int(columns)
        ),
        depth_margin_m=(
            _env_float("KUAVO_RACK_ROLLER_DEPTH_MARGIN_M", DEFAULT_ROLLER_DEPTH_MARGIN_M)
            if depth_margin_m is None else float(depth_margin_m)
        ),
        recess_m=(
            _env_float("KUAVO_RACK_ROLLER_RECESS_M", DEFAULT_ROLLER_RECESS_M)
            if recess_m is None else float(recess_m)
        ),
        static_friction=(
            _env_float("KUAVO_RACK_ROLLER_STATIC_FRICTION", DEFAULT_ROLLER_STATIC_FRICTION)
            if static_friction is None else float(static_friction)
        ),
        dynamic_friction=(
            _env_float("KUAVO_RACK_ROLLER_DYNAMIC_FRICTION", DEFAULT_ROLLER_DYNAMIC_FRICTION)
            if dynamic_friction is None else float(dynamic_friction)
        ),
    )
    if settings.diameter_m <= 0.0 or settings.length_m <= 0.0:
        raise ValueError("Roller diameter/length must be positive.")
    if settings.gap_m < 0.0 or settings.end_margin_m < 0.0 or settings.depth_margin_m < 0.0:
        raise ValueError("Roller gap/margins cannot be negative.")
    if settings.recess_m < 0.0:
        raise ValueError("Roller recess_m cannot be negative.")
    if settings.recess_m > settings.diameter_m:
        raise ValueError(
            "Roller recess_m cannot exceed diameter_m (the roller would sit below "
            "the original shelf surface)."
        )
    if settings.rows < 2 or settings.columns < 1:
        raise ValueError("Roller grid needs at least 2 rows and 1 column.")
    if settings.static_friction < 0.0 or settings.dynamic_friction < 0.0:
        raise ValueError("Roller friction values cannot be negative.")
    if settings.dynamic_friction > settings.static_friction:
        raise ValueError("Roller dynamic_friction cannot exceed static_friction.")
    rack_width_scale = layout_scale("rack")[0]
    usable_shelf_width = RACK_SHELF_WIDTH_RAW * rack_width_scale
    if settings.usable_width_m > usable_shelf_width + 1.0e-4:
        raise ValueError(
            f"Roller grid needs {settings.usable_width_m:.3f} m but the shelf's usable "
            f"width is {usable_shelf_width:.3f} m. Reduce columns/length/gap."
        )
    return settings


def export_rack_roller_environment(settings: RackRollerSettings) -> None:
    """Pass launcher settings through the delayed manager_env import."""
    os.environ["KUAVO_RACK_ROLLERS"] = "1" if settings.enabled else "0"
    os.environ["KUAVO_RACK_ROLLER_DIAMETER_M"] = str(settings.diameter_m)
    os.environ["KUAVO_RACK_ROLLER_LENGTH_M"] = str(settings.length_m)
    os.environ["KUAVO_RACK_ROLLER_GAP_M"] = str(settings.gap_m)
    os.environ["KUAVO_RACK_ROLLER_END_MARGIN_M"] = str(settings.end_margin_m)
    os.environ["KUAVO_RACK_ROLLER_ROWS"] = str(settings.rows)
    os.environ["KUAVO_RACK_ROLLER_COLUMNS"] = str(settings.columns)
    os.environ["KUAVO_RACK_ROLLER_DEPTH_MARGIN_M"] = str(settings.depth_margin_m)
    os.environ["KUAVO_RACK_ROLLER_RECESS_M"] = str(settings.recess_m)
    os.environ["KUAVO_RACK_ROLLER_STATIC_FRICTION"] = str(settings.static_friction)
    os.environ["KUAVO_RACK_ROLLER_DYNAMIC_FRICTION"] = str(settings.dynamic_friction)


def _column_centers_raw(settings: RackRollerSettings, rack_width_scale: float) -> list[float]:
    left_edge_raw = (
        RACK_SHELF_CENTER_LOCAL_X_RAW
        - (settings.usable_width_m / 2.0) / rack_width_scale
    )
    pitch_raw = (settings.length_m + settings.gap_m) / rack_width_scale
    first_center_raw = left_edge_raw + (settings.end_margin_m + settings.length_m / 2.0) / rack_width_scale
    return [first_center_raw + column * pitch_raw for column in range(settings.columns)]


def _row_depths_raw(settings: RackRollerSettings) -> list[float]:
    depth_start = settings.depth_margin_m
    depth_end = RACK_RAMP_BACK_DEPTH_RAW - settings.depth_margin_m
    if depth_end <= depth_start:
        raise ValueError("Roller depth margin leaves no usable ramp length.")
    if settings.rows == 1:
        return [(depth_start + depth_end) / 2.0]
    step = (depth_end - depth_start) / (settings.rows - 1)
    return [depth_start + row * step for row in range(settings.rows)]


def roller_local_pitch_quat(rack_slope_rad: float) -> Quat:
    """Same small ramp-tilt pitch every box uses, so rollers sit flush too."""
    return (
        math.cos(-rack_slope_rad / 2.0),
        math.sin(-rack_slope_rad / 2.0),
        0.0,
        0.0,
    )


def _usda_float(value: float) -> str:
    return f"{value:.6f}"


def _tier_fragment(
    tier: int,
    settings: RackRollerSettings,
    rack_slope_rad: float,
    materials_root: str,
    tier_root: str,
) -> str:
    """Build one shelf tier's Xform: its own ArticulationRootAPI plus rollers/joints."""
    # Each roller's joint explicitly targets a per-tier fixed "Base" link
    # (mirroring button_station.usda's proven Base/FixedJoint pattern)
    # instead of leaving physics:body0 unset. An unset body0 is documented
    # as "attach to world", but its localPos0 frame is not guaranteed to
    # follow this asset's own outer placement transform the way a prim's
    # own xformOp does; an explicit, zero-offset Base prim removes that
    # ambiguity so every roller's fixed anchor reliably tracks wherever
    # this whole file gets placed, and lets it actually spin freely.
    rack_scale = layout_scale("rack")
    tier_name = RACK_SHELF_NAMES[tier]
    radius_m = settings.diameter_m / 2.0
    column_centers_raw = _column_centers_raw(settings, rack_scale[0])
    row_depths_raw = _row_depths_raw(settings)
    surface_z_raw_at_back = rack_tier_surface_z(tier - 1) / rack_scale[2]
    shelf_local_z_offset_raw = RACK_SHELF_LOCAL_Z_OFFSETS[tier] / rack_scale[2]
    pitch_quat = roller_local_pitch_quat(rack_slope_rad)

    bodies: list[str] = []
    joints: list[str] = []
    for row, depth_raw in enumerate(row_depths_raw):
        local_z_raw = (
            surface_z_raw_at_back
            - math.tan(rack_slope_rad) * (RACK_RAMP_BACK_DEPTH_RAW - depth_raw)
            - settings.recess_m / rack_scale[2]
            + radius_m / rack_scale[2]
            - shelf_local_z_offset_raw
        )
        for column, lateral_raw in enumerate(column_centers_raw):
            name = f"Roller_r{row:02d}_c{column:02d}"
            pos = (lateral_raw, -depth_raw, local_z_raw)
            bodies.append(
                f"""
        def Xform "{name}" (
            prepend apiSchemas = ["PhysicsMassAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
        )
        {{
            float physics:mass = {_usda_float(settings.mass_kg)}
            # Explicit 0.0, not omitted: PhysxRigidBodyAPI's own schema
            # default for angularDamping is nonzero, so leaving this
            # unauthored would silently reintroduce body-level drag on top
            # of the joint drive's damping below. All bearing resistance
            # is intentionally authored in exactly one place (the joint).
            float physxRigidBody:angularDamping = 0.0
            double3 xformOp:translate = ({_usda_float(pos[0])}, {_usda_float(pos[1])}, {_usda_float(pos[2])})
            quatf xformOp:orient = ({_usda_float(pitch_quat[0])}, {_usda_float(pitch_quat[1])}, {_usda_float(pitch_quat[2])}, {_usda_float(pitch_quat[3])})
            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]

            def Cylinder "Geom" (
                prepend apiSchemas = ["MaterialBindingAPI", "PhysicsCollisionAPI", "PhysxCollisionAPI"]
            )
            {{
                rel material:binding = <{materials_root}/RollerMetal>
                rel material:binding:physics = <{materials_root}/RollerContactMaterial>
                uniform token axis = "X"
                double height = {_usda_float(settings.length_m)}
                double radius = {_usda_float(radius_m)}
            }}
        }}"""
            )
            joints.append(
                f"""
        def PhysicsRevoluteJoint "{name}_Joint" (
            prepend apiSchemas = ["PhysicsDriveAPI:angular"]
        )
        {{
            uniform token physics:axis = "X"
            rel physics:body0 = <{tier_root}/Base>
            rel physics:body1 = <{tier_root}/{name}>
            point3f physics:localPos0 = ({_usda_float(pos[0])}, {_usda_float(pos[1])}, {_usda_float(pos[2])})
            point3f physics:localPos1 = (0, 0, 0)
            quatf physics:localRot0 = ({_usda_float(pitch_quat[0])}, {_usda_float(pitch_quat[1])}, {_usda_float(pitch_quat[2])}, {_usda_float(pitch_quat[3])})
            quatf physics:localRot1 = (1, 0, 0, 0)
            uniform token physics:drive:angular:type = "force"
            float physics:drive:angular:stiffness = 0
            float physics:drive:angular:damping = {_usda_float(settings.angular_damping)}
            float physics:drive:angular:maxForce = 0.05
            float physics:drive:angular:targetVelocity = 0
        }}"""
            )

    body_text = "".join(bodies)
    joint_text = "".join(joints)
    return f"""
        def Xform "{tier_name}"
        {{
            over "shelf_ramp"
            {{
                double3 xformOp:scale = ({_usda_float(RACK_SHELF_WIDTH_RAW)}, 0.8799999952316284, 0.02)
                double3 xformOp:translate = ({_usda_float(RACK_SHELF_CENTER_LOCAL_X_RAW)}, -0.41499999999999987, 0.032)
            }}

            def Mesh "shelf_ramp_front" (
                prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI", "PhysxTriangleMeshCollisionAPI", "PhysicsMeshCollisionAPI"]
            )
            {{
                float3[] extent = [(-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)]
                int[] faceVertexCounts = [4, 4, 4, 4, 4, 4]
                int[] faceVertexIndices = [0, 1, 3, 2, 4, 6, 7, 5, 6, 2, 3, 7, 4, 5, 1, 0, 4, 0, 2, 6, 5, 7, 3, 1]
                uniform token physics:approximation = "boundingCube"
                bool physics:collisionEnabled = 1
                point3f[] points = [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5)]
                uniform token subdivisionScheme = "none"
                quatd xformOp:orient = (1, 0, 0, 0)
                double3 xformOp:scale = ({_usda_float(RACK_SHELF_WIDTH_RAW)}, 0.01, 0.03)
                double3 xformOp:translate = ({_usda_float(RACK_SHELF_CENTER_LOCAL_X_RAW)}, 0.015, 0.02)
                uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient", "xformOp:scale"]
            }}

            def Xform "RollerDeck" (
                prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysicsFilteredPairsAPI"]
            )
            {{
                rel physics:filteredPairs = </RackRoller/RackBody>

                def Xform "Base" (
                    prepend apiSchemas = ["PhysicsMassAPI", "PhysicsRigidBodyAPI", "PhysxRigidBodyAPI"]
                )
                {{
                    float physics:mass = 1
                }}

                def PhysicsFixedJoint "BaseFixedJoint"
                {{
                    rel physics:body1 = <{tier_root}/Base>
                }}

{body_text}
{joint_text}
            }}
        }}"""


def generate_roller_deck_usda(settings: RackRollerSettings, rack_slope_rad: float) -> str:
    """Author one self-contained file: three shelf tiers, each an independent
    free-spinning roller deck.

    Positions are expressed in the same raw, pre-scale Rack.usd-local units
    as rack_box_layout, so the caller places this asset at the plain rack
    anchor pose/scale exactly like the visual rack mesh.

    Rollers sit settings.recess_m lower than the bare shelf surface, so a
    box resting on top of them only rises by box_clearance_m (diameter
    minus recess) instead of the full diameter. The roller variant thins
    each shelf_ramp collision from 5 cm to 2 cm and shifts it downward,
    retaining a separate front support. This removes roller/ramp overlap
    geometrically while the shelf still catches boxes between rollers. No
    explicit PhysicsCollisionGroup is needed. Each RollerDeck instead uses
    PhysicsFilteredPairsAPI against RackBody: box contacts with both the
    rollers and shelf stay enabled, while near-contact roller/rack pairs are
    skipped without duplicate membership in Isaac Lab's per-environment CPU
    collision group.

    The editable source keeps each roller deck at
    RackBody/Rack/shelf_0N/RollerDeck, directly beside its shelf geometry.
    Runtime scenes load rack_roller_runtime.usda, which references this file,
    keeps RackBody as one kinematic GPU contact-filter target, and re-exposes
    the three RollerDeck articulations as siblings. This avoids nesting an
    articulation under a rigid body while preserving every authored geometry,
    joint and material edit in this source file.

    Also includes a RackBody child that references Rack.usd directly (same
    directory, relative asset path), so this one file is the complete rack
    (body plus rollers) rather than an add-on that must be layered next to
    the plain Rack.usd. scene.py swaps this file in for Rack.usd wherever
    --rack-rollers is set, instead of spawning both.

    The ramp edits are authored by this generator as well as shipped in
    rack_roller.usda, so regenerating the asset preserves the clearance.
    """
    root_name = "RackRoller"
    materials_root = f"/{root_name}/Looks"
    tiers = "\n".join(
        _tier_fragment(
            tier,
            settings,
            rack_slope_rad,
            materials_root,
            f"/{root_name}/RackBody/Rack/{RACK_SHELF_NAMES[tier]}/RollerDeck",
        )
        for tier in RACK_ROLLER_TIERS
    )
    return f"""#usda 1.0
(
    defaultPrim = "{root_name}"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "{root_name}"
{{
    def Xform "RackBody" (
        prepend references = @./Rack.usd@
    )
    {{
        def Xform "Rack"
        {{
{tiers}
        }}
    }}

    def Scope "Looks"
    {{
        def Material "RollerMetal"
        {{
            token outputs:surface.connect = </{root_name}/Looks/RollerMetal/Shader.outputs:surface>
            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.62, 0.63, 0.65)
                float inputs:metallic = 0.85
                float inputs:roughness = 0.35
                token outputs:surface
            }}
        }}
        def Material "RollerContactMaterial" (
            prepend apiSchemas = ["PhysicsMaterialAPI"]
        )
        {{
            float physics:staticFriction = {_usda_float(settings.static_friction)}
            float physics:dynamicFriction = {_usda_float(settings.dynamic_friction)}
            float physics:restitution = 0
        }}
    }}
}}
"""


def write_roller_deck_usda(settings: RackRollerSettings, rack_slope_rad: float, path: Path) -> Path:
    """(Re)build the merged three-tier roller-deck asset at path.

    The result is fully self-contained: Rack.usd's own geometry (including
    the editable shelf_ramp meshes) is flattened directly into this file
    instead of being kept as a live USD reference. That way, editing this
    file needs no edit-target/composition knowledge, and later changes to
    Rack.usd do not silently change what this file already shipped.

    Flattening needs a working ``from pxr import Usd``. The plain
    env_isaaclab_232 conda Python does not have this on its default
    sys.path; Isaac Sim's own Python does, and so does adding
    isaacsim's bundled ``omni.usd.libs`` pxr package plus its native
    libraries directory to PYTHONPATH/LD_LIBRARY_PATH.
    """
    try:
        from pxr import Usd
    except ImportError as exc:
        raise RuntimeError(
            "Building rack_roller.usda needs a working 'pxr' (USD) import to "
            "flatten Rack.usd's geometry into the output file. Run this from "
            "Isaac Sim's Python, or add isaacsim's extscache omni.usd.libs "
            "pxr package and its native libraries to PYTHONPATH/LD_LIBRARY_PATH."
        ) from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".usda", dir=path.parent, delete=False, encoding="utf-8"
    ) as handle:
        handle.write(generate_roller_deck_usda(settings, rack_slope_rad))
        temp_path = Path(handle.name)
    try:
        stage = Usd.Stage.Open(str(temp_path))
        flattened_layer = stage.Flatten(addSourceFileComment=False)
        flattened_layer.Export(str(path))
    finally:
        temp_path.unlink(missing_ok=True)
    return path
