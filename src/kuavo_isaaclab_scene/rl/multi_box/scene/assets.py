"""V2-only workcell and physical box pool assembly."""

from __future__ import annotations

from dataclasses import dataclass

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg

from ..spec import MultiBoxSpec
from .kinematic import PoseControlledKinematicObject
from .spawn import physical_asset_names, physical_asset_types
from ...scenes.workcell import group, spawn_kinematic_rack
from ....core.paths import ASSET_DIR, BOX_ATLAS_ASSETS
from ....envs.scene_physics import build_contact_box_spawn
from ....workcell.box_flap_friction import resolve_flap_friction_settings
from ....workcell.rack_rollers import (
    RACK_ROLLER_TIERS,
    RACK_SHELF_LOCAL_Z_OFFSETS,
    rack_roller_status,
    rack_visual_asset,
    resolve_rack_roller_settings,
)
from ....workcell.workcell_layout import (
    local_point_to_world,
    offset,
    position,
    remap_quat,
    rotation,
    scale,
)


CONVEYOR_PART_NAMES = (
    "conveyor_surface",
    "conveyor_rail_left",
    "conveyor_rail_right",
    "conveyor_leg_0",
    "conveyor_leg_1",
    "conveyor_leg_2",
    "conveyor_leg_3",
)
RACK_ROLLER_ASSET_NAMES = tuple(f"rack_roller_deck_{tier:02d}" for tier in RACK_ROLLER_TIERS)


@dataclass(frozen=True)
class BoxPoolEntry:
    pool_id: int
    asset_name: str
    box_type: str


def box_pool_entries() -> tuple[BoxPoolEntry, ...]:
    names = physical_asset_names()
    return tuple(
        BoxPoolEntry(index, name, box_type)
        for index, (name, box_type) in enumerate(zip(names, physical_asset_types(), strict=True))
    )


def _kinematic_cuboid(path, size, pos, color, *, rot, visible=True, material=None):
    return RigidObjectCfg(
        class_type=PoseControlledKinematicObject,
        prim_path=path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=visible,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004, rest_offset=0.0),
            physics_material=material,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def add_randomizable_workcell(scene, parallel):
    """Add only the physical assets needed by rack-to-conveyor training."""
    roller_settings = resolve_rack_roller_settings()
    scene.ground = AssetBaseCfg(
        prim_path="/World/Ground", collision_group=-1,
        spawn=sim_utils.CuboidCfg(
            size=(parallel.ground_extent, parallel.ground_extent, 0.10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.11, 0.12, 0.13))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.06)),
    )
    for key, suffix in (
        ("workcell_groups", ""),
        ("racks_group", "/Racks"),
        ("conveyor_group", "/ConveyorSystem"),
        ("conveyor_frame_group", "/ConveyorSystem/Frame"),
        ("staging_boxes_group", "/StagingBoxes"),
    ):
        setattr(scene, key, group("{ENV_REGEX_NS}/Workcell" + suffix))

    rack_path = "{ENV_REGEX_NS}/Workcell/Racks/Rack"
    if roller_settings.enabled:
        print(rack_roller_status(roller_settings, include_clearance=True), flush=True)
        scene.rack_assembly = AssetBaseCfg(
            prim_path=rack_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(rack_visual_asset(roller_settings)),
                scale=scale("rack"),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    solver_position_iteration_count=32,
                    solver_velocity_iteration_count=8,
                ),
            ),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=position("rack"), rot=rotation("rack")),
        )
        # Register the already spawned physics roots separately. Reset moves
        # RackBody and all three RollerDeck articulations with one rigid
        # transform, preserving the randomized rack as a coherent assembly.
        scene.rack = RigidObjectCfg(
            class_type=PoseControlledKinematicObject,
            prim_path=rack_path + "/RackBody",
            spawn=None,
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position("rack"), rot=rotation("rack")),
        )
        for tier, name in zip(RACK_ROLLER_TIERS, RACK_ROLLER_ASSET_NAMES, strict=True):
            setattr(scene, name, ArticulationCfg(
                prim_path=rack_path + f"/RollerDeck_{tier:02d}",
                spawn=None,
                init_state=ArticulationCfg.InitialStateCfg(
                    pos=local_point_to_world(
                        "rack", (0.0, 0.0, RACK_SHELF_LOCAL_Z_OFFSETS[tier])),
                    rot=rotation("rack"),
                    joint_vel={".*": 0.0},
                ),
                actuators={},
            ))
    else:
        scene.rack = RigidObjectCfg(
            class_type=PoseControlledKinematicObject,
            prim_path=rack_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(ASSET_DIR / "Rack.usd"), scale=scale("rack"),
                func=spawn_kinematic_rack,
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=position("rack"), rot=rotation("rack")),
        )

    conveyor_rot = remap_quat("conveyor", (1.0, 0.0, 0.0, 0.0))
    belt_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=0.9, dynamic_friction=0.8, restitution=0.0)
    scene.conveyor_surface = _kinematic_cuboid(
        "{ENV_REGEX_NS}/Workcell/ConveyorSystem/Surface", (2.55, 0.68, 0.03),
        offset("conveyor", (1.29, 0.43, 0.753)), (0.10, 0.12, 0.14),
        rot=conveyor_rot, material=belt_material)
    scene.conveyor_rail_left = _kinematic_cuboid(
        "{ENV_REGEX_NS}/Workcell/ConveyorSystem/Frame/RailLeft", (2.55, 0.035, 0.10),
        offset("conveyor", (1.29, 0.0725, 0.685)), (0.35, 0.38, 0.40), rot=conveyor_rot)
    scene.conveyor_rail_right = _kinematic_cuboid(
        "{ENV_REGEX_NS}/Workcell/ConveyorSystem/Frame/RailRight", (2.55, 0.035, 0.10),
        offset("conveyor", (1.29, 0.7875, 0.685)), (0.35, 0.38, 0.40), rot=conveyor_rot)
    for index, (x, y) in enumerate(((0.24, 0.14), (0.24, 0.72), (2.34, 0.14), (2.34, 0.72))):
        setattr(scene, f"conveyor_leg_{index}", _kinematic_cuboid(
            f"{{ENV_REGEX_NS}}/Workcell/ConveyorSystem/Frame/Leg{index}",
            (0.065, 0.065, 0.64), offset("conveyor", (x, y, 0.31)),
            (0.35, 0.38, 0.40), rot=conveyor_rot))


def _parking_position(index):
    # Outside the 1.5 m robot workspace, on the ground and separated from peers.
    return (2.25 + 0.42 * (index // 6), -1.55 + 0.42 * (index % 6), 0.02)


def add_box_pool(scene, spec: MultiBoxSpec) -> tuple[BoxPoolEntry, ...]:
    """Spawn six S2-small, six S2-medium and six S3-small candidates."""
    spec.validate()
    friction = resolve_flap_friction_settings(randomize_default=False)
    actuator = ImplicitActuatorCfg(
        joint_names_expr=["joint_front", "joint_back", "joint_left", "joint_right"],
        effort_limit_sim=5.0, velocity_limit_sim=10.0,
        stiffness=0.0, damping=0.05,
        friction=friction.static, dynamic_friction=friction.dynamic,
    )
    entries = box_pool_entries()
    for entry in entries:
        cfg = ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Workcell/StagingBoxes/{entry.asset_name}",
            spawn=build_contact_box_spawn(BOX_ATLAS_ASSETS[entry.box_type], (1.0, 1.0, 1.0)),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=_parking_position(entry.pool_id),
                rot=(1.0, 0.0, 0.0, 0.0),
                joint_pos={"joint_(front|back|left|right)": 0.0},
                joint_vel={".*": 0.0},
            ),
            actuators={"flaps": actuator},
        )
        cfg.spawn.activate_contact_sensors = True
        setattr(scene, entry.asset_name, cfg)
    return entries
