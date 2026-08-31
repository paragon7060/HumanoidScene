#!/usr/bin/env python3
"""Selectable Kuavo gravity-rack to conveyor workcell for Isaac Lab.

The scene models one complete industrial task:

* two gravity-fed rack lanes with three open totes each,
* a safety fence and a gated green completion button,
* a stopped conveyor with multi-worker slot reservation,
* optional pre-filled conveyor slots from other workers,
* button-gated conveyor motion after all six rack totes are loaded.

The ``--auto-demo`` mode is an oracle task-system demonstration.  It validates
gravity feed, slot allocation, queue pushing, button gating, and belt startup;
it does not claim to be a Kuavo manipulation controller.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import sys

from .box_flap_friction import resolve_flap_friction_settings
from .gripper_config import (
    add_gripper_cli_args,
    export_gripper_cli,
    resolve_gripper_settings,
)
from .robot_model import add_robot_model_cli_args, export_robot_model_cli
from .rack_box_layout import (
    RACK_BACK_ROW_DEPTH_RAW,
    RACK_FRONT_ROW_DEPTH_RAW,
    RACK_SHELF_CENTER_LOCAL_X_RAW,
    build_box_spawn_plan,
    format_rack_box_layout,
    load_captured_box_poses,
    rack_box_count,
    rack_instance_names,
    rack_shelf_point,
    resolve_rack_box_layout,
    resolve_rack_box_pose_path,
)
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Spawn the Kuavo rack-to-conveyor factory scene.")
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of simulation steps before exit. Use 0 to run until the app is closed.",
)
parser.add_argument(
    "--save-stage",
    type=Path,
    default=None,
    help="Optional path at which to save the composed stage (.usd/.usda/.usdc).",
)
parser.add_argument(
    "--screenshot",
    type=Path,
    default=None,
    help="Optional path at which to save a rendered viewport preview (.png).",
)
parser.add_argument(
    "--auto-demo",
    action="store_true",
    help="Run an oracle task-system demo (box transport is scripted, not robot-controlled).",
)
parser.add_argument(
    "--demo-speed",
    type=float,
    default=1.0,
    help="Time scale for --auto-demo. Use 4 for a short smoke test.",
)
parser.add_argument(
    "--prefill",
    type=int,
    choices=range(0, 4),
    default=0,
    metavar="{0,1,2,3}",
    help="Number of conveyor slots initially occupied by other workers.",
)
parser.add_argument(
    "--verify-button",
    action="store_true",
    help="Run a physical press/release check with a temporary kinematic probe.",
)
rack_layout_group = parser.add_mutually_exclusive_group()
rack_layout_group.add_argument(
    "--rack-boxes",
    type=str,
    default=None,
    metavar="SPEC",
    help=(
        "Rack contents, e.g. '1:small*2,medium;2:large;3:xlarge*2'. "
        "Shelves are numbered bottom-to-top."
    ),
)
rack_layout_group.add_argument(
    "--rack-box-layout",
    type=Path,
    default=None,
    metavar="JSON",
    help="JSON file defining shelf 1/2/3 box lists or type/count maps.",
)
captured_pose_group = parser.add_mutually_exclusive_group()
captured_pose_group.add_argument(
    "--rack-box-poses",
    type=Path,
    default=None,
    metavar="JSON",
    help="Exact Rack-anchor-relative box poses captured from an edited Isaac Sim stage.",
)
captured_pose_group.add_argument(
    "--ignore-captured-box-poses",
    action="store_true",
    help="Ignore the default rack_box_poses.json file for this run.",
)
parser.add_argument(
    "--flap-static-friction",
    type=float,
    default=None,
    metavar="VALUE",
    help="Static friction coefficient for all box flap revolute joints.",
)
parser.add_argument(
    "--flap-dynamic-friction",
    type=float,
    default=None,
    metavar="VALUE",
    help="Dynamic friction coefficient for all box flap revolute joints.",
)
parser.add_argument(
    "--randomize-flap-friction",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Randomize box flap static/dynamic friction once at scene startup.",
)
parser.add_argument(
    "--flap-static-friction-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
)
parser.add_argument(
    "--flap-dynamic-friction-range",
    type=float,
    nargs=2,
    default=None,
    metavar=("MIN", "MAX"),
)
add_robot_model_cli_args(parser)
add_gripper_cli_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
export_robot_model_cli(args_cli)
export_gripper_cli(args_cli)
try:
    RACK_BOX_LAYOUT = resolve_rack_box_layout(args_cli.rack_boxes, args_cli.rack_box_layout)
    CAPTURED_RACK_BOX_POSE_PATH = resolve_rack_box_pose_path(
        args_cli.rack_box_poses,
        ignore=args_cli.ignore_captured_box_poses,
    )
    CAPTURED_RACK_BOX_COUNT = (
        len(load_captured_box_poses(CAPTURED_RACK_BOX_POSE_PATH))
        if CAPTURED_RACK_BOX_POSE_PATH is not None
        else 0
    )
    FLAP_FRICTION = resolve_flap_friction_settings(
        static=args_cli.flap_static_friction,
        dynamic=args_cli.flap_dynamic_friction,
        randomize=args_cli.randomize_flap_friction,
        static_range=args_cli.flap_static_friction_range,
        dynamic_range=args_cli.flap_dynamic_friction_range,
        randomize_default=False,
    )
    GRIPPER_SETTINGS = resolve_gripper_settings()
except (OSError, ValueError) as exc:
    parser.error(str(exc))
CONFIGURED_RACK_BOX_COUNT = rack_box_count(RACK_BOX_LAYOUT)
CUSTOM_RACK_BOXES_ACTIVE = bool(CONFIGURED_RACK_BOX_COUNT or CAPTURED_RACK_BOX_COUNT)
if args_cli.demo_speed <= 0.0:
    parser.error("--demo-speed must be positive.")
if args_cli.auto_demo and args_cli.verify_button:
    parser.error("--auto-demo and --verify-button are separate scripted checks.")
if CUSTOM_RACK_BOXES_ACTIVE and (args_cli.auto_demo or args_cli.verify_button):
    parser.error(
        "--auto-demo/--verify-button use the legacy six-tote oracle. "
        "Run them without --rack-boxes/--rack-box-layout and add "
        "--ignore-captured-box-poses if rack_box_poses.json exists."
    )
if args_cli.verify_button and args_cli.steps == 0:
    args_cli.steps = 300
# Robot-mounted cameras (head, waist, both wrists) are always part of the
# scene config below, so Kit's camera rendering pipeline must always be
# enabled. Without this, the camera sensor prims fail to initialize and
# corrupt articulation physics-view creation on `sim.reset()`.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils import stage as stage_utils
from isaaclab.utils import configclass
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
from pxr import Gf, PhysxSchema

from .task_system import (
    ConveyorSlotManager,
    PlacementMode,
    PlacementPlan,
    RackConveyorTask,
    TaskPhase,
)
from .workcell_layout import (
    LAYOUT_PATH,
    local_quat_to_world,
    offset as layout_offset,
    position as layout_position,
    remap_point,
    remap_quat,
    rotate_default_vector,
    rotation as layout_rotation,
    scale as layout_scale,
)
from .camera_viewports import open_camera_viewports
from .gripper_runtime import (
    build_gripper_articulation_cfg,
    build_gripper_attachment_cfg,
    build_gripper_group_cfg,
)
from .paths import ASSET_DIR
from .scene_physics import build_box_flap_actuator, build_contact_box_spawn, configure_robot_asset_physics
from .robot_model import resolve_robot_model


ROBOT_MODEL = resolve_robot_model()
KUAVO_USD = Path(ROBOT_MODEL.usd_path)
OPEN_TOTE_USD = ASSET_DIR / "open_tote.usda"
BUTTON_STATION_USD = ASSET_DIR / "button_station.usda"
WORKCELL_GROUPS_USD = ASSET_DIR / "workcell_groups.usda"
RACK_USD_LOCAL = ASSET_DIR / "Rack.usd"
SMALL_BOX_USD = ASSET_DIR / "SmallBox.usd"
MEDIUM_BOX_USD = ASSET_DIR / "MediumBox.usd"
LARGE_BOX_USD = ASSET_DIR / "LargeBox.usd"
XLARGE_BOX_USD = ASSET_DIR / "XLargeBox.usd"
FACTORY_USD = f"{NUCLEUS_ASSET_ROOT_DIR}/Isaac/Environments/Simple_Warehouse/warehouse.usd"
RACK_USD = str(RACK_USD_LOCAL)
CONVEYOR_USD = (
    f"{NUCLEUS_ASSET_ROOT_DIR}/NVIDIA/Assets/DigitalTwin/Assets/Warehouse/"
    "Equipment/Conveyors/ConveyorBelt_A/ConveyorBelt_A08_PR_NVD_01.usd"
)

# Coordinates are expressed in the environment frame. Kuavo faces +X.
# Measured directly from assets/Rack.usd's shelf-ramp tilt (~5.114 deg).
RACK_SLOPE_DEG = 5.114147010769473
RACK_SLOPE_RAD = math.radians(RACK_SLOPE_DEG)
RACK_BOX_LOCAL_PITCH_QUAT = (
    math.cos(-RACK_SLOPE_RAD / 2.0),
    math.sin(-RACK_SLOPE_RAD / 2.0),
    0.0,
    0.0,
)
RACK_POSITION = layout_position("rack")
RACK_SCALE = layout_scale("rack")
RACK_BOX_WORLD_ROT = local_quat_to_world("rack", RACK_BOX_LOCAL_PITCH_QUAT)
RACK_BOX_SPAWN_PLAN = build_box_spawn_plan(
    RACK_BOX_LAYOUT,
    RACK_SLOPE_RAD,
    CAPTURED_RACK_BOX_POSE_PATH,
)
CONFIGURED_RACK_BOX_IDS = rack_instance_names(RACK_BOX_SPAWN_PLAN)
# The legacy front/back pair uses the same Rack.usd-local depth coordinates as
# the configurable local boxes. All rack poses therefore share one reference.
RACK_BOX_IDS = (
    "rack_tier0_front",
    "rack_tier0_back",
    "rack_tier1_front",
    "rack_tier1_back",
    "rack_tier2_front",
    "rack_tier2_back",
)

CONVEYOR_VISUAL_POS = layout_position("conveyor")
CONVEYOR_VISUAL_ROT = layout_rotation("conveyor")
CONVEYOR_GEOMETRY_ROT = remap_quat("conveyor", (1.0, 0.0, 0.0, 0.0))
CONVEYOR_OBJECT_ROT = CONVEYOR_GEOMETRY_ROT
CONVEYOR_SLOT_PITCH = 0.26
CONVEYOR_SLOTS = tuple(
    remap_point(
        "conveyor",
        (0.65 + CONVEYOR_SLOT_PITCH * slot_id, -0.52, 0.775),
    )
    for slot_id in range(9)
)
CONVEYOR_Y = CONVEYOR_SLOTS[0][1]
CONVEYOR_Z = CONVEYOR_SLOTS[0][2]
CONVEYOR_BELT_DIRECTION = rotate_default_vector("conveyor", (1.0, 0.0, 0.0))
# Button-station placement. Change this one tuple to move the complete yellow
# post, dark bezel, and spring-loaded green plunger together.
BUTTON_STATION_POS = layout_position("button_station")
BUTTON_STATION_ROT = layout_rotation("button_station")
BUTTON_PRESS_THRESHOLD = 0.006
BELT_SPEED = 0.22


def rack_box_position(lane_id: int, depth_id: int) -> tuple[float, float, float]:
    """Position of one tote resting on a real rack shelf surface.

    ``lane_id`` selects front (0, closest to Kuavo) or back (1, far side)
    along the rack's local depth axis; ``depth_id`` selects the tier
    (0 = bottom shelf ... 2 = top shelf).
    Totes sit directly on the measured ``RackShelf_*`` mesh surface, offset
    only by half the tote height so the tote base touches the shelf.
    """
    depth_raw = RACK_FRONT_ROW_DEPTH_RAW if lane_id == 0 else RACK_BACK_ROW_DEPTH_RAW
    return rack_shelf_point(
        depth_id + 1,
        depth_raw,
        RACK_SHELF_CENTER_LOCAL_X_RAW,
        0.012,
        RACK_SLOPE_RAD,
    )


def task_tote_position(lane_id: int, depth_id: int) -> tuple[float, float, float]:
    """Keep legacy oracle totes off the rack when local-box layout is active."""
    if CUSTOM_RACK_BOXES_ACTIVE:
        return (4.10 + 0.32 * lane_id, 1.45 + 0.34 * depth_id, 0.015)
    return rack_box_position(lane_id, depth_id)


def static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    pos: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    visible: bool = True,
    opacity: float = 1.0,
    metallic: float = 0.15,
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None,
) -> AssetBaseCfg:
    """Create a collidable static cuboid configuration."""
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=visible,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.65,
                metallic=metallic,
                opacity=opacity,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
    )


def tote_cfg(
    name: str,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> RigidObjectCfg:
    """Create one hollow open-top tote from the local compound-collider USD."""
    return RigidObjectCfg(
        prim_path=(
            f"{{ENV_REGEX_NS}}/Workcell/LegacyTask/Totes/{name}"
            if name.startswith("rack_")
            else f"{{ENV_REGEX_NS}}/Workcell/ConveyorSystem/ForeignTotes/{name}"
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OPEN_TOTE_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                linear_damping=0.04,
                angular_damping=0.08,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=position,
            rot=rotation,
        ),
    )


def build_tote_collection_cfg() -> RigidObjectCollectionCfg:
    """Build six task totes and three optional foreign-worker totes."""
    objects: dict[str, RigidObjectCfg] = {}
    for depth_id in range(3):
        for lane_id, lane_name in enumerate(("front", "back")):
            name = f"rack_tier{depth_id}_{lane_name}"
            objects[name] = tote_cfg(
                name,
                task_tote_position(lane_id, depth_id),
                RACK_BOX_WORLD_ROT,
            )

    for foreign_id in range(3):
        name = f"foreign_{foreign_id}"
        position = (
            CONVEYOR_SLOTS[foreign_id]
            if foreign_id < args_cli.prefill
            else (3.25 + 0.30 * foreign_id, 2.4, 0.015)
        )
        objects[name] = tote_cfg(
            name,
            position,
            CONVEYOR_OBJECT_ROT
            if foreign_id < args_cli.prefill
            else (1.0, 0.0, 0.0, 0.0),
        )

    return RigidObjectCollectionCfg(rigid_objects=objects)


# --- New local box assets (rack-configurable, otherwise floor-staged) -------
# Each box type shares an identical hollow open-top body with four
# free-swinging flap lids, authored in the source USD as a PhysX
# articulation (an enabled ArticulationRootAPI on `Body`). Isaac Lab's
# RigidObjectCfg raises at load time if it finds an enabled articulation
# root beneath its prim, so these must be spawned as ArticulationCfg
# instead of the RigidObjectCfg pattern used for the open-tote totes above.
#
# Two of each type are always spawned.  ``RACK_BOX_SPAWN_PLAN`` places the
# requested subset on shelves and leaves every unused instance in a labeled
# floor staging slot.  The shared parser/size dictionary lives in
# ``rack_box_layout.py``.
STAGING_BOX_TYPES = (
    ("SmallBox", SMALL_BOX_USD),
    ("MediumBox", MEDIUM_BOX_USD),
    ("LargeBox", LARGE_BOX_USD),
    ("XLargeBox", XLARGE_BOX_USD),
)
# The flap joints have no authored drive in the source USD (they are free
# hinges under gravity); a soft implicit actuator just damps them instead
# of leaving raw PhysX solver defaults, and covers all joints so Isaac Lab
# does not warn about un-actuated articulation DOFs.
BOX_FLAP_ACTUATOR = build_box_flap_actuator(FLAP_FRICTION)


def staging_box_cfg(box_name: str, usd_path: Path, index: int) -> ArticulationCfg:
    """Spawn one hollow box at its selected rack or floor-staging pose."""
    instance_name = f"{box_name}_{index}"
    spec = RACK_BOX_SPAWN_PLAN[instance_name]
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Workcell/StagingBoxes/{instance_name}",
        spawn=build_contact_box_spawn(usd_path, spec.scale),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=spec.position,
            rot=spec.rotation,
            joint_pos={
                "joint_front": 0.0,
                "joint_back": 0.0,
                "joint_left": 0.0,
                "joint_right": 0.0,
            },
            joint_vel={".*": 0.0},
        ),
        actuators={"flaps": BOX_FLAP_ACTUATOR},
    )


LOCAL_BOX_SCENE_KEYS = tuple(
    spec.scene_key for spec in RACK_BOX_SPAWN_PLAN.values()
)


def randomize_box_flap_joint_friction(scene: InteractiveScene) -> None:
    """Sample per-box, per-flap PhysX static/dynamic joint friction."""
    for scene_key in LOCAL_BOX_SCENE_KEYS:
        box = scene[scene_key]
        static = torch.empty_like(box.data.joint_pos).uniform_(*FLAP_FRICTION.static_range)
        dynamic = torch.empty_like(box.data.joint_pos).uniform_(*FLAP_FRICTION.dynamic_range)
        dynamic = torch.minimum(dynamic, static)
        box.write_joint_friction_coefficient_to_sim(static)
        box.write_joint_dynamic_friction_coefficient_to_sim(dynamic)


KUAVO_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Kuavo",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(KUAVO_USD),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=2.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Both packaged wheel models use a ground-aligned fixed-base root.
        pos=layout_position("robot"),
        rot=layout_rotation("robot"),
        joint_pos={
            "wheel_.*_joint": 0.0,
            "knee_joint": 0.0,
            "leg_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "zarm_.*_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "zhead_.*_joint": 0.0,
            **(
                {"[lr]_[fb]_bar_[13]_joint": 0.0}
                if ROBOT_MODEL.has_integrated_grippers
                else {}
            ),
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "height_axis": ImplicitActuatorCfg(
            joint_names_expr=["knee_joint", "leg_joint", "waist_pitch_joint"],
            effort_limit_sim=700.0,
            velocity_limit_sim=25.0,
            stiffness=400.0,
            damping=40.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_.*_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=10.0,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=["zarm_.*_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=20.0,
            stiffness=220.0,
            damping=22.0,
        ),
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "zhead_.*_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=10.0,
            stiffness=120.0,
            damping=15.0,
        ),
        **(
            {
                "integrated_grippers": ImplicitActuatorCfg(
                    joint_names_expr=["[lr]_[fb]_bar_[13]_joint"],
                    effort_limit_sim=5.0,
                    velocity_limit_sim=5.0,
                    stiffness=100.0,
                    damping=10.0,
                    friction=0.02,
                )
            }
            if ROBOT_MODEL.has_integrated_grippers
            else {}
        ),
    },
)

configure_robot_asset_physics(KUAVO_CFG, ROBOT_MODEL, GRIPPER_SETTINGS)

BUTTON_STATION_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/ButtonStation",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(BUTTON_STATION_USD),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=BUTTON_STATION_POS,
        rot=BUTTON_STATION_ROT,
        joint_pos={"ButtonJoint": 0.0},
        joint_vel={"ButtonJoint": 0.0},
    ),
    actuators={
        "return_spring": ImplicitActuatorCfg(
            joint_names_expr=["ButtonJoint"],
            effort_limit_sim=120.0,
            velocity_limit_sim=1.0,
            stiffness=420.0,
            damping=18.0,
        ),
    },
)


@configclass
class RackToConveyorSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the rack-to-conveyor transfer task."""

    factory = AssetBaseCfg(
        prim_path="/World/Factory",
        spawn=sim_utils.UsdFileCfg(usd_path=FACTORY_USD),
    )

    # Backup collision plane sits just below the warehouse floor and is not
    # visible from the working area.
    ground = static_cuboid(
        "/World/Ground",
        (20.0, 20.0, 0.10),
        (0.0, 0.0, -0.06),
        (0.12, 0.13, 0.14),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=550.0,
            color=(0.72, 0.80, 0.92),
            visible_in_primary_ray=False,
        ),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=2100.0,
            color=(1.0, 0.91, 0.78),
            angle=1.2,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.9239, 0.0, 0.3827, 0.0)),
    )
    work_light_left = AssetBaseCfg(
        prim_path="/World/WorkLightLeft",
        spawn=sim_utils.CylinderLightCfg(
            intensity=9000.0,
            color=(0.82, 0.91, 1.0),
            length=2.2,
            radius=0.035,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, 1.15, 3.2),
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )
    work_light_right = AssetBaseCfg(
        prim_path="/World/WorkLightRight",
        spawn=sim_utils.CylinderLightCfg(
            intensity=9000.0,
            color=(0.82, 0.91, 1.0),
            length=2.2,
            radius=0.035,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.8, -1.15, 3.2),
            rot=(0.7071, 0.0, 0.7071, 0.0),
        ),
    )

    # Spawn the editing hierarchy before any nested Isaac Lab assets. This
    # keeps the Stage tree manageable and satisfies ENV_REGEX_NS cloning.
    workcell_groups: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    racks_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/Racks",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    rack: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/Racks/Rack",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(WORKCELL_GROUPS_USD),
            scale=RACK_SCALE,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=RACK_POSITION,
            rot=layout_rotation("rack"),
        ),
    )
    legacy_task_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/LegacyTask",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    legacy_totes_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/LegacyTask/Totes",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    safety_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    fence_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/Fence",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    conveyor_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    foreign_totes_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/ForeignTotes",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    cameras_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/Cameras",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    staging_boxes_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/StagingBoxes",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )

    robot: ArticulationCfg = KUAVO_CFG
    # Spawn the common parent before the child gripper articulations.
    grippers_group: AssetBaseCfg | None = build_gripper_group_cfg(GRIPPER_SETTINGS)
    left_gripper: ArticulationCfg | None = build_gripper_articulation_cfg(
        GRIPPER_SETTINGS, "left"
    )
    right_gripper: ArticulationCfg | None = build_gripper_articulation_cfg(
        GRIPPER_SETTINGS, "right"
    )
    gripper_attachments: AssetBaseCfg | None = build_gripper_attachment_cfg(
        GRIPPER_SETTINGS
    )
    button_station: ArticulationCfg = BUTTON_STATION_CFG

    # Two instances of every local box USD are always present. The shared
    # spawn plan selects rack or floor-staging placement for each instance.
    small_box_0: ArticulationCfg = staging_box_cfg("SmallBox", SMALL_BOX_USD, 0)
    small_box_1: ArticulationCfg = staging_box_cfg("SmallBox", SMALL_BOX_USD, 1)
    medium_box_0: ArticulationCfg = staging_box_cfg("MediumBox", MEDIUM_BOX_USD, 0)
    medium_box_1: ArticulationCfg = staging_box_cfg("MediumBox", MEDIUM_BOX_USD, 1)
    large_box_0: ArticulationCfg = staging_box_cfg("LargeBox", LARGE_BOX_USD, 0)
    large_box_1: ArticulationCfg = staging_box_cfg("LargeBox", LARGE_BOX_USD, 1)
    xlarge_box_0: ArticulationCfg = staging_box_cfg("XLargeBox", XLARGE_BOX_USD, 0)
    xlarge_box_1: ArticulationCfg = staging_box_cfg("XLargeBox", XLARGE_BOX_USD, 1)

    # Rack owns the captured world pose/scale. The supplied USD is composed
    # below it at an identity local transform so the Stage hierarchy remains
    # predictable and editable.
    rack_visual = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/Racks/Rack/Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=RACK_USD,
            scale=(1.0, 1.0, 1.0),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # Safety fence alongside the racks, modeled after the supplied factory
    # image.  The dark panel is translucent so the rack remains visible.
    fence_panel = static_cuboid(
        "{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/Panel",
        (1.55, 0.025, 1.55),
        layout_position("fence"),
        (0.035, 0.045, 0.055),
        rot=layout_rotation("fence"),
        opacity=0.42,
        metallic=0.85,
    )
    fence_top_rail = static_cuboid(
        "{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/TopRail",
        (1.62, 0.055, 0.055),
        layout_offset("fence", (0.0, 0.0, 0.78)),
        (0.03, 0.035, 0.04),
        rot=remap_quat("fence", (1.0, 0.0, 0.0, 0.0)),
        metallic=0.9,
    )
    fence_back_post = static_cuboid(
        "{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/BackPost",
        (0.065, 0.065, 1.78),
        layout_offset("fence", (-0.64, 0.0, -0.03)),
        (0.04, 0.05, 0.06),
        rot=remap_quat("fence", (1.0, 0.0, 0.0, 0.0)),
        metallic=0.9,
    )
    button_probe = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/ButtonVerificationProbe",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.06),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(-4.0, -4.0, 1.0)),
    )

    # Detailed NVIDIA Digital Twin conveyor. The invisible physical deck below
    # remains the authoritative collision and belt-motion surface.
    conveyor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CONVEYOR_USD,
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=CONVEYOR_VISUAL_POS,
            rot=CONVEYOR_VISUAL_ROT,
        ),
    )
    conveyor_surface = AssetBaseCfg(
        # OmniGraph graphs cannot be authored below Isaac Lab's cloned
        # environment scope. This task has one environment, so the physical
        # deck lives at /World while remaining aligned with the visual asset.
        prim_path="/World/ConveyorSurface",
        spawn=sim_utils.CuboidCfg(
            size=(2.55, 0.68, 0.03),
            visible=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.0,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=layout_offset("conveyor", (1.29, 0.43, 0.753)),
            rot=CONVEYOR_GEOMETRY_ROT,
        ),
    )

    infeed_region = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/InfeedRegion",
        spawn=sim_utils.CuboidCfg(
            size=(0.24, 0.28, 0.006),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.95, 0.70, 0.04),
                emissive_color=(0.12, 0.06, 0.0),
                opacity=0.35,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=CONVEYOR_SLOTS[0],
            rot=CONVEYOR_GEOMETRY_ROT,
        ),
    )

    totes: RigidObjectCollectionCfg = build_tote_collection_cfg()

    # Robot-authored head camera plus a virtual waist policy view.
    head_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{ROBOT_MODEL.head_camera_body}/HeadCamera",
        update_period=0.0,
        height=120,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.08, 8.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=ROBOT_MODEL.head_camera_mount.pos,
            rot=ROBOT_MODEL.head_camera_mount.rot,
            convention="ros",
        ),
    )
    waist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo/waist_yaw_link/WaistCamera",
        update_period=0.0,
        height=120,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0,
            focus_distance=1.5,
            horizontal_aperture=24.0,
            clipping_range=(0.05, 6.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.10, 0.0, 0.05),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )
    # S200062: physical D405 position plus body-to-ROS-optical rotation.
    # S63: adapted rig for Robotiq's opposite (+Z) finger reach. Both sensor
    # poses use ROS optical axes (+Z forward, -Y up), not body +X forward.
    left_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{ROBOT_MODEL.wrist_camera_bodies['left']}/LeftWristCamera",
        update_period=0.0,
        height=120,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=0.4,
            horizontal_aperture=24.0,
            clipping_range=(0.03, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=ROBOT_MODEL.wrist_camera_mounts["left"].pos,
            rot=ROBOT_MODEL.wrist_camera_mounts["left"].rot,
            convention="ros",
        ),
    )
    right_wrist_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{ROBOT_MODEL.wrist_camera_bodies['right']}/RightWristCamera",
        update_period=0.0,
        height=120,
        width=160,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=12.0,
            focus_distance=0.4,
            horizontal_aperture=24.0,
            clipping_range=(0.03, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=ROBOT_MODEL.wrist_camera_mounts["right"].pos,
            rot=ROBOT_MODEL.wrist_camera_mounts["right"].rot,
            convention="ros",
        ),
    )

    preview_camera = (
        CameraCfg(
            prim_path="{ENV_REGEX_NS}/Workcell/Cameras/PreviewCamera",
            update_period=0.0,
            height=720,
            width=1280,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=22.0,
                focus_distance=4.0,
                horizontal_aperture=24.0,
                clipping_range=(0.05, 1000.0),
            ),
        )
        if args_cli.screenshot is not None
        else None
    )


def setup_conveyor_belt() -> None:
    """Apply the PhysX surface-velocity API, initially disabled."""
    stage = stage_utils.get_current_stage()
    conveyor_prim = stage.GetPrimAtPath("/World/ConveyorSurface")
    if not conveyor_prim.IsValid():
        raise RuntimeError("Conveyor collision surface was not created.")

    surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI.Apply(conveyor_prim)
    surface_velocity.GetSurfaceVelocityAttr().Set(
        Gf.Vec3f(*(BELT_SPEED * component for component in CONVEYOR_BELT_DIRECTION))
    )
    surface_velocity.GetSurfaceVelocityEnabledAttr().Set(False)
    surface_velocity.GetSurfaceVelocityLocalSpaceAttr().Set(False)


def set_conveyor_enabled(enabled: bool) -> None:
    """Start or stop the physical conveyor deck."""
    stage = stage_utils.get_current_stage()
    conveyor_prim = stage.GetPrimAtPath("/World/ConveyorSurface")
    surface_velocity = PhysxSchema.PhysxSurfaceVelocityAPI(conveyor_prim)
    surface_velocity.GetSurfaceVelocityEnabledAttr().Set(enabled)


def setup_workcell_details() -> None:
    """Spawn conveyor slot markers and fence mesh lines."""
    slot_cfg = sim_utils.CuboidCfg(
        size=(0.225, 0.275, 0.004),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.05, 0.78, 0.24),
            emissive_color=(0.0, 0.04, 0.006),
            opacity=0.16,
            roughness=0.72,
        ),
    )
    for slot_id, center in enumerate(CONVEYOR_SLOTS[1:], start=1):
        slot_cfg.func(
            f"/World/envs/env_0/Workcell/ConveyorSystem/Slots/Slot{slot_id:02d}",
            slot_cfg,
            translation=(center[0], center[1], center[2] - 0.003),
            orientation=CONVEYOR_GEOMETRY_ROT,
        )

    fence_wire_cfg = sim_utils.CuboidCfg(
        size=(1.47, 0.012, 0.014),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.025, 0.03, 0.035),
            roughness=0.30,
            metallic=0.92,
        ),
    )
    for wire_id in range(9):
        fence_wire_cfg.func(
            f"/World/envs/env_0/Workcell/SafetySystem/Fence/Wires/Horizontal{wire_id:02d}",
            fence_wire_cfg,
            translation=layout_offset("fence", (0.0, -0.018, -0.65 + 0.16 * wire_id)),
            orientation=remap_quat("fence", (1.0, 0.0, 0.0, 0.0)),
        )
    fence_vertical_cfg = sim_utils.CuboidCfg(
        size=(0.014, 0.012, 1.43),
        visual_material=fence_wire_cfg.visual_material,
    )
    for wire_id in range(13):
        fence_vertical_cfg.func(
            f"/World/envs/env_0/Workcell/SafetySystem/Fence/Wires/Vertical{wire_id:02d}",
            fence_vertical_cfg,
            translation=layout_offset(
                "fence",
                (-0.58 + 0.095 * wire_id, -0.019, -0.01),
            ),
            orientation=remap_quat("fence", (1.0, 0.0, 0.0, 0.0)),
        )


def set_button_visual(phase: TaskPhase) -> None:
    """Show disabled, armed, and accepted button states with light emission."""
    if phase is TaskPhase.TRANSFERRING:
        diffuse, emissive = (0.02, 0.34, 0.055), (0.0, 0.025, 0.002)
    elif phase is TaskPhase.WAITING_FOR_BUTTON:
        diffuse, emissive = (0.03, 0.92, 0.12), (0.02, 0.42, 0.045)
    else:
        diffuse, emissive = (0.08, 1.0, 0.24), (0.05, 0.75, 0.12)

    stage = stage_utils.get_current_stage()
    root_path = "/World/envs/env_0/Workcell/SafetySystem/ButtonStation/Plunger"
    for prim in stage.Traverse():
        if prim.GetPath().pathString.startswith(root_path):
            diffuse_attr = prim.GetAttribute("inputs:diffuseColor")
            emissive_attr = prim.GetAttribute("inputs:emissiveColor")
            if diffuse_attr.IsValid():
                diffuse_attr.Set(Gf.Vec3f(*diffuse))
            if emissive_attr.IsValid():
                emissive_attr.Set(Gf.Vec3f(*emissive))


def reset_scene(scene: InteractiveScene) -> None:
    """Reset the robot and all task/foreign totes."""
    robot = scene["robot"]
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(
        robot.data.default_joint_pos.clone(),
        robot.data.default_joint_vel.clone(),
    )
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())

    button_station = scene["button_station"]
    button_station.write_joint_state_to_sim(
        button_station.data.default_joint_pos.clone(),
        button_station.data.default_joint_vel.clone(),
    )
    button_station.set_joint_position_target(button_station.data.default_joint_pos.clone())

    totes: RigidObjectCollection = scene["totes"]
    tote_state = totes.data.default_object_state.clone()
    tote_state[..., :3] += scene.env_origins.unsqueeze(1)
    totes.write_object_link_pose_to_sim(tote_state[..., :7])
    totes.write_object_com_velocity_to_sim(tote_state[..., 7:])

    scene.reset()


def tote_positions(scene: InteractiveScene) -> dict[str, tuple[float, float, float]]:
    """Return current tote root positions in the environment frame."""
    totes: RigidObjectCollection = scene["totes"]
    positions = totes.data.object_link_pos_w[0] - scene.env_origins[0]
    return {
        name: tuple(float(value) for value in positions[index].tolist())
        for index, name in enumerate(totes.object_names)
    }


def workcell_box_positions(scene: InteractiveScene) -> dict[str, tuple[float, float, float]]:
    """Return conveyor objects plus the selected local-box task instances."""
    positions = tote_positions(scene)
    for instance_name in CONFIGURED_RACK_BOX_IDS:
        spec = RACK_BOX_SPAWN_PLAN[instance_name]
        root_position = scene[spec.scene_key].data.root_pos_w[0] - scene.env_origins[0]
        positions[instance_name] = tuple(float(value) for value in root_position.tolist())
    return positions


def button_is_pressed(scene: InteractiveScene) -> bool:
    """Read the real prismatic-plunger displacement."""
    displacement = scene["button_station"].data.joint_pos[0, 0]
    return bool(displacement >= BUTTON_PRESS_THRESHOLD)


def set_button_probe_pose(scene: InteractiveScene, y_position: float) -> None:
    """Move the verification probe while preserving normal scene behavior."""
    probe = scene["button_probe"]
    pose = probe.data.root_pose_w.clone()
    probe_position = remap_point("button_station", (0.45, y_position, 1.15))
    pose[0, :3] = scene.env_origins[0] + torch.tensor(probe_position, device=pose.device)
    pose[0, 3:7] = torch.tensor(
        remap_quat("button_station", (1.0, 0.0, 0.0, 0.0)),
        device=pose.device,
    )
    probe.write_root_pose_to_sim(pose)


def apply_demo_plan(
    scene: InteractiveScene,
    slot_manager: ConveyorSlotManager,
    plan: PlacementPlan,
) -> None:
    """Execute an oracle placement plan by writing tote poses.

    This is intentionally isolated from the task logic.  A real Kuavo
    controller should execute the same plan through IK/grasp actions, then call
    ``slot_manager.commit(plan)`` only after verifying the release pose.
    """
    if plan.mode not in (PlacementMode.PLACE, PlacementMode.PUSH_THEN_PLACE):
        return

    totes: RigidObjectCollection = scene["totes"]
    poses = totes.data.object_link_pose_w.clone()
    velocities = torch.zeros_like(totes.data.object_com_vel_w)
    positions = tote_positions(scene)
    name_to_id = {name: object_id for object_id, name in enumerate(totes.object_names)}

    if plan.mode is PlacementMode.PUSH_THEN_PLACE:
        occupied = slot_manager.occupied_slots(positions)
        box_to_slot = {box_id: slot_id for slot_id, box_id in occupied.items()}
        for push_box_id in plan.push_box_ids:
            old_slot = box_to_slot[push_box_id]
            next_center = CONVEYOR_SLOTS[old_slot + 1]
            object_id = name_to_id[push_box_id]
            poses[0, object_id, :3] = torch.tensor(next_center, device=poses.device)
            poses[0, object_id, 3:7] = torch.tensor(CONVEYOR_OBJECT_ROT, device=poses.device)

    object_id = name_to_id[plan.box_id]
    poses[0, object_id, :3] = torch.tensor(plan.target, device=poses.device)
    poses[0, object_id, 3:7] = torch.tensor(CONVEYOR_OBJECT_ROT, device=poses.device)
    totes.write_object_link_pose_to_sim(poses)
    totes.write_object_com_velocity_to_sim(velocities)
    slot_manager.commit(plan)


def transfer_demo_box(
    scene: InteractiveScene,
    slot_manager: ConveyorSlotManager,
    box_id: str,
) -> bool:
    """Reserve and execute one scripted transfer, including queue pushing."""
    positions = tote_positions(scene)
    plan = slot_manager.reserve("kuavo_0", box_id, positions)
    if plan.mode is PlacementMode.BLOCKED:
        print(f"[WARN] Conveyor is full; cannot insert {box_id}.")
        return False
    if plan.mode is PlacementMode.WAIT:
        print(f"[INFO] Infeed is reserved by another worker; {box_id} waits.")
        return False

    if plan.mode is PlacementMode.PUSH_THEN_PLACE:
        print(
            f"[PLAN] Push {len(plan.push_box_ids)} queued tote(s) by "
            f"{plan.push_distance:.2f} m, then place {box_id}."
        )
    else:
        print(f"[PLAN] Empty infeed detected; place {box_id} directly.")
    apply_demo_plan(scene, slot_manager, plan)
    return True


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Monitor the complete task and optionally run its oracle demonstration."""
    sim_dt = sim.get_physics_dt()
    max_steps = args_cli.steps
    step_count = 0
    previous_phase = TaskPhase.TRANSFERRING
    button_was_pressed = False
    max_button_displacement = 0.0
    verification_press_seen = False
    verification_box_id = 0
    belt_started = False
    demo_pair_id = 0
    demo_event_times = tuple(value / args_cli.demo_speed for value in (3.0, 9.5, 16.0))
    demo_button_time = 20.0 / args_cli.demo_speed

    reset_scene(scene)
    set_conveyor_enabled(False)
    slot_manager = ConveyorSlotManager(CONVEYOR_SLOTS)
    active_task_box_ids = CONFIGURED_RACK_BOX_IDS or RACK_BOX_IDS
    active_task_box_count = len(active_task_box_ids)
    task = RackConveyorTask(active_task_box_ids)
    print("[INFO] Kuavo single-rack gravity-feed workcell is ready.")
    print(f"[INFO] Shared workcell layout: {LAYOUT_PATH}")
    print(
        f"[INFO] Rack slope: {RACK_SLOPE_DEG:.1f} degrees; "
        f"105.1x88.1x216.5 cm rack, three shelf tiers."
    )
    print(f"[INFO] Local rack-box layout: {format_rack_box_layout(RACK_BOX_LAYOUT)}")
    if CAPTURED_RACK_BOX_POSE_PATH is not None:
        print(f"[INFO] Captured Rack-anchor-relative box poses: {CAPTURED_RACK_BOX_POSE_PATH}")
    if FLAP_FRICTION.randomize:
        print(
            "[INFO] Flap joint friction randomized at startup: "
            f"static={FLAP_FRICTION.static_range}, dynamic={FLAP_FRICTION.dynamic_range}."
        )
    else:
        print(
            "[INFO] Flap joint friction fixed: "
            f"static={FLAP_FRICTION.static}, dynamic={FLAP_FRICTION.dynamic}."
        )
    print(f"[INFO] Conveyor has {len(CONVEYOR_SLOTS)} reserved slots; {args_cli.prefill} pre-filled.")
    print(
        "[INFO] Button input uses the ButtonStation prismatic joint "
        f"(threshold={BUTTON_PRESS_THRESHOLD * 1000.0:.1f} mm)."
    )
    print(
        f"[INFO] Conveyor is stopped until all {active_task_box_count} task boxes "
        "are loaded and the green button is pressed."
    )
    if args_cli.auto_demo:
        print("[INFO] Oracle task-system demo enabled; this is not a robot manipulation policy.")
    if args_cli.verify_button:
        print("[VERIFY] Kinematic probe will physically press and release the button.")

    while simulation_app.is_running() and (max_steps <= 0 or step_count < max_steps):
        scene["robot"].set_joint_position_target(scene["robot"].data.default_joint_pos)
        scene["button_station"].set_joint_position_target(
            scene["button_station"].data.default_joint_pos
        )
        if args_cli.verify_button:
            if step_count < 40:
                probe_y = 0.70
            elif step_count < 120:
                probe_y = 0.70 + 0.105 * ((step_count - 40) / 79.0)
            elif step_count < 160:
                probe_y = 0.805
            elif step_count < 220:
                probe_y = 0.805 - 0.105 * ((step_count - 160) / 59.0)
            else:
                probe_y = 0.70
            set_button_probe_pose(scene, probe_y)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        elapsed = step_count * sim_dt

        positions = workcell_box_positions(scene)
        occupied = slot_manager.occupied_slots(positions)
        task.update_transferred(tuple(occupied.values()))

        if args_cli.verify_button and verification_box_id < len(RACK_BOX_IDS):
            if step_count >= 2 + 3 * verification_box_id:
                if transfer_demo_box(
                    scene,
                    slot_manager,
                    RACK_BOX_IDS[verification_box_id],
                ):
                    verification_box_id += 1

        if args_cli.auto_demo and demo_pair_id < len(demo_event_times):
            if elapsed >= demo_event_times[demo_pair_id]:
                pair = (
                    RACK_BOX_IDS[demo_pair_id * 2],
                    RACK_BOX_IDS[demo_pair_id * 2 + 1],
                )
                if all(transfer_demo_box(scene, slot_manager, box_id) for box_id in pair):
                    demo_pair_id += 1

        button_displacement = float(scene["button_station"].data.joint_pos[0, 0].item())
        max_button_displacement = max(max_button_displacement, button_displacement)
        pressed = button_is_pressed(scene)
        verification_press_seen |= pressed
        button_rising_edge = pressed and not button_was_pressed
        button_was_pressed = pressed
        auto_button = args_cli.auto_demo and elapsed >= demo_button_time
        if (button_rising_edge or auto_button) and task.phase is not TaskPhase.COMPLETE:
            if task.press_button():
                source = "oracle demo" if auto_button else "physical plunger travel"
                print(f"[BUTTON] Valid green-button press accepted from {source}.")
            elif button_rising_edge:
                print(
                    f"[BUTTON] Ignored early press: {len(task.remaining_box_ids)} "
                    "rack box(es) remain."
                )

        if task.phase is not previous_phase:
            set_button_visual(task.phase)
            if task.phase is TaskPhase.WAITING_FOR_BUTTON:
                print(
                    f"[TASK] All {active_task_box_count} rack boxes are on the conveyor. "
                    "Green button is now armed."
                )
            previous_phase = task.phase

        if task.conveyor_enabled and not belt_started:
            set_conveyor_enabled(True)
            belt_started = True
            print(
                f"[SUCCESS] Task complete at {elapsed:.2f} s. "
                f"Conveyor started at {BELT_SPEED:.2f} m/s."
            )
        step_count += 1

    final_positions = workcell_box_positions(scene)
    final_occupied = slot_manager.occupied_slots(final_positions)
    print(
        f"[INFO] Finished after {step_count} steps. Phase={task.phase.name}, "
        f"occupied_slots={len(final_occupied)}, "
        f"transferred={len(task.transferred_box_ids)}/{active_task_box_count}."
    )
    if args_cli.verify_button:
        released = float(scene["button_station"].data.joint_pos[0, 0].item()) < 0.002
        task_completed = task.phase is TaskPhase.COMPLETE and belt_started
        print(
            f"[VERIFY] button max_travel={max_button_displacement * 1000.0:.2f} mm, "
            f"press_detected={verification_press_seen}, spring_returned={released}, "
            f"physical_press_completed_task={task_completed}."
        )
        if not verification_press_seen or not released or not task_completed:
            raise RuntimeError("Physical button/task verification failed.")


def save_camera_preview(
    sim: SimulationContext,
    scene: InteractiveScene,
    output_path: Path,
) -> None:
    """Render and save a camera preview after referenced assets have loaded."""
    from PIL import Image

    camera = scene["preview_camera"]
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera.set_world_poses_from_view(
        eyes=torch.tensor([[3.45, -3.35, 2.45]], device=sim.device),
        targets=torch.tensor([[0.85, 0.0, 0.85]], device=sim.device),
    )

    # Give streamed meshes, materials, and RTX lighting time to settle.
    for _ in range(90):
        sim.render()
        camera.update(sim.get_physics_dt(), force_recompute=True)

    rgb = camera.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
    Image.fromarray(rgb).save(output_path)
    print(f"[INFO] Saved camera preview to: {output_path}")


def main() -> None:
    """Create, optionally save, and run the scene."""
    if not KUAVO_USD.is_file():
        raise FileNotFoundError(
            f"Converted Kuavo USD not found: {KUAVO_USD}\n"
            "Run ./convert_kuavo.sh before launching the scene."
        )
    if not OPEN_TOTE_USD.is_file():
        raise FileNotFoundError(f"Open-tote USD not found: {OPEN_TOTE_USD}")
    if not BUTTON_STATION_USD.is_file():
        raise FileNotFoundError(f"Button-station USD not found: {BUTTON_STATION_USD}")
    if not WORKCELL_GROUPS_USD.is_file():
        raise FileNotFoundError(f"Workcell-groups USD not found: {WORKCELL_GROUPS_USD}")

    sim_cfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        device=args_cli.device,
        gravity=(0.0, 0.0, -9.81),
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(3.45, -3.35, 2.45), target=(0.85, 0.0, 0.85))

    scene = InteractiveScene(RackToConveyorSceneCfg(num_envs=1, env_spacing=3.0))
    setup_conveyor_belt()
    setup_workcell_details()
    sim.reset()
    print(
        f"[INFO] Gripper preset: {GRIPPER_SETTINGS.name} "
        f"(active sides: {GRIPPER_SETTINGS.active_sides or 'none'})."
    )
    if FLAP_FRICTION.randomize:
        randomize_box_flap_joint_friction(scene)
    set_button_visual(TaskPhase.TRANSFERRING)
    open_camera_viewports(
        scene,
        ["head_camera", "waist_camera", "left_wrist_camera", "right_wrist_camera"],
        headless=args_cli.headless,
    )

    if args_cli.save_stage is not None:
        stage_path = args_cli.save_stage.expanduser().resolve()
        stage_path.parent.mkdir(parents=True, exist_ok=True)
        if not stage_utils.save_stage(str(stage_path), save_and_reload_in_place=False):
            raise RuntimeError(f"Failed to save stage: {stage_path}")
        print(f"[INFO] Saved composed scene to: {stage_path}")

    if args_cli.screenshot is not None:
        save_camera_preview(sim, scene, args_cli.screenshot)

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    if args_cli.steps > 0:
        # Kit can block while unloading extensions on this local Isaac Sim 5.0
        # installation. Finite batch physics, capture, and USD writes are
        # already complete.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    simulation_app.close()
