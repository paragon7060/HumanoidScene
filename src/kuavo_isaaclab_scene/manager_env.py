"""Isaac Lab ManagerBasedRLEnv configuration for robustness training/evaluation."""

from __future__ import annotations

import math
from pathlib import Path

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    ArticulationCfg,
    AssetBaseCfg,
    RigidObjectCfg,
    RigidObjectCollectionCfg,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.simulation_cfg import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import manager_mdp as workcell_mdp
from .box_flap_friction import resolve_flap_friction_settings
from .groot_lerobot_bridge import CONTROLLED_JOINT_NAMES
from .gripper_config import resolve_gripper_settings
from .gripper_runtime import (
    build_gripper_action_cfg,
    build_gripper_articulation_cfg,
    build_gripper_attachment_cfg,
    build_gripper_group_cfg,
)
from .paths import ASSET_DIR
from .rack_box_layout import (
    RACK_BACK_ROW_DEPTH_RAW,
    RACK_FRONT_ROW_DEPTH_RAW,
    RACK_SHELF_CENTER_LOCAL_X_RAW,
    build_box_spawn_plan,
    rack_instance_names,
    rack_shelf_point,
    resolve_rack_box_layout,
    resolve_rack_box_pose_path,
)
from .workcell_layout import (
    local_quat_to_world,
    offset as layout_offset,
    position as layout_position,
    quat_rotate,
    remap_point,
    remap_quat,
    rotation as layout_rotation,
    scale as layout_scale,
)
from .robot_model import resolve_robot_model


ROBOT_MODEL = resolve_robot_model()
KUAVO_USD = Path(ROBOT_MODEL.usd_path)
OPEN_TOTE_USD = ASSET_DIR / "open_tote.usda"
SAFETY_WORKER_USD = ASSET_DIR / "safety_worker.usda"
MOBILE_ROBOT_USD = ASSET_DIR / "mobile_robot.usda"
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
RACK_BOX_LAYOUT = resolve_rack_box_layout()
CAPTURED_RACK_BOX_POSE_PATH = resolve_rack_box_pose_path()
RACK_BOX_SPAWN_PLAN = build_box_spawn_plan(
    RACK_BOX_LAYOUT,
    RACK_SLOPE_RAD,
    CAPTURED_RACK_BOX_POSE_PATH,
)
CONFIGURED_RACK_BOX_COUNT = sum(len(boxes) for boxes in RACK_BOX_LAYOUT.values())
CUSTOM_RACK_BOXES_ACTIVE = bool(
    CONFIGURED_RACK_BOX_COUNT or CAPTURED_RACK_BOX_POSE_PATH is not None
)
FLAP_FRICTION = resolve_flap_friction_settings(randomize_default=True)
GRIPPER_SETTINGS = resolve_gripper_settings()
FLAP_STATIC_RESET_RANGE = (
    FLAP_FRICTION.static_range
    if FLAP_FRICTION.randomize
    else (FLAP_FRICTION.static, FLAP_FRICTION.static)
)
FLAP_DYNAMIC_RESET_RANGE = (
    FLAP_FRICTION.dynamic_range
    if FLAP_FRICTION.randomize
    else (FLAP_FRICTION.dynamic, FLAP_FRICTION.dynamic)
)
LOCAL_BOX_SCENE_KEYS = (
    "small_box_0",
    "small_box_1",
    "medium_box_0",
    "medium_box_1",
    "large_box_0",
    "large_box_1",
    "xlarge_box_0",
    "xlarge_box_1",
)
ACTIVE_RACK_BOX_INSTANCE_NAMES = rack_instance_names(RACK_BOX_SPAWN_PLAN)
ACTIVE_RACK_BOX_SCENE_KEYS = tuple(
    RACK_BOX_SPAWN_PLAN[name].scene_key for name in ACTIVE_RACK_BOX_INSTANCE_NAMES
)
TASK_OBJECT_PARAMS = {"box_asset_names": ACTIVE_RACK_BOX_SCENE_KEYS}


def rack_box_position(lane_id: int, depth_id: int) -> tuple[float, float, float]:
    """Position of one tote resting on a real rack shelf surface.

    ``lane_id`` selects front (0, closest to Kuavo) or back (1, far side)
    along the rack's local depth axis; ``depth_id`` selects the tier
    (0 = bottom shelf ... 2 = top shelf).
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
    """Park legacy manager-task totes away when local rack boxes are selected."""
    if CUSTOM_RACK_BOXES_ACTIVE:
        return (4.10 + 0.32 * lane_id, 1.45 + 0.34 * depth_id, 0.015)
    return rack_box_position(lane_id, depth_id)


def static_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    opacity: float = 1.0,
    metallic: float = 0.15,
) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.62,
                metallic=metallic,
                opacity=opacity,
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position, rot=rotation),
    )


def kinematic_cuboid(
    prim_path: str,
    size: tuple[float, float, float],
    position: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    visible: bool = True,
    physics_material: sim_utils.RigidBodyMaterialCfg | None = None,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            visible=visible,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.0,
            ),
            physics_material=physics_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=color,
                roughness=0.45,
                metallic=0.72,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=position, rot=rotation),
    )


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
        pos=workcell_mdp.BUTTON_STATION_POS,
        rot=layout_rotation("button_station"),
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


# --- New local box assets (rack-configurable, otherwise floor-staged) -------
# Each box type shares an identical hollow open-top body with four
# free-swinging flap lids, authored in the source USD as a PhysX
# articulation (an enabled ArticulationRootAPI on `Body`). Isaac Lab's
# RigidObjectCfg raises at load time if it finds an enabled articulation
# root beneath its prim, so these must be spawned as ArticulationCfg
# instead of the RigidObjectCfg pattern used for the open-tote totes above.
#
# Two of each type are always spawned. The shared rack-box layout selects the
# subset placed on shelves and leaves the remainder in floor staging slots.
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
BOX_FLAP_ACTUATOR = ImplicitActuatorCfg(
    joint_names_expr=["joint_front", "joint_back", "joint_left", "joint_right"],
    effort_limit_sim=5.0,
    velocity_limit_sim=10.0,
    stiffness=0.0,
    damping=0.05,
    friction=FLAP_FRICTION.static,
    dynamic_friction=FLAP_FRICTION.dynamic,
)


def staging_box_cfg(box_name: str, usd_path: Path, index: int) -> ArticulationCfg:
    """Spawn one hollow box at its selected rack or floor-staging pose."""
    instance_name = f"{box_name}_{index}"
    spec = RACK_BOX_SPAWN_PLAN[instance_name]
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Workcell/StagingBoxes/{instance_name}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(usd_path),
            scale=spec.scale,
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
            ),
        ),
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


def build_totes_cfg() -> RigidObjectCollectionCfg:
    objects: dict[str, RigidObjectCfg] = {}
    for depth_id in range(3):
        for lane_id, lane_name in enumerate(("front", "back")):
            name = f"rack_tier{depth_id}_{lane_name}"
            objects[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Workcell/LegacyTask/Totes/{name}",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=str(OPEN_TOTE_USD),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        max_depenetration_velocity=0.8,
                        linear_damping=0.06,
                        angular_damping=0.10,
                        solver_position_iteration_count=10,
                        solver_velocity_iteration_count=3,
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=task_tote_position(lane_id, depth_id),
                    rot=RACK_BOX_WORLD_ROT,
                ),
            )
    for foreign_id in range(3):
        name = f"foreign_{foreign_id}"
        objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Workcell/ConveyorSystem/ForeignTotes/{name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(OPEN_TOTE_USD),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.8,
                    linear_damping=0.06,
                    angular_damping=0.10,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.45 + 0.27 * foreign_id, 1.55, 0.015),
            ),
        )
    return RigidObjectCollectionCfg(rigid_objects=objects)


def build_cargo_cfg() -> RigidObjectCollectionCfg:
    objects: dict[str, RigidObjectCfg] = {}
    for tote_id in range(6):
        depth_id = tote_id // 2
        lane_id = tote_id % 2
        tote_position = task_tote_position(lane_id, depth_id)
        tote_rotation = RACK_BOX_WORLD_ROT
        for item_id in range(2):
            name = f"cargo_{tote_id}_{item_id}"
            local = (-0.045, -0.052, 0.042) if item_id == 0 else (0.045, 0.052, 0.042)
            offset = quat_rotate(tote_rotation, local)
            position = tuple(tote_position[axis] + offset[axis] for axis in range(3))
            common = dict(
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    max_depenetration_velocity=0.5,
                    linear_damping=0.08,
                    angular_damping=0.10,
                    solver_position_iteration_count=10,
                    solver_velocity_iteration_count=3,
                    sleep_threshold=0.002,
                    stabilization_threshold=0.001,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.12 if item_id == 0 else 0.18),
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.0025,
                    rest_offset=0.0,
                ),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.82,
                    dynamic_friction=0.68,
                    restitution=0.01,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.12 + 0.11 * tote_id, 0.35, 0.72 - 0.08 * item_id),
                    roughness=0.68,
                ),
            )
            spawn = (
                sim_utils.SphereCfg(radius=0.026, **common)
                if item_id == 0
                else sim_utils.CuboidCfg(size=(0.048, 0.040, 0.036), **common)
            )
            objects[name] = RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Workcell/LegacyTask/Cargo/{name}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=position,
                    rot=tote_rotation,
                ),
            )
    return RigidObjectCollectionCfg(rigid_objects=objects)


@configclass
class RobustWorkcellSceneCfg(InteractiveSceneCfg):
    factory = AssetBaseCfg(
        prim_path="/World/Factory",
        spawn=sim_utils.UsdFileCfg(usd_path=FACTORY_USD),
    )
    ground = static_cuboid(
        "/World/Ground",
        (30.0, 30.0, 0.10),
        (0.0, 0.0, -0.06),
        (0.11, 0.12, 0.13),
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
    legacy_cargo_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/LegacyTask/Cargo",
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
    obstacles_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/DynamicObstacles",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )
    staging_boxes_group: AssetBaseCfg = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/StagingBoxes",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORKCELL_GROUPS_USD)),
    )

    robot: ArticulationCfg = KUAVO_CFG
    # Isaac Lab's regex-based child spawners require this parent to exist.
    grippers_group: AssetBaseCfg | None = build_gripper_group_cfg(GRIPPER_SETTINGS)
    left_gripper: ArticulationCfg | None = build_gripper_articulation_cfg(
        GRIPPER_SETTINGS, "left"
    )
    right_gripper: ArticulationCfg | None = build_gripper_articulation_cfg(
        GRIPPER_SETTINGS, "right"
    )
    # Must be declared after both hands: its spawner aligns the mount frames
    # and authors the external fixed joints before physics initialization.
    gripper_attachments: AssetBaseCfg | None = build_gripper_attachment_cfg(
        GRIPPER_SETTINGS
    )
    button_station: ArticulationCfg = BUTTON_STATION_CFG

    # New local box assets (assets/{Small,Medium,Large,XLarge}Box.usd):
    # two of each spawned; rack_box_layout.py determines rack/floor placement.
    small_box_0: ArticulationCfg = staging_box_cfg("SmallBox", SMALL_BOX_USD, 0)
    small_box_1: ArticulationCfg = staging_box_cfg("SmallBox", SMALL_BOX_USD, 1)
    medium_box_0: ArticulationCfg = staging_box_cfg("MediumBox", MEDIUM_BOX_USD, 0)
    medium_box_1: ArticulationCfg = staging_box_cfg("MediumBox", MEDIUM_BOX_USD, 1)
    large_box_0: ArticulationCfg = staging_box_cfg("LargeBox", LARGE_BOX_USD, 0)
    large_box_1: ArticulationCfg = staging_box_cfg("LargeBox", LARGE_BOX_USD, 1)
    xlarge_box_0: ArticulationCfg = staging_box_cfg("XLargeBox", XLARGE_BOX_USD, 0)
    xlarge_box_1: ArticulationCfg = staging_box_cfg("XLargeBox", XLARGE_BOX_USD, 1)

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
    fence = static_cuboid(
        "{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/Panel",
        (1.55, 0.025, 1.55),
        layout_position("fence"),
        (0.025, 0.035, 0.045),
        rotation=layout_rotation("fence"),
        opacity=0.40,
        metallic=0.90,
    )

    conveyor = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/Visual",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CONVEYOR_USD,
            scale=(0.01, 0.01, 0.01),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=layout_position("conveyor"),
            rot=layout_rotation("conveyor"),
        ),
    )
    conveyor_surface = kinematic_cuboid(
        "{ENV_REGEX_NS}/Workcell/ConveyorSystem/Surface",
        (2.55, 0.68, 0.03),
        layout_offset("conveyor", (1.29, 0.43, 0.753)),
        (0.10, 0.12, 0.14),
        rotation=workcell_mdp.CONVEYOR_GEOMETRY_ROT,
        visible=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.9,
            dynamic_friction=0.8,
            restitution=0.0,
        ),
    )

    totes: RigidObjectCollectionCfg = build_totes_cfg()
    cargo: RigidObjectCollectionCfg = build_cargo_cfg()

    moving_human = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/DynamicObstacles/MovingHuman",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(SAFETY_WORKER_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.55, -1.03, 0.0)),
    )
    moving_robot = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/DynamicObstacles/MovingRobot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(MOBILE_ROBOT_USD),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                max_depenetration_velocity=0.5,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(2.02, 0.05, 0.0)),
    )

    robustness_camera = CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{ROBOT_MODEL.head_camera_body}/RobustnessCamera",
        update_period=1.0 / 30.0,
        height=120,
        width=160,
        data_types=["rgb", "distance_to_image_plane"],
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

    # Virtual waist view retained for the existing policy observation schema.
    waist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo/waist_yaw_link/WaistCamera",
        update_period=1.0 / 30.0,
        height=120,
        width=160,
        data_types=["rgb", "distance_to_image_plane"],
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
        update_period=1.0 / 30.0,
        height=120,
        width=160,
        data_types=["rgb", "distance_to_image_plane"],
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
        update_period=1.0 / 30.0,
        height=120,
        width=160,
        data_types=["rgb", "distance_to_image_plane"],
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


ROBOT_SAFETY_LINKS = [
    "waist_yaw_link",
    "zarm_l1_link",
    "zarm_l2_link",
    "zarm_l3_link",
    "zarm_l4_link",
    "zarm_l5_link",
    "zarm_l6_link",
    "zarm_l7_link",
    "zarm_r1_link",
    "zarm_r2_link",
    "zarm_r3_link",
    "zarm_r4_link",
    "zarm_r5_link",
    "zarm_r6_link",
    "zarm_r7_link",
]


@configclass
class ActionsCfg:
    upper_body = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(CONTROLLED_JOINT_NAMES),
        scale={
            "waist_yaw_joint": 1.0,
            "zarm_.*_joint": 0.45,
        },
        use_default_offset=True,
        preserve_order=True,
    )
    left_gripper = build_gripper_action_cfg(GRIPPER_SETTINGS, "left")
    right_gripper = build_gripper_action_cfg(GRIPPER_SETTINGS, "right")


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=base_mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(CONTROLLED_JOINT_NAMES),
                )
            },
            noise=Unoise(n_min=-0.004, n_max=0.004),
        )
        joint_vel = ObsTerm(
            func=base_mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=list(CONTROLLED_JOINT_NAMES),
                )
            },
            scale=0.1,
            noise=Unoise(n_min=-0.012, n_max=0.012),
        )
        left_gripper_joint_pos = (
            ObsTerm(
                func=base_mdp.joint_pos_rel,
                params={
                    "asset_cfg": SceneEntityCfg(
                        GRIPPER_SETTINGS.asset_name_for("left"),
                        joint_names=list(GRIPPER_SETTINGS.joint_names_for("left")),
                    )
                },
                noise=Unoise(n_min=-0.004, n_max=0.004),
            )
            if "left" in GRIPPER_SETTINGS.active_sides
            else None
        )
        right_gripper_joint_pos = (
            ObsTerm(
                func=base_mdp.joint_pos_rel,
                params={
                    "asset_cfg": SceneEntityCfg(
                        GRIPPER_SETTINGS.asset_name_for("right"),
                        joint_names=list(GRIPPER_SETTINGS.joint_names_for("right")),
                    )
                },
                noise=Unoise(n_min=-0.004, n_max=0.004),
            )
            if "right" in GRIPPER_SETTINGS.active_sides
            else None
        )
        tote_poses = ObsTerm(
            func=workcell_mdp.tote_poses,
            noise=Unoise(n_min=-0.003, n_max=0.003),
        )
        tote_motion = ObsTerm(
            func=workcell_mdp.tote_motion_obs,
            noise=Unoise(n_min=-0.006, n_max=0.006),
        )
        cargo_state = ObsTerm(
            func=workcell_mdp.cargo_state_obs,
            noise=Unoise(n_min=-0.002, n_max=0.002),
        )
        obstacle_state = ObsTerm(
            func=workcell_mdp.obstacle_state_obs,
            noise=Unoise(n_min=-0.008, n_max=0.008),
        )
        progress = ObsTerm(func=workcell_mdp.task_progress_obs, params=TASK_OBJECT_PARAMS)
        button_armed = ObsTerm(func=workcell_mdp.button_armed_obs, params=TASK_OBJECT_PARAMS)
        button_travel = ObsTerm(func=workcell_mdp.button_travel_obs)
        cargo_retained = ObsTerm(func=workcell_mdp.cargo_retained_obs)
        prefill_count = ObsTerm(func=workcell_mdp.prefill_count_obs)
        robustness_difficulty = ObsTerm(func=workcell_mdp.robustness_difficulty_obs)
        last_action = ObsTerm(func=base_mdp.last_action)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class VisionCfg(ObsGroup):
        rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robustness_camera"),
                "data_type": "rgb",
                "normalize": True,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )
        depth = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("robustness_camera"),
                "data_type": "distance_to_image_plane",
                "normalize": False,
            },
            noise=Unoise(n_min=-0.012, n_max=0.012),
            clip=(0.0, 8.0),
        )
        waist_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("waist_camera"),
                "data_type": "rgb",
                "normalize": True,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )
        left_wrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("left_wrist_camera"),
                "data_type": "rgb",
                "normalize": True,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )
        right_wrist_rgb = ObsTerm(
            func=base_mdp.image,
            params={
                "sensor_cfg": SceneEntityCfg("right_wrist_camera"),
                "data_type": "rgb",
                "normalize": True,
            },
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(0.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    vision: VisionCfg = VisionCfg()


@configclass
class EventsCfg:
    robot_material = EventTerm(
        func=base_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarm_.*"),
            "static_friction_range": (0.55, 1.15),
            "dynamic_friction_range": (0.40, 0.95),
            "restitution_range": (0.0, 0.04),
            "num_buckets": 48,
            "make_consistent": True,
        },
    )
    robot_arm_mass = EventTerm(
        func=base_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="zarm_.*"),
            "mass_distribution_params": (0.94, 1.06),
            "operation": "scale",
        },
    )
    actuator_gains = EventTerm(
        func=base_mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(CONTROLLED_JOINT_NAMES),
            ),
            "stiffness_distribution_params": (0.88, 1.12),
            "damping_distribution_params": (0.86, 1.14),
            "operation": "scale",
        },
    )
    left_gripper_material = (
        EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    GRIPPER_SETTINGS.asset_name_for("left"),
                    body_names=GRIPPER_SETTINGS.body_names_for("left"),
                ),
                "static_friction_range": (0.65, 1.35),
                "dynamic_friction_range": (0.50, 1.10),
                "restitution_range": (0.0, 0.03),
                "num_buckets": 48,
                "make_consistent": True,
            },
        )
        if "left" in GRIPPER_SETTINGS.active_sides
        else None
    )
    right_gripper_material = (
        EventTerm(
            func=base_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    GRIPPER_SETTINGS.asset_name_for("right"),
                    body_names=GRIPPER_SETTINGS.body_names_for("right"),
                ),
                "static_friction_range": (0.65, 1.35),
                "dynamic_friction_range": (0.50, 1.10),
                "restitution_range": (0.0, 0.03),
                "num_buckets": 48,
                "make_consistent": True,
            },
        )
        if "right" in GRIPPER_SETTINGS.active_sides
        else None
    )
    left_gripper_gains = (
        EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    GRIPPER_SETTINGS.asset_name_for("left"),
                    joint_names=list(GRIPPER_SETTINGS.joint_names_for("left")),
                ),
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.88, 1.12),
                "operation": "scale",
            },
        )
        if "left" in GRIPPER_SETTINGS.active_sides
        else None
    )
    right_gripper_gains = (
        EventTerm(
            func=base_mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg(
                    GRIPPER_SETTINGS.asset_name_for("right"),
                    joint_names=list(GRIPPER_SETTINGS.joint_names_for("right")),
                ),
                "stiffness_distribution_params": (0.90, 1.10),
                "damping_distribution_params": (0.88, 1.12),
                "operation": "scale",
            },
        )
        if "right" in GRIPPER_SETTINGS.active_sides
        else None
    )
    tote_physics = EventTerm(
        func=workcell_mdp.randomize_collection_physics,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("totes"),
            "mass_scale_range": (0.65, 1.45),
            "static_friction_range": (0.20, 0.52),
            "dynamic_friction_range": (0.15, 0.44),
            "restitution_range": (0.0, 0.035),
        },
    )
    cargo_physics = EventTerm(
        func=workcell_mdp.randomize_collection_physics,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("cargo"),
            "mass_scale_range": (0.55, 1.65),
            "static_friction_range": (0.55, 1.10),
            "dynamic_friction_range": (0.42, 0.92),
            "restitution_range": (0.0, 0.04),
        },
    )
    gravity = EventTerm(
        func=base_mdp.randomize_physics_scene_gravity,
        mode="startup",
        params={
            "gravity_distribution_params": (
                [-0.06, -0.06, -9.95],
                [0.06, 0.06, -9.67],
            ),
            "operation": "abs",
        },
    )

    reset_all = EventTerm(func=base_mdp.reset_scene_to_default, mode="reset")
    reset_flap_friction = EventTerm(
        func=workcell_mdp.randomize_box_flap_joint_friction,
        mode="reset",
        params={
            "asset_names": LOCAL_BOX_SCENE_KEYS,
            "static_friction_range": FLAP_STATIC_RESET_RANGE,
            "dynamic_friction_range": FLAP_DYNAMIC_RESET_RANGE,
        },
    )
    reset_workcell = EventTerm(
        func=workcell_mdp.reset_totes_and_cargo,
        mode="reset",
        params={
            "position_jitter": 0.012,
            "yaw_jitter": math.radians(3.0),
            "cargo_jitter": 0.008,
        },
    )
    reset_movers = EventTerm(
        func=workcell_mdp.reset_movers,
        mode="reset",
    )
    lighting = EventTerm(
        func=workcell_mdp.randomize_lighting,
        mode="reset",
    )
    move_movers = EventTerm(
        func=workcell_mdp.move_movers,
        mode="interval",
        interval_range_s=(0.05, 0.05),
        is_global_time=False,
    )
    cargo_disturbance = EventTerm(
        func=workcell_mdp.disturb_cargo,
        mode="interval",
        interval_range_s=(1.5, 3.0),
        is_global_time=False,
        params={"linear_velocity_range": 0.035},
    )


@configclass
class RewardsCfg:
    progress = RewTerm(
        func=workcell_mdp.task_progress,
        weight=4.0,
        params=TASK_OBJECT_PARAMS,
    )
    cargo_retention = RewTerm(func=workcell_mdp.cargo_retained_fraction, weight=2.0)
    tote_stability = RewTerm(func=workcell_mdp.tote_stability_reward, weight=1.5)
    obstacle_proximity = RewTerm(
        func=workcell_mdp.obstacle_proximity_penalty,
        weight=-2.5,
        params={"robot_cfg": SceneEntityCfg("robot", body_names=ROBOT_SAFETY_LINKS)},
    )
    action_rate = RewTerm(func=base_mdp.action_rate_l2, weight=-0.012)
    joint_velocity = RewTerm(
        func=base_mdp.joint_vel_l2,
        weight=-2.0e-5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=list(CONTROLLED_JOINT_NAMES),
            )
        },
    )
    success = RewTerm(
        func=workcell_mdp.task_success_reward,
        weight=120.0,
        params=TASK_OBJECT_PARAMS,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    cargo_spill = DoneTerm(func=workcell_mdp.cargo_spilled)
    tote_drop = DoneTerm(
        func=workcell_mdp.task_object_dropped,
        params=TASK_OBJECT_PARAMS,
    )
    human_or_robot_contact = DoneTerm(
        func=workcell_mdp.obstacle_collision,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=ROBOT_SAFETY_LINKS),
            "collision_margin": 0.025,
        },
    )
    success = DoneTerm(func=workcell_mdp.task_success, params=TASK_OBJECT_PARAMS)


@configclass
class CurriculumCfg:
    robustness = CurrTerm(
        func=workcell_mdp.robustness_curriculum,
        params={"ramp_steps": 1_500_000},
    )


@configclass
class KuavoRobustWorkcellEnvCfg(ManagerBasedRLEnvCfg):
    scene: RobustWorkcellSceneCfg = RobustWorkcellSceneCfg(
        num_envs=4,
        env_spacing=5.0,
        replicate_physics=True,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.15,
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**20,
        ),
    )

    def __post_init__(self) -> None:
        self.decimation = 4
        self.episode_length_s = 24.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.45, -3.35, 2.45)
        self.viewer.lookat = (0.85, 0.0, 0.85)
        self.rerender_on_reset = True
