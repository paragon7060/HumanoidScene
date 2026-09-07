"""Local task geometry only: no warehouse USD, decorative conveyor or movers."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from ...core.paths import ASSET_DIR
from ...workcell.workcell_layout import position, rotation, scale, offset, remap_quat


def group(path, *, pos=(0., 0., 0.), rot=(1., 0., 0., 0.), scaling=(1., 1., 1.)):
    return AssetBaseCfg(prim_path=path,
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "workcell_groups.usda"), scale=scaling),
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot))


def add_workcell(scene, parallel):
    """Insertion order matters: create parent Xforms before regex child assets."""
    scene.ground = AssetBaseCfg(prim_path="/World/Ground", collision_group=-1,
        spawn=sim_utils.CuboidCfg(size=(parallel.ground_extent, parallel.ground_extent, .10),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.11, .12, .13))),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.06)))
    for key, suffix in (
        ("workcell_groups", ""), ("racks_group", "/Racks"),
        ("safety_group", "/SafetySystem"), ("fence_group", "/SafetySystem/Fence"),
        ("conveyor_group", "/ConveyorSystem"),
        ("foreign_totes_group", "/ConveyorSystem/ForeignTotes"),
        ("contents_group", "/Contents"), ("staging_boxes_group", "/StagingBoxes"),
    ):
        setattr(scene, key, group("{ENV_REGEX_NS}/Workcell" + suffix))
    # scene['rack'] stays the measured Rack anchor; Visual has identity local pose.
    scene.rack = group("{ENV_REGEX_NS}/Workcell/Racks/Rack",
        pos=position("rack"), rot=rotation("rack"), scaling=scale("rack"))
    scene.rack_visual = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Workcell/Racks/Rack/Visual",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "Rack.usd")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., 0.), rot=(1., 0., 0., 0.)))
    scene.fence = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/Fence/Panel",
        spawn=sim_utils.CuboidCfg(size=(1.55, .025, 1.55),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.025, .035, .045), opacity=.4)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=position("fence"), rot=rotation("fence")))
    scene.button_station = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/SafetySystem/ButtonStation",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSET_DIR / "button_station.usda"),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                solver_position_iteration_count=8, solver_velocity_iteration_count=2)),
        init_state=ArticulationCfg.InitialStateCfg(pos=position("button_station"),
            rot=rotation("button_station"), joint_pos={"ButtonJoint": 0.}, joint_vel={"ButtonJoint": 0.}),
        actuators={"return_spring": ImplicitActuatorCfg(joint_names_expr=["ButtonJoint"],
            effort_limit_sim=120., velocity_limit_sim=1., stiffness=420., damping=18.)})
    # Keep the task's calibrated physical surface; do not load the network visual USD.
    scene.conveyor_surface = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Workcell/ConveyorSystem/Surface",
        spawn=sim_utils.CuboidCfg(size=(2.55, .68, .03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.004, rest_offset=0.),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.9, dynamic_friction=.8, restitution=0.),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.10, .12, .14))),
        init_state=RigidObjectCfg.InitialStateCfg(pos=offset("conveyor", (1.29, .43, .753)),
            rot=remap_quat("conveyor", (1., 0., 0., 0.))))


def add_light(scene):
    scene.dome_light = AssetBaseCfg(prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=1800., color=(.85, .85, .85)))
