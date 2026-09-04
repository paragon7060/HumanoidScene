"""Reuse the measured workcell; replace legacy cargo with cargo in real task boxes."""

from dataclasses import dataclass

from pxr import Usd, UsdGeom, UsdPhysics
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

from ...envs.manager_env import LOCAL_BOX_SCENE_KEYS, RACK_BOX_SPAWN_PLAN, RobustWorkcellSceneCfg
from ...robots.robot_model import resolve_robot_model
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.robot_inertials import spawn_teleop_robot, spawn_s56_twofinger_robot


@dataclass(frozen=True)
class BoxGeometry:
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    body_path: str


def box_geometry(asset_cfg) -> BoxGeometry:
    """Measure only the rigid Body, excluding movable flaps, in its own frame."""
    stage = Usd.Stage.Open(asset_cfg.spawn.usd_path)
    if stage is None:
        raise ValueError(f"Unable to open box USD: {asset_cfg.spawn.usd_path}")
    bodies = [p for p in stage.Traverse() if p.GetName() == "Body" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bodies) != 1:
        raise ValueError("Box asset must have one rigid Body; adapt box_geometry() for a different USD.")
    body = bodies[0]
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeUntransformedBound(body).ComputeAlignedRange()
    scale = asset_cfg.spawn.scale
    low = tuple(float(bounds.GetMin()[i]) * scale[i] for i in range(3))
    high = tuple(float(bounds.GetMax()[i]) * scale[i] for i in range(3))
    half = tuple((high[i] - low[i]) / 2 for i in range(3))
    if min(half) <= 0:
        raise ValueError("Invalid box Body bounds.")
    return BoxGeometry(tuple((high[i] + low[i]) / 2 for i in range(3)), half,
                       str(body.GetPath().MakeRelativePath(stage.GetDefaultPrim().GetPath())))


def build_scene(spec, num_envs=8, env_spacing=8.0, cameras=False):
    model, hand = resolve_robot_model(), resolve_gripper_settings()
    if hand.name not in ("s200062_integrated", "s56_twofinger"):
        raise ValueError("Initial RL contact preset supports s200062_integrated/s56_twofinger. "
                         "Configure finger/tool sensors and actions before using another gripper.")
    scene = RobustWorkcellSceneCfg(num_envs=num_envs, env_spacing=env_spacing, replicate_physics=True)
    ground_extent = max(30.0, 2 * env_spacing * num_envs ** 0.5)
    scene.ground.spawn.size = (ground_extent, ground_extent, 0.10)
    # No factory or RTX sensors for the default state-based PPO training path.
    scene.factory = None
    if not cameras:
        for name in ("robustness_camera", "waist_camera", "left_wrist_camera", "right_wrist_camera"):
            setattr(scene, name, None)
    scene.totes = None
    scene.cargo = None
    scene.moving_robot = None
    # The fixed-root planar controller uses the same wheel-contact convention as teleop.
    scene.robot.spawn.func = spawn_teleop_robot if model.has_wheel_base else spawn_s56_twofinger_robot
    scene.robot.spawn.activate_contact_sensors = True
    # Preserve the imported model's self-collision setting; enabling every
    # closed-linkage pair needs a separate collision-filter calibration.
    geometry = {}
    plans = {p.scene_key: p for p in RACK_BOX_SPAWN_PLAN.values()}
    for name in LOCAL_BOX_SCENE_KEYS:
        if name not in spec.box_names:
            setattr(scene, name, None)
    for name in spec.box_names:
        if name not in plans or not plans[name].on_rack:
            raise ValueError(f"{name} is not a captured/configured rack box. Set --rack-boxes or --rack-box-poses first.")
        cfg = getattr(scene, name)
        geometry[name] = box_geometry(cfg)
        if min(geometry[name].half_size) <= 2 * spec.cargo_radius:
            raise ValueError(f"Cargo radius is too large for {name}.")
        for item in range(spec.cargo_per_box):
            setattr(scene, f"cargo_{name}_{item}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Workcell/LegacyTask/Cargo/{name}_{item}",
                spawn=sim_utils.SphereCfg(radius=spec.cargo_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.002, rest_offset=0.0),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.8, dynamic_friction=0.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.4, 0.8))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 3.0 + item * 0.1))))
    # Independent boxes model spaces already occupied by another worker.
    for index in range(spec.prefill_count):
        setattr(scene, f"prefill_{index}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Workcell/ConveyorSystem/ForeignTotes/Prefill{index}",
            spawn=sim_utils.CuboidCfg(size=(0.30, 0.24, 0.18),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.45, 0.30, 0.14))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0, 0, 3.0))))
    targets = [getattr(scene, name).prim_path +
               ("/" + geometry[name].body_path if geometry[name].body_path != "." else "")
               for name in spec.box_names]
    for index, body in enumerate(spec.finger_bodies):
        setattr(scene, f"grasp_contact_{index}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{body}", update_period=0.0, history_length=1,
            filter_prim_paths_expr=targets))
    scene.robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo/(waist_yaw_link|zarm_[lr][1-6]_link)",
        update_period=0.0, history_length=1)
    return scene, geometry
