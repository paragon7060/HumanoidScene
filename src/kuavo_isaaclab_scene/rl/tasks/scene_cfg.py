"""Reuse the measured workcell; replace legacy cargo with cargo in real task boxes."""

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import ContactSensorCfg

from ...envs.manager_env import LOCAL_BOX_SCENE_KEYS, RACK_BOX_SPAWN_PLAN, RobustWorkcellSceneCfg
from ...robots.robot_model import resolve_robot_model
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.robot_inertials import spawn_teleop_robot, spawn_s56_twofinger_robot
from .asset_geometry import BoxGeometry, box_geometry
from .flap_spawn import spawn_locked_flap_box


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
        geometry[name] = box_geometry(cfg, spec.grasp_flaps if spec.grasp_mode == "flap_top" else ())
        if spec.grasp_mode == "flap_top":
            cfg.spawn.func = spawn_locked_flap_box
            cfg.spawn.flap_lock_degrees = spec.flap_lock_degrees
            cfg.actuators["flaps"].stiffness = 2.0
            cfg.actuators["flaps"].damping = 0.2
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
        if spec.grasp_mode == "flap_top":
            flap = spec.grasp_flaps[index // 2]
            targets = [getattr(scene, name).prim_path + "/" + geometry[name].flaps[flap].body_path
                       for name in spec.box_names]
        setattr(scene, f"grasp_contact_{index}", ContactSensorCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{body}", update_period=0.0, history_length=1,
            track_contact_points=spec.grasp_mode == "flap_top", max_contact_data_count_per_prim=32,
            filter_prim_paths_expr=targets))
    collision_bodies = "waist_yaw_link|zarm_[lr][1-6]_link"
    if spec.grasp_mode == "flap_top":
        collision_bodies = "waist_yaw_link|zarm_[lr][1-7]_link|[lr]_twofinger_base"
    scene.robot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Kuavo/(" + collision_bodies + ")",
        update_period=0.0, history_length=1)
    return scene, geometry
