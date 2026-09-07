"""Only selected task boxes and explicitly requested cargo/occupied slots."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from ...core.paths import BOX_ATLAS_ASSETS
from ...envs.scene_physics import build_contact_box_spawn, build_box_flap_actuator
from ...workcell.box_flap_friction import resolve_flap_friction_settings
from .layout import box_spawn_plan
from .asset_geometry import box_geometry
from .flap_spawn import spawn_locked_flap_box


def add_boxes(scene, spec):
    plans = {p.scene_key: p for p in box_spawn_plan().values()}
    friction = resolve_flap_friction_settings(randomize_default=False)
    geometry = {}
    for name in spec.box_names:
        if name not in plans or not plans[name].on_rack:
            raise ValueError(f"{name} is not a captured/configured rack box. Set --rack-box-poses or --rack-boxes.")
        plan = plans[name]
        cfg = ArticulationCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Workcell/StagingBoxes/{plan.instance_name}",
            spawn=build_contact_box_spawn(BOX_ATLAS_ASSETS[plan.box_type], plan.scale),
            init_state=ArticulationCfg.InitialStateCfg(pos=plan.position, rot=plan.rotation,
                joint_pos={"joint_(front|back|left|right)": 0.}, joint_vel={".*": 0.}),
            actuators={"flaps": build_box_flap_actuator(friction)})
        geometry[name] = box_geometry(cfg, spec.grasp_flaps if spec.grasp_mode == "flap_top" else ())
        if spec.grasp_mode == "flap_top":
            cfg.spawn.func = spawn_locked_flap_box
            cfg.spawn.flap_lock_degrees = spec.flap_lock_degrees
            cfg.actuators["flaps"].stiffness = 2.
            cfg.actuators["flaps"].damping = .2
        setattr(scene, name, cfg)
        if spec.cargo_per_box and min(geometry[name].half_size) <= 2 * spec.cargo_radius:
            raise ValueError(f"Cargo radius is too large for {name}.")
        for item in range(spec.cargo_per_box):
            setattr(scene, f"cargo_{name}_{item}", RigidObjectCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Workcell/Contents/{name}_{item}",
                spawn=sim_utils.SphereCfg(radius=spec.cargo_radius,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=.002, rest_offset=0.),
                    physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8, dynamic_friction=.6),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.15, .4, .8))),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 3. + item * .1))))
    for index in range(spec.prefill_count):
        setattr(scene, f"prefill_{index}", RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Workcell/ConveyorSystem/ForeignTotes/Prefill{index}",
            spawn=sim_utils.CuboidCfg(size=(.30, .24, .18),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(), mass_props=sim_utils.MassPropertiesCfg(mass=.3),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.45, .30, .14))),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0., 0., 3.))))
    return geometry
