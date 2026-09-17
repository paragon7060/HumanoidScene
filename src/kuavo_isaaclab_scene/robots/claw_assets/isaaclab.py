"""Independent-hand ArticulationCfg; import only after AppLauncher starts Kit."""

from .package import CLAW_ASSET_DIR, load_claw_asset, load_claw_config
from isaaclab.sim.utils import clone


def author_integrated_claw_contact(root, finger_contact):
    """Apply the package contact model to claws embedded in a host robot USD."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import spawn_rigid_body_material, RigidBodyMaterialCfg
    from .usd import author_claw_distal_pads, disable_reference_colliders
    contact_cfg = load_claw_config()["contact"]
    housing_static = float(contact_cfg["housing_static_friction"])
    housing_dynamic = float(contact_cfg["housing_dynamic_friction"])
    contact_offset = float(contact_cfg["contact_offset_m"])
    rest_offset = float(contact_cfg["rest_offset_m"])

    finger_links = {f"{side}_{jaw}_finger" for side in "lr" for jaw in "fb"}
    hardware = finger_links | {f"{side}_twofinger_base" for side in "lr"}
    wrist_links = {"zarm_l7_link", "zarm_r7_link"}
    hardware |= wrist_links
    material_path = str(root.GetPath()) + "/HandContactMaterial"
    spawn_rigid_body_material(material_path, RigidBodyMaterialCfg(
        static_friction=housing_static, dynamic_friction=housing_dynamic, restitution=0.0))
    finger_material_path = str(root.GetPath()) + "/FingerContactMaterial"
    spawn_rigid_body_material(finger_material_path, RigidBodyMaterialCfg(
        static_friction=finger_contact.static_friction,
        dynamic_friction=finger_contact.dynamic_friction,
        friction_combine_mode=finger_contact.friction_combine_mode,
        restitution=0.0,
    ))
    helper_colliders = disable_reference_colliders(root)
    for link in root.GetChildren():
        if link.GetName() not in hardware:
            continue
        for prim in list(Usd.PrimRange(link)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        if link.GetName() in wrist_links:
            for prim in Usd.PrimRange(link):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        PhysxSchema.PhysxRigidBodyAPI.Apply(link).CreateEnableSpeculativeCCDAttr(True)
    counts = {}
    for link in root.GetChildren():
        if link.GetName() not in hardware:
            continue
        count = 0
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh) or "/visuals/" not in str(prim.GetPath()):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(contact_offset)
            contact.CreateRestOffsetAttr(rest_offset)
            bind_physics_material(str(prim.GetPath()),
                                  finger_material_path if link.GetName() in finger_links else material_path)
            count += 1
        counts[link.GetName()] = count
    if set(counts) != hardware or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Missing hand collision meshes: {hardware - counts.keys()}, counts={counts}")
    pad_count = author_claw_distal_pads(
        root.GetStage(), root, CLAW_ASSET_DIR / "config.json", sides="lr",
        finger_material_path=finger_material_path,
    )
    print(f"[CONTACT] Added {sum(counts.values())} hand collision meshes; separate jaws, "
          f"{pad_count} flat distal pads; "
          f"disabled {helper_colliders} tool-frame colliders; "
          f"{contact_offset * 1_000:g}mm contact offset, speculative CCD; finger friction "
          f"{finger_contact.static_friction}/{finger_contact.dynamic_friction} "
          f"({finger_contact.friction_combine_mode}); housing/wrist friction "
          f"{housing_static}/{housing_dynamic}.", flush=True)


@clone
def spawn_claw(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Use the same authoritative claw physics in standalone and robot scenes."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from .usd import author_claw_contact
    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    side = "left" if root.GetChild("l_twofinger_base") else "right"
    author_claw_contact(root.GetStage(), CLAW_ASSET_DIR / "config.json", side=side, root=root)
    return root


def make_claw_cfg(side: str, prim_path: str, *, fix_base: bool = True,
                  pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg
    from isaaclab.actuators import ImplicitActuatorCfg

    asset = load_claw_asset(side)
    actuator = asset.metadata["actuator"]
    if not asset.usd_path.is_file():
        raise FileNotFoundError(f"Build the claw USDs first: {asset.usd_path}")
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            func=spawn_claw,
            usd_path=str(asset.usd_path),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                fix_root_link=fix_base, enabled_self_collisions=False,
                solver_position_iteration_count=16, solver_velocity_iteration_count=4),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=pos, rot=rot, joint_pos=asset.initial_positions(), joint_vel={".*": 0.0}),
        actuators={
            "drivers": ImplicitActuatorCfg(joint_names_expr=list(asset.motor_names), **actuator),
            "passive": ImplicitActuatorCfg(
                joint_names_expr=[f"{asset.prefix}_[fb]_bar_[34]_joint"],
                effort_limit_sim=5.0, stiffness=0.0, damping=0.0, friction=0.0),
        },
        soft_joint_pos_limit_factor=1.0,
    )
