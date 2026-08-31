"""Contact geometry for S200062 hands and thin, articulated cardboard boxes."""


def add_hand_colliders(root):
    """Use individual visual meshes as convex colliders, never a hull of both jaws."""
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema
    from isaaclab.sim import bind_physics_material
    from isaaclab.sim.spawners.materials import spawn_rigid_body_material, RigidBodyMaterialCfg

    hardware = {f"{s}_{part}" for s in "lr" for part in (
        "twofinger_base", "f_finger", "b_finger",
    )}
    wrist_links = {"zarm_l7_link", "zarm_r7_link"}
    hardware |= wrist_links
    # The imported four-bar mechanism is a tree: passive decorative bars do
    # not follow a closed linkage. Giving those bars colliders blocks the jaw
    # opening and kicks thin flaps away. Contact belongs to the driven finger
    # bodies and housing; keep the passive bar visuals unchanged.
    material_path = str(root.GetPath()) + "/HandContactMaterial"
    spawn_rigid_body_material(material_path, RigidBodyMaterialCfg(
        static_friction=1.0, dynamic_friction=.8, restitution=0.0))
    for prim in list(Usd.PrimRange(root)):
        if "_end_effector" in str(prim.GetPath()) and prim.IsInstance():
            prim.SetInstanceable(False)
    helper_colliders = 0
    for prim in Usd.PrimRange(root):
        if "_end_effector" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.CollisionAPI):
            # A dummy sphere at the tool reference can occupy the jaw gap.
            # Coordinate frames are not physical gripper hardware.
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            helper_colliders += 1
    for link in root.GetChildren():
        if link.GetName() not in hardware:
            continue
        # Imported visuals are instanced; overrides must be authored on their
        # local prims rather than on shared instance proxies.
        for prim in list(Usd.PrimRange(link)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        if link.GetName() in wrist_links:
            # Source wrist cylinders extend 140mm below the joint, versus
            # 67mm for the real wrist mesh: the excess occupies the open jaw.
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
            contact.CreateContactOffsetAttr(.002)
            contact.CreateRestOffsetAttr(0.)
            bind_physics_material(str(prim.GetPath()), material_path)
            count += 1
        counts[link.GetName()] = count
    if set(counts) != hardware or any(count == 0 for count in counts.values()):
        raise RuntimeError(f"Missing hand collision meshes: {hardware - counts.keys()}, counts={counts}")
    print(f"[CONTACT] Added {sum(counts.values())} hand collision meshes; separate jaws, "
          f"disabled {helper_colliders} tool-frame colliders; "
          "2mm contact offset, speculative CCD, friction 1.0/0.8.", flush=True)


def spawn_contact_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd, UsdPhysics, PhysxSchema

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            body.CreateEnableSpeculativeCCDAttr(True)
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            contact = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            contact.CreateContactOffsetAttr(.002)
            contact.CreateRestOffsetAttr(0.)
    return root
