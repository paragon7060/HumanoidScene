"""Offline contact finalization of the independent claw (no Kit startup)."""

import json


def author_claw_contact(stage, metadata_path, *, side, finger_contact=None, root=None):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from ..twofinger_linkage import require_closed_linkages

    root = root if root is not None else stage.GetDefaultPrim()
    require_closed_linkages(root, sides=side[0])
    config = json.loads(metadata_path.read_text())
    contact = config["contact"]
    if finger_contact is not None:
        contact.update(finger_static_friction=finger_contact.static_friction,
                       finger_dynamic_friction=finger_contact.dynamic_friction,
                       friction_combine_mode=finger_contact.friction_combine_mode)
    materials = {}
    for name, role in (("FingerContactMaterial", "finger"), ("HandContactMaterial", "housing")):
        path = f"{root.GetPath()}/{name}"
        material = UsdShade.Material.Define(stage, path)
        api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(contact[f"{role}_static_friction"])
        api.CreateDynamicFrictionAttr(contact[f"{role}_dynamic_friction"])
        api.CreateRestitutionAttr(0.0)
        material.GetPrim().AddAppliedSchema("PhysxMaterialAPI")
        material.GetPrim().CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
            contact["friction_combine_mode"])
        materials[role] = material
    housing = config["sides"][side]["root_link"]
    fingers = {f"{side[0]}_{jaw}_finger" for jaw in "fb"}
    counts = {}
    for link in root.GetChildren():
        if link.GetName() not in fingers | {housing}:
            continue
        # Local overrides must not be written through shared instance proxies.
        for prim in list(Usd.PrimRange(link)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        link.AddAppliedSchema("PhysxRigidBodyAPI")
        link.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        colliders = 0
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh) or "/collisions/" not in str(prim.GetPath()):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(contact["contact_offset_m"])
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(contact["rest_offset_m"])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials["finger" if link.GetName() in fingers else "housing"], materialPurpose="physics")
            colliders += 1
        counts[link.GetName()] = colliders
    if set(counts) != fingers | {housing} or any(value == 0 for value in counts.values()):
        raise RuntimeError(f"Independent claw contact meshes missing: {counts}")
    root.SetCustomDataByKey("kuavo:clawPackageVersion", config["schema_version"])
