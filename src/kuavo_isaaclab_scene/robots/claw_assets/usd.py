"""Offline contact finalization of the independent claw (no Kit startup)."""

import json
import math


def author_claw_inertials(root, metadata_path, *, side):
    """Apply the package's authoritative masses/CoMs/inertias at spawn/build.

    Use absolute values, so repeated finalization never compounds scaling.
    Only claw links are touched, including when composed into a host robot.
    """
    from pxr import Gf, UsdPhysics

    config = json.loads(metadata_path.read_text())
    estimates = json.loads((metadata_path.parent / "inertial_estimates.json").read_text())["links"]
    names = config["sides"][side]["link_names"]
    total = sum(estimates[name]["mass_kg"] for name in names)
    if not math.isclose(total, config["sides"][side]["total_mass_kg"], abs_tol=1e-9):
        raise ValueError("Claw inertial estimates and configured total mass disagree")
    for name in names:
        link = root.GetChild(name)
        if not link or not link.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Missing claw rigid body for mass configuration: {name}")
        values = estimates[name]
        api = UsdPhysics.MassAPI.Apply(link)
        api.CreateMassAttr(values["mass_kg"])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*values["com_m"]))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values["diagonal_inertia_kg_m2"]))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))


def author_claw_contact(stage, metadata_path, *, side, finger_contact=None, root=None):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from ..twofinger_linkage import require_closed_linkages

    root = root if root is not None else stage.GetDefaultPrim()
    require_closed_linkages(root, sides=side[0])
    author_claw_inertials(root, metadata_path, side=side)
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
