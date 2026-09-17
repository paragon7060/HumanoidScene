"""Offline contact finalization of the independent claw (no Kit startup)."""

import json
import math


def disable_reference_colliders(root):
    """Disable visual-only TCP marker collisions below a host robot root."""
    from pxr import Usd, UsdPhysics

    # URDF-imported collision groups are commonly instanceable. Expand only
    # the marker groups so their collision prims can receive local overrides.
    for prim in list(Usd.PrimRange(root)):
        if "_end_effector" in str(prim.GetPath()) and prim.IsInstance():
            prim.SetInstanceable(False)
    disabled = 0
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if "_end_effector" not in path or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        disabled += 1
    return disabled


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


def author_claw_distal_pads(stage, root, metadata_path, *, sides, finger_material_path):
    """Author the shared flat fingertip contact pads on integrated or standalone claws."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    config = json.loads(metadata_path.read_text())
    contact = config["contact"]
    pad_cfg = contact["distal_pad"]
    size = tuple(float(value) for value in pad_cfg["size_m"])
    if len(size) != 3 or any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError("Claw distal pad size must contain three positive finite values")
    centers = pad_cfg["center_m"]
    if set(centers) != {"f", "b"}:
        raise ValueError("Claw distal pad requires front/back centers")
    material = UsdShade.Material.Get(stage, finger_material_path)
    if not material:
        raise RuntimeError(f"Missing finger physics material: {finger_material_path}")
    count = 0
    for letter in sides:
        for jaw in "fb":
            link = root.GetChild(f"{letter}_{jaw}_finger")
            if not link:
                raise RuntimeError(f"Missing claw finger link: {letter}_{jaw}_finger")
            center = tuple(float(value) for value in centers[jaw])
            if len(center) != 3 or any(not math.isfinite(value) for value in center):
                raise ValueError(f"Invalid distal pad center for jaw {jaw}")
            pad = UsdGeom.Cube.Define(stage, link.GetPath().AppendChild("distal_contact_pad"))
            pad.CreateSizeAttr(1.0)
            pad.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
            xform = UsdGeom.Xformable(pad)
            ops = {op.GetOpType(): op for op in xform.GetOrderedXformOps()}
            translate = ops.get(UsdGeom.XformOp.TypeTranslate) or xform.AddTranslateOp()
            scale = ops.get(UsdGeom.XformOp.TypeScale) or xform.AddScaleOp()
            translate.Set(Gf.Vec3d(*center))
            scale.Set(Gf.Vec3f(*size))
            xform.SetXformOpOrder([translate, scale])
            prim = pad.GetPrim()
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
                contact["contact_offset_m"])
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
                contact["rest_offset_m"])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, materialPurpose="physics")
            prim.SetCustomDataByKey("kuavo:distalPadLengthM", size[2])
            count += 1
    return count


def author_claw_contact(stage, metadata_path, *, side, finger_contact=None, root=None):
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    from .linkage import require_closed_linkages

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
    # The source finger is a curved, concave CAD part.  Keep its convex hull as
    # a protective body collider, but make the distal 20 mm grasping surface a
    # dedicated flat pad which protrudes slightly and therefore contacts first.
    author_claw_distal_pads(
        stage, root, metadata_path, sides=side[0],
        finger_material_path=materials["finger"].GetPath(),
    )
    disable_reference_colliders(root)
    root.SetCustomDataByKey("kuavo:clawPackageVersion", config["schema_version"])
