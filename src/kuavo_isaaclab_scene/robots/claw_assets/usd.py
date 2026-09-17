"""Offline contact finalization of the independent claw (no Kit startup)."""

from copy import deepcopy
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


def pad_compliance(contact):
    """Validated implicit contact spring of the fingertip pad, or None if rigid.

    The stiffness is the pad's own compression spring, `E * area / thickness`,
    so the configured Young's modulus stays the single physical source. PhysX
    applies it per contact point, so a flat multi-point patch is stiffer than
    this value by the number of points it generates.

    Returning None selects the earlier hard pad, which shares the finger
    material and resolves contact as a rigid surface.
    """
    pad = contact["distal_pad"]
    compliance = pad["compliance"]
    if not compliance.get("enabled", True):
        return None
    size = tuple(float(value) for value in pad["size_m"])
    stiffness = float(compliance["stiffness_n_per_m"])
    damping = float(compliance["damping_n_s_per_m"])
    modulus = float(compliance["youngs_modulus_pa"])
    if not math.isfinite(stiffness) or stiffness <= 0:
        raise ValueError("Soft pad stiffness must be positive and finite")
    if not math.isfinite(damping) or damping < 0:
        raise ValueError("Soft pad damping must be finite and nonnegative")
    if not math.isfinite(modulus) or modulus <= 0:
        raise ValueError("Soft pad Young's modulus must be positive and finite")
    expected = modulus * size[1] * size[2] / size[0]
    if not math.isclose(stiffness, expected, rel_tol=1e-6):
        raise ValueError(f"Soft pad stiffness {stiffness} N/m disagrees with E*A/t {expected} N/m")
    return stiffness, damping


def resolve_finger_contact(config, finger_contact=None):
    """Effective jaw contact values: package defaults under an optional host override.

    Every caller resolves friction here, so an integrated robot and a standalone
    claw cannot drift apart on the values the fingertip pads are built from.
    """
    contact = dict(config["contact"])
    if finger_contact is not None:
        contact.update(finger_static_friction=finger_contact.static_friction,
                       finger_dynamic_friction=finger_contact.dynamic_friction,
                       friction_combine_mode=finger_contact.friction_combine_mode)
        if not getattr(finger_contact, "soft_pad", True):
            pad = deepcopy(contact["distal_pad"])
            pad["compliance"] = {**pad["compliance"], "enabled": False}
            contact["distal_pad"] = pad
    return contact


def _author_soft_pad_material(stage, path, contact):
    """Author the fingertip pad material: jaw friction plus a compliant contact."""
    from pxr import Sdf, UsdPhysics, UsdShade

    stiffness, damping = pad_compliance(contact)
    material = UsdShade.Material.Define(stage, str(path))
    prim = material.GetPrim()
    api = UsdPhysics.MaterialAPI.Apply(prim)
    api.CreateStaticFrictionAttr(contact["finger_static_friction"])
    api.CreateDynamicFrictionAttr(contact["finger_dynamic_friction"])
    api.CreateRestitutionAttr(0.0)
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
        contact["friction_combine_mode"])
    # A rigid pad against a rigid box flap resolves contact as an on/off
    # constraint, so the normal load collapses whenever the flap shifts. The
    # implicit spring keeps that load continuous while the pad deforms.
    prim.CreateAttribute("physxMaterial:compliantContactStiffness", Sdf.ValueTypeNames.Float).Set(
        stiffness)
    prim.CreateAttribute("physxMaterial:compliantContactDamping", Sdf.ValueTypeNames.Float).Set(
        damping)
    return material


def author_claw_distal_pads(root, contact, *, sides):
    """Author the fingertip pads and the material they share.

    This is the one place the pads exist. Standalone claws and claws composed
    into a host robot both land here, so the grasping surface never has to be
    matched up by hand in a second code path.

    The pad is compliant by default. When compliance is disabled the same slab
    is authored as the earlier rigid pad: finger material, no contact spring,
    and invisible because it then only doubles the hull's grasping face.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

    stage = root.GetStage()
    pad_cfg = contact["distal_pad"]
    size = tuple(float(value) for value in pad_cfg["size_m"])
    if len(size) != 3 or any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError("Claw distal pad size must contain three positive finite values")
    centers = pad_cfg["center_m"]
    if set(centers) != {"f", "b"}:
        raise ValueError("Claw distal pad requires front/back centers")
    compliance = pad_compliance(contact)
    if compliance is None:
        material = UsdShade.Material.Get(stage, f"{root.GetPath()}/FingerContactMaterial")
        if not material:
            raise RuntimeError("Rigid distal pads need the jaw materials authored first")
    else:
        material = _author_soft_pad_material(
            stage, f"{root.GetPath()}/SoftPadContactMaterial", contact)
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
            pad.CreateVisibilityAttr(UsdGeom.Tokens.invisible if compliance is None
                                     else UsdGeom.Tokens.inherited)
            pad.CreateDisplayColorAttr([Gf.Vec3f(0.09, 0.09, 0.10)])
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
            prim.SetCustomDataByKey("kuavo:distalPadThicknessM", size[0])
            if compliance is not None:
                prim.SetCustomDataByKey("kuavo:distalPadStiffnessNPerM", compliance[0])
                prim.SetCustomDataByKey("kuavo:distalPadDampingNSPerM", compliance[1])
            else:
                # A baked soft pad must not leave a spring recorded on a rigid one.
                for key in ("kuavo:distalPadStiffnessNPerM", "kuavo:distalPadDampingNSPerM"):
                    prim.ClearCustomDataByKey(key)
            count += 1
    return count, compliance


def author_claw_jaw_contact(root, contact, *, links, finger_links, mesh_scope,
                            replace_colliders=()):
    """Turn claw hardware meshes into convex colliders bound to the jaw materials.

    `mesh_scope` differs by asset, not by intent. The packaged claw and the S63
    composition carry real `collisions` meshes, while the S200062/S56 donors ship
    an empty `collisions` scope, so their `visuals` meshes are the only geometry
    the package can collide with. Everything else here is shared, so a host robot
    and a standalone claw cannot end up with different jaw physics.

    `replace_colliders` names links whose shipped collider is a crude URDF
    primitive rather than hand geometry. Their existing colliders are switched
    off first, so the scoped mesh hull authored here replaces them instead of
    colliding alongside them.
    """
    from pxr import Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

    stage = root.GetStage()
    replace_colliders = frozenset(replace_colliders)
    materials = {}
    for name, role in (("FingerContactMaterial", "finger"), ("HandContactMaterial", "housing")):
        material = UsdShade.Material.Define(stage, f"{root.GetPath()}/{name}")
        prim = material.GetPrim()
        api = UsdPhysics.MaterialAPI.Apply(prim)
        api.CreateStaticFrictionAttr(contact[f"{role}_static_friction"])
        api.CreateDynamicFrictionAttr(contact[f"{role}_dynamic_friction"])
        api.CreateRestitutionAttr(0.0)
        prim.AddAppliedSchema("PhysxMaterialAPI")
        prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set(
            contact["friction_combine_mode"])
        materials[role] = material
    counts = {}
    for link in root.GetChildren():
        name = link.GetName()
        if name not in links:
            continue
        # Local overrides must not be written through shared instance proxies.
        for prim in list(Usd.PrimRange(link)):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        if name in replace_colliders:
            for prim in Usd.PrimRange(link):
                if prim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        link.AddAppliedSchema("PhysxRigidBodyAPI")
        link.CreateAttribute("physxRigidBody:enableSpeculativeCCD", Sdf.ValueTypeNames.Bool).Set(True)
        colliders = 0
        for prim in Usd.PrimRange(link):
            if not prim.IsA(UsdGeom.Mesh) or f"/{mesh_scope}/" not in str(prim.GetPath()):
                continue
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("convexHull")
            prim.AddAppliedSchema("PhysxCollisionAPI")
            prim.CreateAttribute("physxCollision:contactOffset", Sdf.ValueTypeNames.Float).Set(
                contact["contact_offset_m"])
            prim.CreateAttribute("physxCollision:restOffset", Sdf.ValueTypeNames.Float).Set(
                contact["rest_offset_m"])
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                materials["finger" if name in finger_links else "housing"], materialPurpose="physics")
            colliders += 1
        counts[name] = colliders
    missing = sorted((set(links) - set(counts)) | {name for name, value in counts.items() if value == 0})
    if missing:
        raise RuntimeError(f"Claw contact meshes missing from the '{mesh_scope}' scope: {missing}")
    return counts


def author_host_wrist_contact(root, links, *, finger_contact=None):
    """Replace a host's primitive wrist collider with the wrist mesh hull.

    The Kuavo URDFs approximate the claw-carrying wrist with a 80 mm diameter,
    140 mm long cylinder that reaches about 40 mm past the real part. Reaching
    into a shelf, that volume collides well before the hardware does.
    """
    from .package import load_claw_config

    links = frozenset(links)
    contact = resolve_finger_contact(load_claw_config(), finger_contact)
    return author_claw_jaw_contact(root, contact, links=links, finger_links=frozenset(),
                                   mesh_scope="visuals", replace_colliders=links)


def author_claw_contact(stage, metadata_path, *, side, finger_contact=None, root=None):
    from .linkage import require_closed_linkages

    root = root if root is not None else stage.GetDefaultPrim()
    require_closed_linkages(root, sides=side[0])
    author_claw_inertials(root, metadata_path, side=side)
    config = json.loads(metadata_path.read_text())
    contact = resolve_finger_contact(config, finger_contact)
    housing = config["sides"][side]["root_link"]
    fingers = {f"{side[0]}_{jaw}_finger" for jaw in "fb"}
    author_claw_jaw_contact(root, contact, links=fingers | {housing},
                            finger_links=fingers, mesh_scope="collisions")
    # The source finger is a curved, concave CAD part.  Keep its convex hull as
    # a protective body collider, but make the distal 20 mm grasping surface a
    # soft flat pad which protrudes slightly and therefore contacts first.  The
    # hull sits about 0.6 mm behind the pad face and limits how far it sinks.
    author_claw_distal_pads(root, contact, sides=side[0])
    disable_reference_colliders(root)
    root.SetCustomDataByKey("kuavo:clawPackageVersion", config["schema_version"])
