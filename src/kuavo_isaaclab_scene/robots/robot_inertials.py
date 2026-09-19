"""Simulation-only estimates for missing S200062 hand inertials."""

import json
from isaaclab.sim.utils import clone
from ..core.paths import ASSET_DIR


def _move_floating_articulation_root(root_prim) -> None:
    """Move a fixed-export USD's articulation API onto its floating base link.

    Isaac Lab can disable the imported world joint through ``fix_root_link=False``,
    but fixed-export USDs keep ArticulationRootAPI on that now-disabled joint.
    PhysX then has no rigid body at the advertised articulation path.  A floating
    articulation must advertise its actual root rigid body instead.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    source = next(
        (prim for prim in Usd.PrimRange(root_prim)
         if prim.HasAPI(UsdPhysics.ArticulationRootAPI)),
        None,
    )
    target = root_prim.GetChild("base_link")
    if source is None or not target.IsValid() or not target.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"Cannot make {root_prim.GetPath()} floating: expected one articulation root "
            "and a rigid base_link."
        )
    if source == target:
        return

    usd_source = UsdPhysics.ArticulationRootAPI(source)
    physx_source = PhysxSchema.PhysxArticulationAPI(source)
    UsdPhysics.ArticulationRootAPI.Apply(target)
    PhysxSchema.PhysxArticulationAPI.Apply(target)
    for api in (usd_source, physx_source):
        if not api:
            continue
        for name in api.GetSchemaAttributeNames():
            value = source.GetAttribute(name).Get()
            if value is not None:
                target.GetAttribute(name).Set(value)
    # This USD was exported for a fixed installation.  Leaving the world
    # joint enabled would anchor the otherwise floating articulation and can
    # also inject a one-time pose snap when PhysX creates the scene.
    joint = UsdPhysics.Joint(source)
    if joint:
        joint.CreateJointEnabledAttr(False)
    source.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    if physx_source:
        source.RemoveAPI(PhysxSchema.PhysxArticulationAPI)


def _suspend_wheel_ground_contacts(root) -> int:
    """Remove the wheel ground contacts of a floating, wrench-driven chassis.

    The imported wheels are plain cylinders; the real base uses omni rollers.
    On a floating root those cylinders weld the chassis to the floor: sideways
    and yaw commands have to break a 0.8+ friction cone under a 219 kg robot,
    which no reasonable chassis wrench can do, so the base simply stalls.
    The planar drive carries the chassis with its own vertical/tilt PD instead,
    exactly like the previous fixed-root base did, while every link below the
    root stays fully dynamic for contact-correct carrying.
    """
    from pxr import Usd, UsdPhysics

    def is_wheel(prim):
        return any(part.startswith("wheel_") for part in str(prim.GetPath()).split("/"))

    for prim in list(Usd.PrimRange(root)):
        # Collision edits cannot be written through shared instance proxies.
        if prim.IsInstance() and is_wheel(prim):
            prim.SetInstanceable(False)
    disabled = 0
    for prim in Usd.PrimRange(root):
        if is_wheel(prim) and prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
            disabled += 1
    if disabled < 4:
        raise RuntimeError(f"Expected four or more wheel colliders, found {disabled}")
    return disabled


def _make_floating_base(root, model_label: str) -> None:
    """Prepare a fixed-export robot USD to be driven as a floating chassis."""
    _move_floating_articulation_root(root)
    wheels = _suspend_wheel_ground_contacts(root)
    print(f"[PHYSICS] Floating {model_label} root: removed {wheels} cylindrical wheel ground "
          "contacts; the planar drive suspends and moves the chassis with a PD wrench.",
          flush=True)


def _is_floating(cfg) -> bool:
    return cfg.articulation_props is not None and cfg.articulation_props.fix_root_link is False


def _spawn_twofinger_robot(prim_path, cfg, translation=None, orientation=None,
                           *, disable_wheel_contacts=False, model_label="S200062", **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Gf, Usd, UsdPhysics

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    if _is_floating(cfg):
        # A floating chassis is carried by the planar drive's PD wrench, so it
        # must not also rest on the imported cylindrical wheels.
        _make_floating_base(root, model_label)
        disable_wheel_contacts = False
    from .claw_assets.linkage import require_closed_linkages
    require_closed_linkages(root)
    estimates = json.loads((ASSET_DIR / "kuavo_s200062/teleop_inertials.json").read_text())["links"]
    for prim in list(Usd.PrimRange(root)):
        if (disable_wheel_contacts and prim.IsInstance()
                and any(part.startswith("wheel_") for part in str(prim.GetPath()).split("/"))):
            prim.SetInstanceable(False)
    remaining = set(estimates)
    wheel_colliders = 0
    for prim in Usd.PrimRange(root):
        # The base is fixed-root/kinematic; cylindrical wheel colliders cannot
        # represent its omni rollers and resist sideways/yaw movement. Keep
        # visual wheels and joint state, omit only their ground contacts.
        if disable_wheel_contacts and any(part.startswith("wheel_") for part in str(prim.GetPath()).split("/")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                wheel_colliders += 1
        name = prim.GetName()
        if name not in estimates or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        values = estimates[name]
        api = UsdPhysics.MassAPI.Apply(prim)
        api.CreateMassAttr(values["mass_kg"])
        api.CreateCenterOfMassAttr(Gf.Vec3f(*values["com_m"]))
        api.CreateDiagonalInertiaAttr(Gf.Vec3f(*values["diagonal_inertia_kg_m2"]))
        api.CreatePrincipalAxesAttr(Gf.Quatf(1.0))
        remaining.discard(name)
    if remaining:
        raise RuntimeError(f"Missing S200062 hand rigid bodies for inertial correction: {sorted(remaining)}")
    if disable_wheel_contacts and wheel_colliders < 4:
        raise RuntimeError(f"Expected four or more wheel colliders, found {wheel_colliders}")
    from .claw_assets.isaaclab import author_integrated_claw_contact
    from .gripper_config import load_gripper_settings
    preset = "s56_twofinger" if model_label == "S56" else "s200062_integrated"
    author_integrated_claw_contact(root, load_gripper_settings(preset).finger_contact)
    from .end_effector import spawn_center_prims
    spawn_center_prims(root)
    wheel_status = (f"omitted {wheel_colliders} kinematic wheel colliders"
                    if disable_wheel_contacts else "wheel contacts retained")
    print(f"[PHYSICS] Applied {model_label} two-finger simulation estimates to "
          "34 hand/frame links lacking URDF inertials; "
          f"0.743 kg per hand; {wheel_status}; "
          "existing arm/torso inertials retained.", flush=True)
    return root


@clone
def spawn_s200062_robot(prim_path, cfg, translation=None, orientation=None,
                        *, disable_wheel_contacts=False, **kwargs):
    return _spawn_twofinger_robot(
        prim_path,
        cfg,
        translation,
        orientation,
        disable_wheel_contacts=disable_wheel_contacts,
        model_label="S200062",
        **kwargs,
    )


@clone
def spawn_s63_twofinger_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """S63 variant already contains estimated claw inertials and physical loops."""
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from .claw_assets.linkage import require_closed_linkages
    from .gripper_config import resolve_gripper_settings
    from .claw_assets.package import CLAW_ASSET_DIR
    from .claw_assets.usd import author_claw_contact, author_host_wrist_contact
    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    if _is_floating(cfg):
        _make_floating_base(root, "S63")
    require_closed_linkages(root)
    settings = resolve_gripper_settings()
    for side in ("left", "right"):
        author_claw_contact(root.GetStage(), CLAW_ASSET_DIR / "config.json",
                            side=side, finger_contact=settings.finger_contact, root=root)
    # Same wrist treatment the S200062/S56 integration already applies, so all
    # three hosts collide with the wrist hull rather than the URDF cylinder.
    author_host_wrist_contact(root, ("zarm_l7_link", "zarm_r7_link"),
                              finger_contact=settings.finger_contact)
    return root


@clone
def spawn_s56_twofinger_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn the S56 articulation carrying the transplanted S200062 hand rig."""
    return _spawn_twofinger_robot(
        prim_path,
        cfg,
        translation,
        orientation,
        model_label="S56",
        **kwargs,
    )


def spawn_teleop_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Quest-only exception for the kinematically translated/rotated base."""
    return spawn_s200062_robot(prim_path, cfg, translation, orientation,
                              disable_wheel_contacts=True, **kwargs)
