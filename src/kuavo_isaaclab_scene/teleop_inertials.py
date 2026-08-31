"""Simulation-only estimates for missing S200062 hand inertials."""

import json
from isaaclab.sim.utils import clone
from .paths import ASSET_DIR


@clone
def spawn_s200062_robot(prim_path, cfg, translation=None, orientation=None,
                       *, disable_wheel_contacts=False, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Gf, Usd, UsdPhysics

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
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
    from .teleop_contacts import add_hand_colliders
    add_hand_colliders(root)
    wheel_status = (f"omitted {wheel_colliders} kinematic wheel colliders"
                    if disable_wheel_contacts else "wheel contacts retained")
    print("[PHYSICS] Applied simulation estimates to 34 hand/frame links lacking URDF inertials; "
          f"0.743 kg per hand; {wheel_status}; "
          "existing arm/torso inertials retained.", flush=True)
    return root


def spawn_teleop_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Quest-only exception for the kinematically translated/rotated base."""
    return spawn_s200062_robot(prim_path, cfg, translation, orientation,
                              disable_wheel_contacts=True, **kwargs)
