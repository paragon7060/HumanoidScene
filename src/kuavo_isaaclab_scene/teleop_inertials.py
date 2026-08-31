"""Simulation-only estimates for missing S200062 hand inertials."""

import json
from .paths import ASSET_DIR


def spawn_teleop_robot(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Gf, Usd, UsdPhysics

    root = spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    estimates = json.loads((ASSET_DIR / "kuavo_s200062/teleop_inertials.json").read_text())["links"]
    for prim in list(Usd.PrimRange(root)):
        if prim.IsInstance() and any(part.startswith("wheel_") for part in str(prim.GetPath()).split("/")):
            prim.SetInstanceable(False)
    remaining = set(estimates)
    wheel_colliders = 0
    for prim in Usd.PrimRange(root):
        # The base is fixed-root/kinematic; cylindrical wheel colliders cannot
        # represent its omni rollers and resist sideways/yaw movement. Keep
        # visual wheels and joint state, omit only their ground contacts.
        if any(part.startswith("wheel_") for part in str(prim.GetPath()).split("/")):
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
    if wheel_colliders < 4:
        raise RuntimeError(f"Expected four or more wheel colliders, found {wheel_colliders}")
    print("[PHYSICS] Applied simulation estimates to 34 hand/frame links lacking URDF inertials; "
          f"0.743 kg per hand; omitted {wheel_colliders} kinematic wheel colliders; "
          "existing arm/torso inertials retained.", flush=True)
    return root
