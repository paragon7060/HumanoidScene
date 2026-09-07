"""RL-only upright flap constraints; the shared VR/USD files remain reusable."""

from isaaclab.sim.utils import clone
from ...envs.contact_physics import spawn_contact_box


@clone
def spawn_locked_flap_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    from pxr import Usd, UsdPhysics
    root = spawn_contact_box(prim_path, cfg, translation, orientation, **kwargs)
    count = 0
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.RevoluteJoint) and prim.GetName() in (
                "joint_front", "joint_back", "joint_left", "joint_right"):
            joint = UsdPhysics.RevoluteJoint(prim)
            joint.CreateLowerLimitAttr(-cfg.flap_lock_degrees)
            joint.CreateUpperLimitAttr(cfg.flap_lock_degrees)
            count += 1
    if count != 4:
        raise ValueError(f"Expected four flap hinges, found {count}: {prim_path}")
    return root
