"""Contact geometry adapters for robot hands and articulated cardboard boxes."""

from isaaclab.sim.utils import clone
from ..robots.gripper_config import FingerContactSettings


def add_hand_colliders(root, finger_contact: FingerContactSettings = FingerContactSettings()):
    """Compatibility adapter to the package-owned Leju claw contact model."""
    from ..robots.claw_assets.isaaclab import author_integrated_claw_contact
    return author_integrated_claw_contact(root, finger_contact)


@clone
def spawn_contact_box(prim_path, cfg, translation=None, orientation=None, **kwargs):
    # Author contact overrides before the outer decorator copies the asset
    # into additional environments (including copy_from_source=True).
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
