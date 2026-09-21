"""Pose-randomizable kinematic rigid object used by the v2 workcell."""

import torch
from isaaclab.assets import RigidObject


class PoseControlledKinematicObject(RigidObject):
    """Allow pose resets while suppressing invalid zero-velocity PhysX writes."""

    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        rigid_props = None if self.cfg.spawn is None else self.cfg.spawn.rigid_props
        # USD rack rigid-body properties are authored by its custom spawn
        # function after the reference is loaded, so its config field is None.
        if rigid_props is not None and not rigid_props.kinematic_enabled:
            return super().write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        if not torch.isfinite(root_velocity).all() or torch.count_nonzero(root_velocity).item():
            raise ValueError("Kinematic workcell objects accept pose commands only.")
