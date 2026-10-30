"""Avoid an invalid PhysX velocity reset on the inspection scene's fixed belt."""

import torch
from isaaclab.assets import RigidObject


class StationarySurface(RigidObject):
    def write_root_velocity_to_sim(self, root_velocity, env_ids=None):
        # reset_scene_to_default writes zero velocity to every RigidObject,
        # including kinematics. PhysX rejects even zero for a kinematic body.
        # This adapter is assigned only to the stationary conveyor surface;
        # dynamic boxes, hands and other assets keep their original reset.
        if not self.cfg.spawn.rigid_props.kinematic_enabled:
            return super().write_root_velocity_to_sim(root_velocity, env_ids=env_ids)
        if not torch.isfinite(root_velocity).all() or torch.count_nonzero(root_velocity).item():
            raise ValueError("Stationary conveyor surface cannot receive a velocity command")
        # No force/velocity command is needed: pose-only reset keeps this body fixed.
