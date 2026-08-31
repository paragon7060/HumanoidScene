"""Simulation-only planar base motion and articulated waist targets."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from .teleop_body import BODY_JOINTS


class TeleopBodyAction(ActionTerm):
    """Move the fixed articulation root; this is not a wheel dynamics policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, _ = self._asset.find_joints(BODY_JOINTS, preserve_order=True)
        self._actions = torch.zeros((self.num_envs, 6), device=self.device)

    @property
    def action_dim(self):
        return 6

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions[:] = actions
        self._actions[:, :2].clamp_(-.25, .25)
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._actions[:, 2:] = torch.clamp(self._actions[:, 2:], limits[..., 0], limits[..., 1])

    def apply_actions(self):
        self._asset.set_joint_position_target(self._actions[:, 2:], joint_ids=self._joint_ids)
        if torch.any(self._actions[:, :2] != 0):
            pose = self._asset.data.root_pose_w.clone()
            local_velocity = torch.zeros((self.num_envs, 3), device=self.device)
            local_velocity[:, :2] = self._actions[:, :2]
            pose[:, :3] += quat_apply(pose[:, 3:], local_velocity) * self._env.physics_dt
            self._asset.write_root_pose_to_sim(pose)

    def reset(self, env_ids=None):
        self._actions[slice(None) if env_ids is None else env_ids] = 0


@configclass
class TeleopBodyActionCfg(ActionTermCfg):
    class_type: type = TeleopBodyAction
    asset_name: str = "robot"
