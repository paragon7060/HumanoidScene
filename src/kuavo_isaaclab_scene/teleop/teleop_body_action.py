"""Simulation-only planar base motion and articulated waist targets."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from .teleop_body import BODY_JOINTS
from ..robots.robot_model import resolve_robot_model


class TeleopBodyAction(ActionTerm):
    """Move the fixed articulation root; this is not a wheel dynamics policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._has_wheel_base = resolve_robot_model().has_wheel_base
        if self._has_wheel_base:
            self._joint_ids, _ = self._asset.find_joints(BODY_JOINTS, preserve_order=True)
            self._wheel_ids, _ = self._asset.find_joints([
                "wheel_left_front_joint", "wheel_right_front_joint",
                "wheel_left_behind_joint", "wheel_right_behind_joint"], preserve_order=True)
        else:
            # S56 is fixed at its torso and has no telescopic/wheel body axes.
            # Preserve the six-channel teleop schema; its final three commands
            # are held at zero while planar root preview remains available.
            self._joint_ids = []
            self._wheel_ids = []
        angles = torch.tensor([.785398163, -.785398163, 2.356194372, -2.356194372], device=self.device)
        self._wheel_tangents = torch.stack((angles.sin(), -angles.cos()), dim=-1)
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
        self._actions[:, 2].clamp_(-1.2, 1.2)
        if self._has_wheel_base:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids[:3]]
            self._actions[:, 3:] = torch.clamp(
                self._actions[:, 3:], limits[..., 0], limits[..., 1]
            )
        else:
            self._actions[:, 3:] = 0.0

    def apply_actions(self):
        if self._has_wheel_base:
            torso = torch.cat((self._actions[:, 3:], torch.zeros_like(self._actions[:, :1])), -1)
            self._asset.set_joint_position_target(torso, joint_ids=self._joint_ids)
            # The model has four radial omni wheels (r=0.13035 m, offset=0.32879 m).
            # Synchronize wheel spin with the fixed-root planar drive. This remains
            # a kinematic simulation base, not contact-driven wheel locomotion.
            speeds = (
                self._actions[:, :2] @ self._wheel_tangents.T
                - .32879 * self._actions[:, 2:3]
            ) / .13035
            self._asset.set_joint_velocity_target(speeds, joint_ids=self._wheel_ids)
            # These imported wheels have simple cylindrical colliders, not omni
            # rollers. Synchronize their phase with the kinematic root as well;
            # floor friction must not visually stall one side during a turn.
            wheel_q = self._asset.data.joint_pos[:, self._wheel_ids] + speeds * self._env.physics_dt
            self._asset.write_joint_state_to_sim(wheel_q, speeds, joint_ids=self._wheel_ids)
        if torch.any(self._actions[:, :3] != 0):
            pose = self._asset.data.root_pose_w.clone()
            local_velocity = torch.zeros((self.num_envs, 3), device=self.device)
            local_velocity[:, :2] = self._actions[:, :2]
            pose[:, :3] += quat_apply(pose[:, 3:], local_velocity) * self._env.physics_dt
            angle = self._actions[:, 2] * self._env.physics_dt
            rotation = torch.zeros_like(pose[:, 3:]); rotation[:, 0] = torch.cos(angle / 2)
            rotation[:, 3] = torch.sin(angle / 2)
            pose[:, 3:] = quat_mul(rotation, pose[:, 3:])
            self._asset.write_root_pose_to_sim(pose)

    def reset(self, env_ids=None):
        self._actions[slice(None) if env_ids is None else env_ids] = 0


@configclass
class TeleopBodyActionCfg(ActionTermCfg):
    class_type: type = TeleopBodyAction
    asset_name: str = "robot"
