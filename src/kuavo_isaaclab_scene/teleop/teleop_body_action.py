"""Planar base motion and articulated waist targets for Quest teleop."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from .teleop_body import BASE_LINEAR_SPEED_M_S, BASE_YAW_SPEED_RAD_S, BODY_JOINTS
from ..robots.base_drive import WHEEL_ANGLES_RAD, WHEEL_JOINTS, WHEEL_OFFSET_M, WHEEL_RADIUS_M
from ..robots.base_drive_control import FloatingBaseDrive, FloatingBaseDriveCfg
from ..robots.robot_model import resolve_robot_model


class TeleopBodyAction(ActionTerm):
    """Move the base with the selected base model; not a wheel dynamics policy.

    The default kinematic model overwrites the root pose. With a dynamic base
    the shared floating-base drive carries the chassis instead, so a recorded
    demonstration has the same base dynamics an RL policy trains against.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._has_wheel_base = resolve_robot_model().has_wheel_base
        if self._has_wheel_base:
            self._joint_ids, _ = self._asset.find_joints(BODY_JOINTS, preserve_order=True)
            self._wheel_ids, _ = self._asset.find_joints(list(WHEEL_JOINTS), preserve_order=True)
        else:
            # S56 is fixed at its torso and has no telescopic/wheel body axes.
            # Preserve the seven-channel teleop schema; its final four commands
            # are held at zero while planar root preview remains available.
            self._joint_ids = []
            self._wheel_ids = []
        angles = torch.tensor(WHEEL_ANGLES_RAD, device=self.device)
        self._wheel_tangents = torch.stack((angles.sin(), -angles.cos()), dim=-1)
        self._actions = torch.zeros((self.num_envs, 7), device=self.device)
        self._drive = None
        if cfg.dynamic:
            if not self._has_wheel_base:
                raise ValueError("A dynamic base needs the wheeled chassis; this model has none.")
            self._drive = FloatingBaseDrive(self._asset, cfg.drive, env)

    @property
    def action_dim(self):
        return 7

    @property
    def raw_actions(self):
        return self._actions

    @property
    def processed_actions(self):
        return self._actions

    def process_actions(self, actions):
        self._actions[:] = actions
        self._actions[:, :2].clamp_(-BASE_LINEAR_SPEED_M_S, BASE_LINEAR_SPEED_M_S)
        # Match the mapper's vector cap, including direct diagonal commands.
        speed = self._actions[:, :2].norm(dim=-1, keepdim=True)
        self._actions[:, :2] /= (speed / BASE_LINEAR_SPEED_M_S).clamp_min(1.0)
        self._actions[:, 2].clamp_(-BASE_YAW_SPEED_RAD_S, BASE_YAW_SPEED_RAD_S)
        if self._has_wheel_base:
            limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
            self._actions[:, 3:] = torch.clamp(
                self._actions[:, 3:], limits[..., 0], limits[..., 1]
            )
        else:
            self._actions[:, 3:] = 0.0

    def apply_actions(self):
        if self._has_wheel_base:
            self._asset.set_joint_position_target(self._actions[:, 3:], joint_ids=self._joint_ids)
        if self._drive is not None:
            # The drive owns root motion and wheel spin; no state is written.
            self._drive.apply(self._actions[:, :3])
            return
        if self._has_wheel_base:
            # The model has four radial omni wheels (r=0.13035 m, offset=0.32879 m).
            # Synchronize wheel spin with the fixed-root planar drive. This remains
            # a kinematic simulation base, not contact-driven wheel locomotion.
            speeds = (
                self._actions[:, :2] @ self._wheel_tangents.T
                - WHEEL_OFFSET_M * self._actions[:, 2:3]
            ) / WHEEL_RADIUS_M
            self._asset.set_joint_velocity_target(speeds, joint_ids=self._wheel_ids)
            # These imported wheels have simple cylindrical colliders, not omni
            # rollers. Synchronize their phase with the kinematic root as well;
            # floor friction must not visually stall one side during a turn.
            wheel_q = self._asset.data.joint_pos[:, self._wheel_ids] + speeds * self._env.physics_dt
            self._asset.write_joint_state_to_sim(wheel_q, speeds, joint_ids=self._wheel_ids)
        pose = self._asset.data.root_pose_w.clone()
        orientation = pose[:, 3:].clone()
        local_velocity = torch.zeros((self.num_envs, 3), device=self.device)
        local_velocity[:, :2] = self._actions[:, :2]
        linear_world = quat_apply(orientation, local_velocity)
        pose[:, :3] += linear_world * self._env.physics_dt
        angle = self._actions[:, 2] * self._env.physics_dt
        rotation = torch.zeros_like(orientation); rotation[:, 0] = torch.cos(angle / 2)
        rotation[:, 3] = torch.sin(angle / 2)
        pose[:, 3:] = quat_mul(rotation, orientation)
        self._asset.write_root_pose_to_sim(pose)
        # Always pair the kinematic pose write with a matching root velocity,
        # including zero once the joystick is released. Otherwise a grasped
        # box's contact/friction solve sees a stale nonzero root velocity
        # left over from the previous step (spurious slip), or later sees a
        # jump with no velocity at all when motion resumes. Gating this write
        # on nonzero actions (previous behavior) skipped exactly the zeroing
        # step needed on release.
        root_velocity = torch.zeros((self.num_envs, 6), device=self.device)
        root_velocity[:, :3] = linear_world
        root_velocity[:, 5] = self._actions[:, 2]
        self._asset.write_root_velocity_to_sim(root_velocity)

    def reset(self, env_ids=None):
        self._actions[slice(None) if env_ids is None else env_ids] = 0
        if self._drive is not None:
            self._drive.reset(env_ids)


@configclass
class TeleopBodyActionCfg(ActionTermCfg):
    class_type: type = TeleopBodyAction
    asset_name: str = "robot"
    # Set through robots/base_drive.py so every entry point agrees.
    dynamic: bool = False
    drive: FloatingBaseDriveCfg = FloatingBaseDriveCfg()
