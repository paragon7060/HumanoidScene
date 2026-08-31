"""Persistent, bounded arm targets for relative Quest input."""

import torch

from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import apply_delta_pose, compute_pose_error


class PersistentTeleopIKAction(DifferentialInverseKinematicsAction):
    """Accumulate relative input against the target instead of the lagging arm.

    Retain the requested target while limiting each IK correction to 15 cm /
    0.5 rad. Stopping the controller or briefly losing tracking does not cancel
    the remaining goal. Only explicit pause/recalibration re-clutches the robot.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not cfg.controller.use_relative_mode or cfg.controller.command_type != "pose":
            raise ValueError("Persistent teleop IK requires relative pose commands.")
        self._target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._target_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def hold_current_pose(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        position, orientation = self._compute_frame_pose()
        self._target_position[indices] = position[indices]
        self._target_orientation[indices] = orientation[indices]
        self._target_ready[indices] = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._target_ready[slice(None) if env_ids is None else env_ids] = False

    def process_actions(self, actions):
        # Retain upstream action scaling/clipping; replace its temporary target.
        super().process_actions(actions)
        missing = torch.nonzero(~self._target_ready, as_tuple=False).flatten()
        if missing.numel():
            self.hold_current_pose(missing)
        target_position, target_orientation = apply_delta_pose(
            self._target_position, self._target_orientation, self._processed_actions
        )
        position, orientation = self._compute_frame_pose()
        position_error, rotation_error = compute_pose_error(
            position, orientation, target_position, target_orientation, rot_error_type="axis_angle"
        )
        position_error *= (0.15 / position_error.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        rotation_error *= (0.5 / rotation_error.norm(dim=-1, keepdim=True).clamp_min(1e-8)).clamp(max=1.0)
        correction_position, correction_orientation = apply_delta_pose(
            position, orientation, torch.cat((position_error, rotation_error), dim=-1)
        )
        self._target_position, self._target_orientation = target_position, target_orientation
        self._ik_controller.ee_pos_des[:] = correction_position
        self._ik_controller.ee_quat_des[:] = correction_orientation

    def target_position_error(self):
        position, _ = self._compute_frame_pose()
        return (self._target_position - position).norm(dim=-1)
