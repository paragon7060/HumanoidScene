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
        if cfg.controller.command_type != "pose":
            raise ValueError("Persistent teleop IK requires pose commands.")
        self._target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._target_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.orientation_weight = 0.0
        self._following = True
        self._held_joints = None

    def set_following(self, enabled: bool):
        if not enabled and (self._following or self._held_joints is None):
            self._held_joints = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._following = enabled

    def hold_current_pose(self, env_ids=None):
        indices = slice(None) if env_ids is None else env_ids
        position, orientation = self._compute_frame_pose()
        self._target_position[indices] = position[indices]
        self._target_orientation[indices] = orientation[indices]
        self._target_ready[indices] = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        self._target_ready[slice(None) if env_ids is None else env_ids] = False
        self._held_joints = None

    def process_actions(self, actions):
        # Retain upstream action scaling/clipping; replace its temporary target.
        super().process_actions(actions)
        missing = torch.nonzero(~self._target_ready, as_tuple=False).flatten()
        if missing.numel():
            self.hold_current_pose(missing)
        if self.cfg.controller.use_relative_mode:
            target_position, target_orientation = apply_delta_pose(
                self._target_position, self._target_orientation, self._processed_actions
            )
        else:
            target_position = self._processed_actions[:, :3].clone()
            target_orientation = torch.nn.functional.normalize(self._processed_actions[:, 3:7], dim=-1)
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

    def apply_actions(self):
        if not self._following and self._held_joints is not None:
            # Explicit pause holds the captured joint pose, including the
            # redundant joints. Tracking loss while following still runs IK.
            self._asset.set_joint_position_target(self._held_joints, self._joint_ids)
            return
        # Position is the primary task. A controller orientation outside wrist
        # joint travel must not prevent the arm reaching its visible position.
        position, orientation = self._compute_frame_pose()
        position_error, rotation_error = compute_pose_error(
            position, orientation, self._ik_controller.ee_pos_des, self._ik_controller.ee_quat_des,
            rot_error_type="axis_angle",
        )
        jacobian = self._compute_frame_jacobian().clone()
        jacobian[:, 3:] *= self.orientation_weight
        error = torch.cat((position_error, rotation_error * self.orientation_weight), -1)
        joints = self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        identity = torch.eye(6, device=self.device).unsqueeze(0)
        # Remove joints already at a limit when the unconstrained solve asks
        # them to travel farther outward, then solve using the other joints.
        for _ in range(3):
            delta = (jacobian.transpose(1, 2) @ torch.linalg.solve(
                jacobian @ jacobian.transpose(1, 2) + .035 ** 2 * identity, error.unsqueeze(-1)
            )).squeeze(-1)
            blocked = ((joints <= limits[..., 0] + .01) & (delta < 0)
                       | (joints >= limits[..., 1] - .01) & (delta > 0))
            if not torch.any(blocked):
                break
            jacobian = jacobian.masked_fill(blocked.unsqueeze(1), 0.)
        # Redundant wrist/shoulder joints otherwise drift to their stops while
        # solving position alone, trapping the next upward target. Prefer the
        # bent ready pose only in directions that do not move the tool.
        inverse = jacobian.transpose(1, 2) @ torch.linalg.solve(
            jacobian @ jacobian.transpose(1, 2) + .035 ** 2 * identity, identity.expand(self.num_envs, -1, -1)
        )
        nullspace = torch.eye(len(self._joint_ids), device=self.device).unsqueeze(0) - inverse @ jacobian
        rest = self._asset.data.default_joint_pos[:, self._joint_ids]
        delta += .08 * (nullspace @ (rest - joints).unsqueeze(-1)).squeeze(-1)
        desired = torch.clamp(joints + delta.clamp(-.12, .12), limits[..., 0], limits[..., 1])
        self._asset.set_joint_position_target(desired, self._joint_ids)

    def target_position_error(self):
        position, _ = self._compute_frame_pose()
        return (self._target_position - position).norm(dim=-1)
