"""Persistent pose targets with a bounded control-rate joint servo."""

import math

import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import apply_delta_pose, compute_pose_error


class PersistentTeleopIKAction(DifferentialInverseKinematicsAction):
    """Solve once per control tick; hold that command over physics substeps.

    Raw goals persist, including when tracking is lost. Filter sensor noise and
    bound joint velocity/acceleration instead of chasing a new IK solution on
    every physics substep. Explicit pause alone captures the actual joint pose.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.controller.command_type != "pose":
            raise ValueError("Persistent teleop IK requires pose commands.")
        self._target_position = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_orientation = torch.zeros((self.num_envs, 4), device=self.device)
        self._filtered_position = self._target_position.clone()
        self._filtered_orientation = self._target_orientation.clone()
        self._target_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._joint_command = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._joint_velocity = torch.zeros_like(self._joint_command)
        self._gravity_bias = torch.zeros_like(self._joint_command)
        # Optional redundancy/posture target.  The normal teleop path leaves
        # this unset and therefore keeps the historical default-joint
        # null-space preference.  Task planners can set only the wrist-pitch
        # component while still solving the full TCP pose with IK.
        self._posture_target = None
        self._posture_weight = 0.15
        self.orientation_weight = 0.5
        self._following = True
        self._held_joints = None
        self._dt = env.step_dt

    def set_following(self, enabled):
        if not enabled and (self._following or self._held_joints is None):
            self._held_joints = self._asset.data.joint_pos[:, self._joint_ids].clone()
            self._joint_velocity.zero_()
        if enabled and not self._following:
            self.hold_current_pose()
        self._following = enabled

    def hold_current_pose(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        p, q = self._compute_frame_pose()
        self._target_position[ids] = self._filtered_position[ids] = p[ids]
        self._target_orientation[ids] = self._filtered_orientation[ids] = q[ids]
        self._joint_command[ids] = self._asset.data.joint_pos[:, self._joint_ids][ids]
        self._joint_velocity[ids] = 0
        self._target_ready[ids] = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._target_ready[ids] = False
        self._joint_velocity[ids] = 0
        self._held_joints = None
        if env_ids is None:
            self._posture_target = None
        elif self._posture_target is not None:
            self._posture_target[ids] = self._asset.data.default_joint_pos[:, self._joint_ids][ids]

    def set_posture_target(self, target, *, weight: float = 0.15):
        """Set a joint-space null-space target for this arm's IK.

        ``target`` is the complete arm joint vector in the action term's
        joint order.  A planner may copy the current vector and replace only
        ``zarm_*_joint`` pitch, which avoids post-solve joint overwrites that
        would invalidate the requested TCP pose.
        """
        target = torch.as_tensor(target, device=self.device, dtype=self._joint_command.dtype)
        if target.ndim == 1:
            target = target.unsqueeze(0).expand(self.num_envs, -1)
        expected = (self.num_envs, self._num_joints)
        if tuple(target.shape) != expected:
            raise ValueError(f"posture target shape must be {expected}, got {tuple(target.shape)}")
        if not torch.isfinite(target).all():
            raise ValueError("posture target must contain only finite values")
        if not math.isfinite(float(weight)) or weight < 0.0:
            raise ValueError(f"posture weight must be finite and nonnegative, got {weight}")
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._posture_target = torch.clamp(target, limits[..., 0], limits[..., 1]).clone()
        self._posture_weight = float(weight)

    def clear_posture_target(self):
        """Restore the historical default-joint null-space preference."""
        self._posture_target = None
        self._posture_weight = 0.15

    def process_actions(self, actions):
        super().process_actions(actions)
        missing = torch.nonzero(~self._target_ready, as_tuple=False).flatten()
        if missing.numel():
            self.hold_current_pose(missing)
        if self.cfg.controller.use_relative_mode:
            self._target_position, self._target_orientation = apply_delta_pose(
                self._target_position, self._target_orientation, self._processed_actions)
        else:
            self._target_position = self._processed_actions[:, :3].clone()
            self._target_orientation = torch.nn.functional.normalize(self._processed_actions[:, 3:7], dim=-1)
        # Express gravity feedforward as a small implicit-drive position bias.
        # The existing PhysX drive effort cap still limits the entire torque;
        # no external torque is added on top of that cap.
        gravity = self._asset.root_physx_view.get_gravity_compensation_forces()[:, self._joint_ids]
        stiffness = self._asset.data.joint_stiffness[:, self._joint_ids].clamp_min(1.)
        self._gravity_bias = gravity / stiffness
        if not self._following:
            return
        # Quaternion hemisphere continuity prevents sign flips from becoming
        # full turns. Smoothing converges to the unchanged raw requested pose.
        alpha = self._dt / (.045 + self._dt)
        self._filtered_position.lerp_(self._target_position, alpha)
        sign = torch.where((self._filtered_orientation * self._target_orientation).sum(-1, keepdim=True) < 0, -1., 1.)
        self._filtered_orientation = torch.nn.functional.normalize(
            self._filtered_orientation.lerp(self._target_orientation * sign, alpha), dim=-1)
        position, orientation = self._compute_frame_pose()
        ep, er = compute_pose_error(position, orientation, self._filtered_position, self._filtered_orientation,
                                    rot_error_type="axis_angle")
        jac = self._compute_frame_jacobian().clone()
        jac[:, 3:] *= self.orientation_weight
        # Cartesian feedback becomes a joint velocity, not an unscaled pose
        # jump. Damping stays continuous near singularities and joint stops.
        error = torch.cat((ep * 2.5, er * (2.5 * self.orientation_weight)), -1)
        ident = torch.eye(6, device=self.device).expand(self.num_envs, -1, -1)
        joints = self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        rest = self._asset.data.default_joint_pos[:, self._joint_ids]
        posture = rest if self._posture_target is None else self._posture_target
        inverse = jac.transpose(1, 2) @ torch.linalg.solve(
            jac @ jac.transpose(1, 2) + .08 ** 2 * ident, ident)
        velocity = (inverse @ error.unsqueeze(-1)).squeeze(-1)
        # Continuous damping and a weak posture preference avoid abrupt
        # solution changes at a joint stop. Hard bounds are enforced below.
        nullspace = torch.eye(joints.shape[-1], device=self.device) - inverse @ jac
        velocity += self._posture_weight * (nullspace @ (posture - joints).unsqueeze(-1)).squeeze(-1)
        velocity *= (1.5 / velocity.abs().amax(-1, keepdim=True).clamp_min(1.5))
        velocity = torch.clamp(velocity, self._joint_velocity - 12. * self._dt,
                               self._joint_velocity + 12. * self._dt)
        velocity = torch.clamp(velocity, (limits[..., 0] - joints) / self._dt,
                               (limits[..., 1] - joints) / self._dt)
        self._joint_velocity = velocity
        # Integrate the desired velocity so residual drive error does not
        # become a permanent Cartesian offset. Bound lead to prevent windup
        # when a link is blocked by an object or a joint stop.
        command = self._joint_command + velocity * self._dt
        command = torch.clamp(command, joints - .10, joints + .10)
        self._joint_command = torch.clamp(command, limits[..., 0], limits[..., 1])

    def apply_actions(self):
        command = self._held_joints if not self._following and self._held_joints is not None else self._joint_command
        self._asset.set_joint_velocity_target(self._joint_velocity if self._following else torch.zeros_like(self._joint_velocity), self._joint_ids)
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._asset.set_joint_position_target(torch.clamp(command + self._gravity_bias, limits[..., 0], limits[..., 1]), self._joint_ids)

    def target_orientation_error(self):
        p, q = self._compute_frame_pose()
        _, error = compute_pose_error(p, q, p, self._target_orientation, rot_error_type="axis_angle")
        return error.norm(dim=-1)

    def target_position_error(self):
        return (self._target_position - self._compute_frame_pose()[0]).norm(dim=-1)
