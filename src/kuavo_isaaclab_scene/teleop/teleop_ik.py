"""Persistent pose targets with a bounded control-rate joint servo."""

import torch
import numpy as np
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils.math import apply_delta_pose, compute_pose_error
from .teleop_servo import SMOOTH, joint_servo_step
from .urdf_arm_ik import quat_matrix


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
        self.orientation_weight = 0.5
        self.response = SMOOTH
        self._following = True
        self._held_joints = None
        self._dt = env.step_dt
        self._urdf_arm = None
        self.ik_status = {}

    def configure_urdf(self, arm):
        """Validate live USD FK/Jacobian before enabling URDF joint-space IK."""
        if self.num_envs != 1 or self.cfg.body_offset is not None:
            raise ValueError("URDF teleop currently requires one environment and the named tool frame (no body_offset)")
        if set(self._joint_names) != set(arm.names) or self._body_name != arm.tip:
            raise ValueError("USD action joint/tool names differ from URDF")
        ids, _ = self._asset.find_bodies(arm.parent)
        if len(ids) != 1:
            raise ValueError(f"Missing URDF arm parent in USD: {arm.parent}")
        self._urdf_parent_id = ids[0]
        self._urdf_order = [self._joint_names.index(name) for name in arm.names]
        q = self._numpy(self._asset.data.joint_pos[0, self._joint_ids])[self._urdf_order]
        parent_p = self._numpy(self._asset.data.body_pos_w[0, ids[0]])
        parent_r = quat_matrix(self._numpy(self._asset.data.body_quat_w[0, ids[0]]))
        actual_p = parent_r.T @ (self._numpy(self._asset.data.body_pos_w[0, self._body_idx]) - parent_p)
        actual_r = parent_r.T @ quat_matrix(self._numpy(self._asset.data.body_quat_w[0, self._body_idx]))
        actual_jac = self._numpy(self.jacobian_w[0])[:, self._urdf_order].copy()
        actual_jac[:3] = parent_r.T @ actual_jac[:3]
        actual_jac[3:] = parent_r.T @ actual_jac[3:]
        limits = self._numpy(self._asset.data.joint_pos_limits[0, self._joint_ids])[self._urdf_order]
        arm.validate_live(q, actual_p, actual_r, actual_jac, limits)
        self._urdf_rest = q.copy()
        self._urdf_arm = arm
        print(f"[URDF IK] {arm.side}: live USD FK/Jacobian/limits matched; "
              f"parent={arm.parent}; tool={arm.tip}; reach bound={.95 * arm.reach:.3f}m", flush=True)

    @staticmethod
    def _numpy(value):
        return value.detach().cpu().numpy()

    def _process_urdf(self, joints, limits):
        order = self._urdf_order
        parent_p = self._numpy(self._asset.data.body_pos_w[0, self._urdf_parent_id])
        parent_r = quat_matrix(self._numpy(self._asset.data.body_quat_w[0, self._urdf_parent_id]))
        root_p = self._numpy(self._asset.data.root_pos_w[0])
        root_r = quat_matrix(self._numpy(self._asset.data.root_quat_w[0]))
        target = parent_r.T @ (root_p + root_r @ self._numpy(self._filtered_position[0]) - parent_p)
        target_r = parent_r.T @ root_r @ quat_matrix(self._numpy(self._filtered_orientation[0]))
        bounds = self._numpy(limits[0])[order]
        velocity, _, self.ik_status = self._urdf_arm.step(
            self._numpy(joints[0])[order], target, target_r, self._urdf_rest,
            self._numpy(self._joint_velocity[0])[order], self._dt, self.response,
            self.orientation_weight, bounds[:, 0], bounds[:, 1])
        self._joint_velocity[0, order] = torch.as_tensor(velocity, device=self.device, dtype=joints.dtype)
        command = self._joint_command + self._joint_velocity * self._dt
        command = torch.clamp(command, joints - .10, joints + .10)
        self._joint_command = torch.clamp(command, limits[..., 0], limits[..., 1])

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
        alpha = self._dt / (self.response.smoothing_s + self._dt)
        self._filtered_position.lerp_(self._target_position, alpha)
        sign = torch.where((self._filtered_orientation * self._target_orientation).sum(-1, keepdim=True) < 0, -1., 1.)
        self._filtered_orientation = torch.nn.functional.normalize(
            self._filtered_orientation.lerp(self._target_orientation * sign, alpha), dim=-1)
        if self._urdf_arm is not None:
            self._process_urdf(self._asset.data.joint_pos[:, self._joint_ids],
                               self._asset.data.joint_pos_limits[:, self._joint_ids])
            return
        position, orientation = self._compute_frame_pose()
        ep, er = compute_pose_error(position, orientation, self._filtered_position, self._filtered_orientation,
                                    rot_error_type="axis_angle")
        jac = self._compute_frame_jacobian().clone()
        jac[:, 3:] *= self.orientation_weight
        # Cartesian feedback becomes a joint velocity, not an unscaled pose
        # jump. Damping stays continuous near singularities and joint stops.
        error = torch.cat((ep, er * self.orientation_weight), -1)
        joints = self._asset.data.joint_pos[:, self._joint_ids]
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        rest = self._asset.data.default_joint_pos[:, self._joint_ids]
        self._joint_velocity, self._joint_command = joint_servo_step(
            jac, error, joints, rest, limits, self._joint_velocity,
            self._joint_command, self._dt, self.response)

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
