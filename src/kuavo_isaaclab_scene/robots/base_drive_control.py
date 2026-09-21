"""Floating-base planar drive: one PD wrench controller for teleop and RL."""

import torch
from isaaclab.utils import configclass
from isaaclab.utils.math import (axis_angle_from_quat, euler_xyz_from_quat,
                                 quat_apply_inverse, quat_conjugate, quat_mul, wrap_to_pi)

from .base_drive import (WHEEL_ANGLES_RAD, WHEEL_JOINTS, WHEEL_OFFSET_M, WHEEL_RADIUS_M)


@configclass
class FloatingBaseDriveCfg:
    """Tuning for the wrench-driven chassis.

    The stiffness/damping pairs are acceleration gains, so they read directly as
    a natural frequency and damping ratio and stay valid when the mass or the
    inertia estimate changes.
    """

    position_stiffness: float = 60.0
    velocity_damping: float = 16.0
    yaw_stiffness: float = 40.0
    yaw_damping: float = 13.0
    max_position_error: float = 0.12
    max_linear_acceleration: float = 2.0
    max_yaw_acceleration: float = 4.0
    height_stiffness: float = 120.0
    height_damping: float = 22.0
    max_vertical_acceleration: float = 15.0
    tilt_stiffness: float = 120.0
    tilt_damping: float = 22.0
    max_tilt_acceleration: float = 10.0
    # Lower bounds only. The measured values come from the live link layout.
    yaw_inertia_radius: float = 0.35
    tilt_inertia_radius: float = 0.30


class FloatingBaseDrive:
    """Suspend and move an unfixed articulation root with a PD wrench.

    The chassis carries no ground contact: the imported wheels are plain
    cylinders rather than the real omni rollers, and on a floating root their
    friction cone simply welds a 200 kg robot to the floor. This controller
    holds the chassis up and level, and moves it to a target integrated from
    the same planar velocity command the kinematic base uses, so every link
    below the root keeps contact-correct dynamics.
    """

    def __init__(self, asset, cfg: FloatingBaseDriveCfg, env):
        if asset.is_fixed_base:
            raise ValueError(
                "A dynamic base requires scene.robot.spawn.articulation_props."
                "fix_root_link=False; a fixed root would accept the wrench and never move."
            )
        self._asset, self.cfg, self._env = asset, cfg, env
        device, num_envs = asset.device, asset.num_instances
        self._device, self._num_envs = device, num_envs
        self._target_xy = torch.zeros(num_envs, 2, device=device)
        self._target_yaw = torch.zeros(num_envs, device=device)
        self._target_height = torch.zeros(num_envs, device=device)
        # Residual roll/pitch of the spawn pose, with yaw removed, so the
        # attitude target can be rebuilt for any commanded yaw.
        self._level_quat = torch.zeros(num_envs, 4, device=device)
        self._level_quat[:, 0] = 1.0
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._root_body_id = torch.tensor([0], dtype=torch.long, device=device)
        self._force_b = torch.zeros(num_envs, 1, 3, device=device)
        self._torque_b = torch.zeros_like(self._force_b)
        self._body_mass = asset.data.default_mass.to(device)
        self._total_mass = self._body_mass.sum(dim=1).clamp_min(1.0)
        gravity = getattr(env.cfg.sim, "gravity", (0.0, 0.0, -9.81))
        self._gravity = torch.tensor(gravity, device=device).expand(num_envs, 3)
        # Summed local diagonals only. The parallel-axis terms dominate for a
        # chassis whose mass sits a metre above the root, and they are added
        # from live link positions in apply().
        body_inertia = asset.data.default_inertia.to(device)
        self._local_inertia = torch.stack(
            (body_inertia[..., 0].sum(dim=1),
             body_inertia[..., 4].sum(dim=1),
             body_inertia[..., 8].sum(dim=1)), dim=-1)
        self._inertia_floor = self._total_mass[:, None] * torch.tensor(
            [cfg.tilt_inertia_radius ** 2, cfg.tilt_inertia_radius ** 2,
             cfg.yaw_inertia_radius ** 2], device=device)
        self._wheel_ids, wheel_names = asset.find_joints(list(WHEEL_JOINTS), preserve_order=True)
        if len(self._wheel_ids) != len(WHEEL_JOINTS):
            raise ValueError(f"A dynamic base requires four wheel joints, found {wheel_names}.")
        angles = torch.tensor(WHEEL_ANGLES_RAD, device=device)
        self._wheel_tangents = torch.stack((angles.sin(), -angles.cos()), dim=-1)

    def reset(self, env_ids=None) -> None:
        """Re-capture the held pose after the environment moves the root."""
        self._initialized[slice(None) if env_ids is None else env_ids] = False

    def _yaw_quat(self, yaw):
        quat = torch.zeros(*yaw.shape, 4, device=self._device)
        quat[..., 0] = (yaw / 2).cos()
        quat[..., 3] = (yaw / 2).sin()
        return quat

    def _inertia(self):
        """Inertia about the root in world axes, parallel-axis terms included.

        Recomputed from live link positions so an extended arm raises the
        commanded torque instead of quietly lowering the effective gain.
        """
        root_com = self._asset.data.root_com_pos_w
        offsets = self._asset.data.body_com_pos_w - root_com.unsqueeze(1)
        squares = self._body_mass.unsqueeze(-1) * offsets.square()
        moments = torch.stack(
            (squares[..., 1] + squares[..., 2],
             squares[..., 0] + squares[..., 2],
             squares[..., 0] + squares[..., 1]), dim=-1).sum(dim=1)
        return torch.maximum(self._local_inertia + moments, self._inertia_floor)

    def apply(self, velocity) -> None:
        """Track one planar velocity command (vx, vy, yaw rate) with a wrench."""
        cfg = self.cfg
        position = self._asset.data.root_pos_w
        orientation = self._asset.data.root_quat_w
        yaw = euler_xyz_from_quat(orientation)[2]
        initialize = ~self._initialized
        self._target_xy[initialize] = position[initialize, :2]
        self._target_yaw[initialize] = yaw[initialize]
        self._target_height[initialize] = position[initialize, 2]
        self._level_quat[initialize] = quat_mul(
            quat_conjugate(self._yaw_quat(yaw[initialize])), orientation[initialize])
        self._initialized[initialize] = True

        dt = self._env.physics_dt
        cos_yaw, sin_yaw = self._target_yaw.cos(), self._target_yaw.sin()
        local_x, local_y = velocity[:, 0], velocity[:, 1]
        desired_velocity_w = torch.stack(
            (cos_yaw * local_x - sin_yaw * local_y,
             sin_yaw * local_x + cos_yaw * local_y), dim=-1)
        self._target_xy += desired_velocity_w * dt
        self._target_yaw = wrap_to_pi(self._target_yaw + velocity[:, 2] * dt)

        position_error = self._target_xy - position[:, :2]
        error_norm = position_error.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        position_error *= (cfg.max_position_error / error_norm).clamp_max(1.0)
        acceleration = (
            cfg.position_stiffness * position_error
            + cfg.velocity_damping * (desired_velocity_w - self._asset.data.root_lin_vel_w[:, :2])
        )
        acceleration_norm = acceleration.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        acceleration *= (cfg.max_linear_acceleration / acceleration_norm).clamp_max(1.0)

        # Attitude error as a world-frame rotation vector. Reading roll/pitch
        # from an Euler decomposition and applying the correction about world
        # x/y only agrees while the base faces its spawn direction: those Euler
        # angles are measured about body axes, so after a quarter turn the roll
        # correction lands on the pitch axis and the loop pumps the chassis
        # instead of damping it. A rotation-vector error is expressed in the
        # same world frame as the torque, at every yaw.
        attitude_error = axis_angle_from_quat(
            quat_mul(quat_mul(self._yaw_quat(self._target_yaw), self._level_quat),
                     quat_conjugate(orientation)))
        yaw_acceleration = (
            cfg.yaw_stiffness * attitude_error[:, 2]
            + cfg.yaw_damping * (velocity[:, 2] - self._asset.data.root_ang_vel_w[:, 2])
        ).clamp(-cfg.max_yaw_acceleration, cfg.max_yaw_acceleration)
        tilt_acceleration = (
            cfg.tilt_stiffness * attitude_error[:, :2]
            - cfg.tilt_damping * self._asset.data.root_ang_vel_w[:, :2]
        ).clamp(-cfg.max_tilt_acceleration, cfg.max_tilt_acceleration)
        vertical_acceleration = (
            cfg.height_stiffness * (self._target_height - position[:, 2])
            - cfg.height_damping * self._asset.data.root_lin_vel_w[:, 2]
        ).clamp(-cfg.max_vertical_acceleration, cfg.max_vertical_acceleration)

        inertia = self._inertia()
        force_w = torch.zeros(self._num_envs, 3, device=self._device)
        torque_w = torch.zeros_like(force_w)
        force_w[:, :2] = self._total_mass[:, None] * acceleration
        force_w[:, 2] = self._total_mass * vertical_acceleration
        torque_w[:, :2] = inertia[:, :2] * tilt_acceleration
        torque_w[:, 2] = inertia[:, 2] * yaw_acceleration
        whole_body_com = (
            self._body_mass[:, :, None] * self._asset.data.body_com_pos_w
        ).sum(dim=1) / self._total_mass[:, None]
        lever = whole_body_com - self._asset.data.root_com_pos_w
        # This force is applied on the root body, below the whole-body CoM.
        # Add its equivalent couple so accelerating a raised torso translates
        # the robot instead of pitching it.
        torque_w += torch.cross(lever, force_w, dim=-1)

        # Cancel weight at the whole-body CoM.  Commanded joint acceleration is
        # deliberately not converted to a root feedforward wrench: contacts,
        # actuator limits and the implicit drive can make achieved acceleration
        # differ from that prediction, which pumps the floating base instead of
        # damping it.  The attitude loop rejects the actual joint reaction.
        weight_w = -self._total_mass[:, None] * self._gravity
        force_w += weight_w
        torque_w += torch.cross(lever, weight_w, dim=-1)
        # WrenchComposer caches link poses for permanent wrenches. Convert the
        # current world-frame command here so a changing root yaw does not make
        # that cache rotate the force with an old orientation.
        self._force_b[:, 0] = quat_apply_inverse(orientation, force_w)
        self._torque_b[:, 0] = quat_apply_inverse(orientation, torque_w)
        self._asset.permanent_wrench_composer.set_forces_and_torques(
            forces=self._force_b,
            torques=self._torque_b,
            body_ids=self._root_body_id,
            is_global=False,
        )
        # Visual wheel spin only; these wheels no longer touch the floor.
        self._asset.set_joint_velocity_target(
            (velocity[:, :2] @ self._wheel_tangents.T - WHEEL_OFFSET_M * velocity[:, 2:3])
            / WHEEL_RADIUS_M,
            joint_ids=self._wheel_ids,
        )
