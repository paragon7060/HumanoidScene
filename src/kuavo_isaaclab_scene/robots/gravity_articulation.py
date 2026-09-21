"""Isaac Lab adapter; import after AppLauncher starts Kit."""

import logging

import torch
from isaaclab.assets import Articulation

from .gravity_compensation import (
    feedforward_joint_ids, gravity_drive_bias, gravity_joint_ids, wbc_acceleration_profile)


class GravityCompensatedArticulation(Articulation):
    """Apply body/arm gravity feedforward once at every physics-step write.

    PhysX retains implicit PD and its force limits. Logical targets exposed to
    action managers, IK, observations and recordings are never modified.
    """

    gravity_compensation_enabled = True

    def _initialize_impl(self):
        super()._initialize_impl()
        from .robot_model import resolve_dynamics_profile
        requested_dynamics_profile = resolve_dynamics_profile()
        self._gravity_joint_ids = gravity_joint_ids(self.joint_names)
        if not self._gravity_joint_ids:
            raise ValueError("No supported Kuavo body/arm joints found for gravity compensation")
        implicit = set()
        for actuator in self.actuators.values():
            if actuator.is_implicit_model:
                ids = actuator.joint_indices
                implicit.update(range(self.num_joints) if isinstance(ids, slice)
                                else [int(i) for i in ids])
        if not set(self._gravity_joint_ids).issubset(implicit):
            raise ValueError("Gravity drive compensation requires implicit body/arm actuators")
        self.gravity_compensation_torque = torch.zeros_like(self.data.joint_pos_target)
        self.command_feedforward_torque = torch.zeros_like(self.data.joint_pos_target)
        self.command_feedforward_mask = torch.zeros_like(self.data.joint_pos_target, dtype=torch.bool)
        self.total_feedforward_torque = torch.zeros_like(self.data.joint_pos_target)
        # A floating articulation prepends six root coordinates to every
        # generalized quantity. The acceleration task below stays a joint-DOF
        # task on both roots; the floating case simply reads the joint block
        # of those arrays (see _apply_actuator_model).
        self.dynamics_profile = requested_dynamics_profile
        self.command_feedforward_mode = (
            "inverse_dynamics" if self.dynamics_profile.endswith("-id") else "off"
        )
        self.inverse_dynamics_torque = torch.zeros_like(self.data.joint_pos_target)
        gains = wbc_acceleration_profile(
            self.joint_names, device=self.device, dtype=self.data.joint_pos.dtype
        )
        self._wbc_accel_kp, self._wbc_accel_kd, self._wbc_accel_limit = (
            value.unsqueeze(0) for value in gains
        )
        if self.command_feedforward_mode == "inverse_dynamics":
            self.command_feedforward_mask[:, feedforward_joint_ids(
                self.joint_names, self.dynamics_profile, gains[0].tolist())] = True
        self._inverse_dynamics_update_pending = True
        self.gravity_compensation_bias = torch.zeros_like(self.data.joint_pos_target)
        logging.getLogger(__name__).info("Kuavo gravity compensation: %d body/arm joints; implicit drive force caps retained",
                                        len(self._gravity_joint_ids))
        feedforward = int(self.command_feedforward_mask[0].sum())
        print(f"[GRAVITY] ON: {len(self._gravity_joint_ids)} body/arm joints; "
              f"dynamics_profile={self.dynamics_profile}; "
              f"inverse dynamics on {feedforward} joints, gravity-PD on the rest; "
              "updated each physics write; locked joints excluded; "
              "logical targets and drive force caps retained.", flush=True)
        if not self.is_fixed_base:
            print("[GRAVITY] Floating root: joint feedforward uses the joint block; "
                  "the base drive independently holds root height and attitude.",
                  flush=True)

    def _apply_actuator_model(self):
        super()._apply_actuator_model()
        # Query the complete physical robot (including claw/cameras), then
        # select only driven body/arm DOFs. No object payload is attached here.
        gravity = self.root_physx_view.get_gravity_compensation_forces()
        # Floating articulations prepend six root generalized coordinates;
        # actuator targets and all buffers below contain joint DOFs only.
        # Derive this from the actuator buffer instead of an implementation
        # detail on Articulation so lightweight test doubles follow the same
        # path as the real simulation object.
        joint_count = self.data.joint_pos_target.shape[1]
        gravity = gravity[:, -joint_count:]
        ids = self._gravity_joint_ids
        self.gravity_compensation_torque.zero_()
        self.gravity_compensation_torque[:, ids] = gravity[:, ids]
        if self.command_feedforward_mode == "inverse_dynamics":
            if self._inverse_dynamics_update_pending:
                desired_acceleration = (
                    self._wbc_accel_kp * (self.data.joint_pos_target - self.data.joint_pos)
                    + self._wbc_accel_kd * (self.data.joint_vel_target - self.data.joint_vel)
                )
                desired_acceleration = torch.clamp(
                    desired_acceleration, -self._wbc_accel_limit, self._wbc_accel_limit
                )
                mass_full = self.root_physx_view.get_generalized_mass_matrices()
                coriolis_full = self.root_physx_view.get_coriolis_and_centrifugal_compensation_forces()
                if mass_full.shape[-1] != joint_count:
                    # Floating root: keep the joint block. That block is the
                    # joint-space inertia seen with the root held, which is the
                    # model used by the independently stabilized planar base.
                    # Do not turn desired joint acceleration into a root wrench:
                    # it is only a model command and can differ substantially
                    # from achieved acceleration under limits and contacts.
                    mass = mass_full[:, -joint_count:, -joint_count:]
                    coriolis = coriolis_full[:, -joint_count:]
                else:
                    mass = mass_full
                    coriolis = coriolis_full
                self.inverse_dynamics_torque[:] = (
                    torch.bmm(mass, desired_acceleration.unsqueeze(-1)).squeeze(-1)
                    + coriolis + gravity
                )
                self._inverse_dynamics_update_pending = False
            total = torch.where(
                self.command_feedforward_mask, self.inverse_dynamics_torque, gravity
            )
        elif self.command_feedforward_mode == "replace_gravity":
            total = torch.where(self.command_feedforward_mask,
                                self.command_feedforward_torque, gravity)
        elif self.command_feedforward_mode == "add_to_gravity":
            total = gravity + torch.where(self.command_feedforward_mask,
                                          self.command_feedforward_torque, 0.)
        else:
            total = gravity
        self.total_feedforward_torque[:] = total
        bias = gravity_drive_bias(total, self.data.joint_stiffness,
                                 self.data.joint_pos_limits, self._gravity_joint_ids)
        self.gravity_compensation_bias[:] = bias
        self._joint_pos_target_sim += bias
        # Match Isaac Lab's approximate actuator torque telemetry to the bias.
        # These quantities are estimates, not a measured motor output torque.
        self.data.computed_torque[:, ids] += self.total_feedforward_torque[:, ids]
        limits = self.data.joint_effort_limits[:, ids]
        self.data.applied_torque[:, ids] = self.data.computed_torque[:, ids].clamp(-limits, limits)

    def set_command_feedforward_torque(self, torque, mode="off", joint_mask=None):
        """Set diagnostic/model feedforward without bypassing the drive effort cap."""
        if mode not in ("off", "replace_gravity", "add_to_gravity", "inverse_dynamics"):
            raise ValueError(f"Unknown command feedforward mode: {mode}")
        if torque.shape != self.command_feedforward_torque.shape:
            raise ValueError("Command feedforward torque must match the articulation joint tensor")
        if not torch.isfinite(torque).all():
            raise ValueError("Non-finite command feedforward torque")
        if joint_mask is None:
            joint_mask = torch.ones_like(torque, dtype=torch.bool)
        if joint_mask.shape != torque.shape or joint_mask.dtype != torch.bool:
            raise ValueError("Command feedforward mask must be a matching bool tensor")
        self.command_feedforward_torque[:] = torque
        self.command_feedforward_mask[:] = joint_mask
        self.command_feedforward_mode = mode
        self._inverse_dynamics_update_pending = True

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        super().set_joint_position_target(target, joint_ids, env_ids)
        self._inverse_dynamics_update_pending = True

    def set_joint_velocity_target(self, target, joint_ids=None, env_ids=None):
        super().set_joint_velocity_target(target, joint_ids, env_ids)
        self._inverse_dynamics_update_pending = True

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if hasattr(self, "inverse_dynamics_torque"):
            ids = slice(None) if env_ids is None else env_ids
            self.inverse_dynamics_torque[ids] = 0
            self._inverse_dynamics_update_pending = True
