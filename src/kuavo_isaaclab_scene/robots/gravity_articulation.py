"""Isaac Lab adapter; import after AppLauncher starts Kit."""

import logging

import torch
from isaaclab.assets import Articulation

from .gravity_compensation import gravity_drive_bias, gravity_joint_ids


class GravityCompensatedArticulation(Articulation):
    """Apply body/arm gravity feedforward once at every physics-step write.

    PhysX retains implicit PD and its force limits. Logical targets exposed to
    action managers, IK, observations and recordings are never modified.
    """

    gravity_compensation_enabled = True

    def _initialize_impl(self):
        super()._initialize_impl()
        if not self.is_fixed_base:
            raise ValueError("Kuavo gravity drive compensation requires a fixed articulation root; floating-base contact dynamics need a separate controller")
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
        self.gravity_compensation_bias = torch.zeros_like(self.data.joint_pos_target)
        logging.getLogger(__name__).info("Kuavo gravity compensation: %d body/arm joints; implicit drive force caps retained",
                                        len(self._gravity_joint_ids))
        print(f"[GRAVITY] ON: {len(self._gravity_joint_ids)} body/arm joints; "
              "updated each physics write; locked joints excluded; "
              "logical targets and drive force caps retained.", flush=True)

    def _apply_actuator_model(self):
        super()._apply_actuator_model()
        # Query the complete physical robot (including claw/cameras), then
        # select only driven body/arm DOFs. No object payload is attached here.
        gravity = self.root_physx_view.get_gravity_compensation_forces()
        bias = gravity_drive_bias(gravity, self.data.joint_stiffness,
                                 self.data.joint_pos_limits, self._gravity_joint_ids)
        self.gravity_compensation_bias[:] = bias
        self.gravity_compensation_torque[:] = bias * self.data.joint_stiffness
        self._joint_pos_target_sim += bias
        # Match Isaac Lab's approximate actuator torque telemetry to the bias.
        # These quantities are estimates, not a measured motor output torque.
        ids = self._gravity_joint_ids
        self.data.computed_torque[:, ids] += self.gravity_compensation_torque[:, ids]
        limits = self.data.joint_effort_limits[:, ids]
        self.data.applied_torque[:, ids] = self.data.computed_torque[:, ids].clamp(-limits, limits)
