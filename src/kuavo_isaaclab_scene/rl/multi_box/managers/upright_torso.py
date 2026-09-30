"""Two-axis upright torso action for the S63 multi-box grasp policy."""

from __future__ import annotations

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ...mdp.settling import gate_actions
from ....robots.robot_model import resolve_robot_model
from ..geometry.upright_torso import (
    TORSO_FORWARD_LIMIT_M,
    TORSO_HEIGHT_RANGE_M,
    TORSO_JOINT_NAMES,
    TORSO_XZ_SPEED_M_S,
    planar_position,
    torso_links_from_urdf,
    upright_joint_step,
)


class UprightTorsoAction(ActionTerm):
    """Integrate X/Z commands while commanding constant torso pitch."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._joint_ids, names = self._asset.find_joints(
            list(TORSO_JOINT_NAMES), preserve_order=True)
        if tuple(names) != TORSO_JOINT_NAMES:
            raise ValueError(f"Upright torso needs {TORSO_JOINT_NAMES}; got {names}")
        self._links = torch.tensor(
            torso_links_from_urdf(resolve_robot_model("s63", "leju-twofinger").urdf_path),
            device=self.device, dtype=self._asset.data.joint_pos.dtype,
        )
        self._raw = torch.zeros((self.num_envs, 2), device=self.device)
        self._joint_targets = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._target_xz = planar_position(self._joint_targets[:, :2], self._links)
        self._origin_xz = self._target_xz.clone()
        self._pitch_reference = self._joint_targets.sum(-1)

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._target_xz

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = gate_actions(self._env, actions).clamp(-1, 1)
        desired = self._target_xz + self._raw * (self.cfg.speed_m_s * self._env.step_dt)
        desired[:, 0] = desired[:, 0].clamp(
            self._origin_xz[:, 0] - self.cfg.forward_limit_m,
            self._origin_xz[:, 0] + self.cfg.forward_limit_m,
        )
        nominal_z = self._links[:, 1].sum()
        desired[:, 1] = desired[:, 1].clamp(
            nominal_z + self.cfg.height_range_m[0],
            nominal_z + self.cfg.height_range_m[1],
        )
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._joint_targets[:], self._target_xz[:] = upright_joint_step(
            self._joint_targets, desired, self._pitch_reference, self._links, limits)

    def apply_actions(self) -> None:
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids=None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._joint_targets[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._target_xz[ids] = planar_position(self._joint_targets[ids, :2], self._links)
        self._origin_xz[ids] = self._target_xz[ids]
        self._pitch_reference[ids] = self._joint_targets[ids].sum(-1)


@configclass
class UprightTorsoActionCfg(ActionTermCfg):
    class_type: type = UprightTorsoAction
    asset_name: str = "robot"
    speed_m_s: float = TORSO_XZ_SPEED_M_S
    forward_limit_m: float = TORSO_FORWARD_LIMIT_M
    height_range_m: tuple[float, float] = TORSO_HEIGHT_RANGE_M
