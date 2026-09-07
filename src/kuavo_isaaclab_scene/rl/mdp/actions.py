"""Bounded planar drive and incremental hands; no object attachment or pose servo."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from ...robots.gripper_runtime import InterpolatedJointPositionAction, InterpolatedJointPositionActionCfg
from ...robots.gripper_action import interpolate_signed_gripper_action
from .body_lock import FixedBody, ARM_JOINT_NAMES
from .settling import gate_actions


class PlanarDrive(ActionTerm):
    """Fixed-root simulation abstraction, NOT a contact-driven locomotion policy."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity = torch.zeros_like(self._raw)
        self._scale = torch.tensor(cfg.velocity_limits, device=self.device)
        self._accel = torch.tensor(cfg.acceleration_limits, device=self.device)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._velocity

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)
        delta = self._raw * self._scale - self._velocity
        limit = self._accel * self._env.step_dt
        self._velocity += delta.clamp(-limit, limit)

    def apply_actions(self):
        pose = self._asset.data.root_pose_w.clone()
        velocity = torch.zeros(self.num_envs, 3, device=self.device)
        velocity[:, :2] = self._velocity[:, :2]
        pose[:, :3] += quat_apply(pose[:, 3:], velocity) * self._env.physics_dt
        angle = self._velocity[:, 2] * self._env.physics_dt
        delta = torch.zeros_like(pose[:, 3:])
        delta[:, 0], delta[:, 3] = (angle / 2).cos(), (angle / 2).sin()
        pose[:, 3:] = quat_mul(delta, pose[:, 3:])
        self._asset.write_root_pose_to_sim(pose)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._velocity[ids] = 0


@configclass
class PlanarDriveCfg(ActionTermCfg):
    class_type: type = PlanarDrive
    asset_name: str = "robot"
    velocity_limits: tuple[float, float, float] = (0.25, 0.25, 0.70)
    acceleration_limits: tuple[float, float, float] = (0.5, 0.5, 1.2)


class JointDeltaTargets(JointPositionAction):
    """Integrate normalized joint deltas once per control step, then hold PD targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._targets = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._processed_actions[:] = self._targets

    def process_actions(self, actions):
        self._raw_actions[:] = actions.clamp(-1, 1)
        self._targets += self._raw_actions * self._scale
        limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids]
        self._targets[:] = self._targets.clamp(limits[..., 0], limits[..., 1])
        self._processed_actions[:] = self._targets

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 0
        self._targets[ids] = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._processed_actions[ids] = self._targets[ids]


@configclass
class JointDeltaTargetsCfg(JointPositionActionCfg):
    class_type: type = JointDeltaTargets
    use_default_offset: bool = False


class ArmsOnlyJointTargets(JointDeltaTargets):
    """14 arm actions; latch the non-arm body after reset and hold at each substep."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        actual = tuple(self._asset.joint_names[i] for i in self._joint_ids)
        if actual != ARM_JOINT_NAMES:
            raise ValueError("arms-only action order must be left joints 1..7 then right joints 1..7.")
        self.body_lock = FixedBody(self._asset, tolerance=cfg.body_lock_tolerance)

    def process_actions(self, actions):
        actions = gate_actions(self._env, actions)
        self._raw_actions[:] = actions.clamp(-1, 1)
        self._targets += self._raw_actions * self._scale
        # VR measured poses can lie outside the softer training margin. Preserve
        # zero-action hold at those poses while respecting actual physical limits.
        limits = self._asset.data.joint_pos_limits[:, self._joint_ids]
        self._targets[:] = self._targets.clamp(limits[..., 0], limits[..., 1])
        self._processed_actions[:] = self._targets

    def reset(self, env_ids=None):
        super().reset(env_ids)
        # ManagerBasedRLEnv invokes this after all reset events, including the
        # named initial-state event, and before command observations are reset.
        self.body_lock.reset(env_ids)

    def apply_actions(self):
        self.body_lock.apply()
        super().apply_actions()


@configclass
class ArmsOnlyJointTargetsCfg(JointDeltaTargetsCfg):
    class_type: type = ArmsOnlyJointTargets
    body_lock_tolerance: float = 1e-4


class IncrementalGripper(InterpolatedJointPositionAction):
    """Zero action preserves a grasp, including after loading a reset snapshot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._signed_target = torch.ones(self.num_envs, 1, device=self.device)

    def process_actions(self, actions):
        actions = gate_actions(self._env, actions)
        self._raw_actions[:] = actions.clamp(-1, 1)
        self._signed_target.add_(self._raw_actions * self.cfg.delta_scale).clamp_(-1, 1)
        self._processed_actions[:] = interpolate_signed_gripper_action(
            self._signed_target, self._open_command, self._close_command)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[ids][:, self._joint_ids]
        direction = self._close_command - self._open_command
        closed = ((q - self._open_command) * direction).sum(-1) / direction.square().sum().clamp_min(1e-8)
        self._signed_target[ids, 0] = 1 - 2 * closed.clamp(0, 1)
        self._raw_actions[ids] = 0
        self._processed_actions[ids] = q


@configclass
class IncrementalGripperCfg(InterpolatedJointPositionActionCfg):
    class_type: type = IncrementalGripper
    delta_scale: float = 0.12
