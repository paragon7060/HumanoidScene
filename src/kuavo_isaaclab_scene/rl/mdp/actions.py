"""Bounded planar drive and incremental hands; no object attachment or pose servo."""

import torch
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from ...robots.base_drive_control import FloatingBaseDrive, FloatingBaseDriveCfg
from ...robots.gripper_runtime import InterpolatedJointPositionAction, InterpolatedJointPositionActionCfg
from .body_lock import FixedBody, ARM_JOINT_NAMES
from .settling import gate_actions


class PlanarDrive(ActionTerm):
    """Planar velocity command with kinematic or force-driven root motion."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros(self.num_envs, 3, device=self.device)
        self._velocity = torch.zeros_like(self._raw)
        self._scale = torch.tensor(cfg.velocity_limits, device=self.device)
        self._accel = torch.tensor(cfg.acceleration_limits, device=self.device)
        self._drive = None
        if cfg.dynamic:
            self._drive = FloatingBaseDrive(self._asset, cfg.drive, env)

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
        actions = gate_actions(self._env, actions)
        self._raw[:] = actions.clamp(-1, 1)
        delta = self._raw * self._scale - self._velocity
        limit = self._accel * self._env.step_dt
        self._velocity += delta.clamp(-limit, limit)
        # A masked command alone would only decelerate an existing velocity.
        # Initial waiting must hold the base completely, just like both hands.
        if self._env.cfg.task.reset_settle_seconds > 0 and not self._env.cfg.task.reset_bank:
            command = self._env.command_manager.get_term("workcell")
            self._velocity[~command.settling.ready] = 0

    def apply_actions(self):
        if self._drive is not None:
            self._drive.apply(self._velocity)
            return
        pose = self._asset.data.root_pose_w.clone()
        orientation = pose[:, 3:].clone()
        velocity = torch.zeros(self.num_envs, 3, device=self.device)
        velocity[:, :2] = self._velocity[:, :2]
        linear_world = quat_apply(orientation, velocity)
        pose[:, :3] += linear_world * self._env.physics_dt
        angle = self._velocity[:, 2] * self._env.physics_dt
        delta = torch.zeros_like(orientation)
        delta[:, 0], delta[:, 3] = (angle / 2).cos(), (angle / 2).sin()
        pose[:, 3:] = quat_mul(delta, orientation)
        self._asset.write_root_pose_to_sim(pose)
        # Pair the kinematic pose write with the matching root rigid-body
        # velocity. Without this, PhysX's contact/friction solve for anything
        # gripped by a downstream articulation link (the fingers) sees a
        # stale/near-zero root velocity between teleported poses, so ANY base
        # motion looks like a slip event at the grasp contact even though the
        # gripper never moved relative to the box. Pure arm motion never hit
        # this because it never touches the root pose.
        root_velocity = torch.zeros(self.num_envs, 6, device=self.device)
        root_velocity[:, :3] = linear_world
        root_velocity[:, 5] = self._velocity[:, 2]
        self._asset.write_root_velocity_to_sim(root_velocity)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._velocity[ids] = 0
        if self._drive is not None:
            self._drive.reset(env_ids)


@configclass
class PlanarDriveCfg(ActionTermCfg):
    class_type: type = PlanarDrive
    asset_name: str = "robot"
    velocity_limits: tuple[float, float, float] = (0.25, 0.25, 0.70)
    acceleration_limits: tuple[float, float, float] = (0.5, 0.5, 1.2)
    # Dynamic mode is opt-in: it changes the task physics and needs an
    # unfixed articulation root. Entry points set it through
    # robots/base_drive.py so teleop, RL and evaluation agree.
    dynamic: bool = False
    drive: FloatingBaseDriveCfg = FloatingBaseDriveCfg()


class JointDeltaTargets(JointPositionAction):
    """Integrate normalized joint deltas once per control step, then hold PD targets."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._targets = self._asset.data.joint_pos[:, self._joint_ids].clone()
        self._processed_actions[:] = self._targets

    def process_actions(self, actions):
        actions = gate_actions(self._env, actions)
        self._raw_actions[:] = actions.clamp(-1, 1)
        self._targets += self._raw_actions * self._scale
        # Captured initial poses may exceed the training margin. Preserve zero
        # hold for flap picking while keeping physical articulation limits.
        limits = (self._asset.data.joint_pos_limits if self._env.cfg.task.grasp_mode == "flap_top"
                  else self._asset.data.soft_joint_pos_limits)[:, self._joint_ids]
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
    """7 or 14 arm actions; physically lock inactive joints after reset."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        actual = tuple(self._asset.joint_names[i] for i in self._joint_ids)
        expected = (ARM_JOINT_NAMES if cfg.active_arm == "both" else
                    tuple(f"zarm_{cfg.active_arm[0]}{i}_joint" for i in range(1, 8)))
        if actual != expected:
            raise ValueError(f"arms-only action order must be {expected}.")
        inactive = "l" if cfg.active_arm == "right" else "r"
        extra = () if cfg.active_arm == "both" else (rf"zarm_{inactive}[1-7]_joint", rf"{inactive}_.*_joint")
        self.body_lock = FixedBody(self._asset, tolerance=cfg.body_lock_tolerance, extra_patterns=extra)

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
    active_arm: str = "both"


class IncrementalGripper(InterpolatedJointPositionAction):
    """Zero action preserves a grasp, including after loading a reset snapshot."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._signed_target = torch.ones(self.num_envs, 1, device=self.device)
        self._force_close_requested = torch.zeros_like(self._signed_target, dtype=torch.bool)

    def process_actions(self, actions):
        actions = gate_actions(self._env, actions)
        self._raw_actions[:] = actions.clamp(-1, 1)
        self._force_close_requested[:] = torch.where(self._raw_actions < 0, True,
            torch.where(self._raw_actions > 0, False, self._force_close_requested))
        self._signed_target.add_(self._raw_actions * self.cfg.delta_scale).clamp_(-1, 1)
        self._set_joint_targets(self._targets_from_signed(self._signed_target))

    def _force_closing(self):
        return self._force_close_requested

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[ids][:, self._joint_ids]
        direction = self._close_command - self._open_command
        closed = ((q - self._open_command) * direction).sum(-1) / direction.square().sum().clamp_min(1e-8)
        self._signed_target[ids, 0] = 1 - 2 * closed.clamp(0, 1)
        if self._position_mapping is not None:
            self._position_mapping.reset(env_ids, closed)
            self._signed_target[ids] = 1 - self._position_mapping.previous_percent[ids] / 50
        self._raw_actions[ids] = 0
        self._force_close_requested[ids] = self._signed_target[ids] < 0
        self._reset_joint_targets(q, env_ids)
        if self._force_drive is not None:
            self._force_drive.reset(env_ids)


@configclass
class IncrementalGripperCfg(InterpolatedJointPositionActionCfg):
    class_type: type = IncrementalGripper
    delta_scale: float = 0.12


class BinaryGripper(InterpolatedJointPositionAction):
    """Execute one binary hand command: 0=open and 1=close.

    RL policies still emit a scalar in their normal continuous action vector.
    Positive values become 1 and zero/negative values become 0, keeping the
    two choices balanced around a freshly initialized policy mean of zero.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._close_requested = torch.zeros(self.num_envs, 1, dtype=torch.bool, device=self.device)

    def _command_enabled(self):
        if self._env.cfg.task.reset_settle_seconds <= 0 or self._env.cfg.task.reset_bank:
            return torch.ones(self.num_envs, 1, dtype=torch.bool, device=self.device)
        command = self._env.command_manager.get_term("workcell")
        return command.settling.ready[:, None]

    def process_actions(self, actions):
        close = actions > 0
        enabled = self._command_enabled()
        self._close_requested[:] = torch.where(enabled, close, self._close_requested)
        self._raw_actions[:] = self._close_requested.float()
        signed = 1. - 2. * self._raw_actions
        if self._position_mapping is not None:
            # Do not let a gated reset environment silently rewrite its
            # hysteresis state to "open" while its physical pose is held.
            held = 1. - self._position_mapping.previous_percent / 50.
            signed = torch.where(enabled, signed, held)
        targets = self._targets_from_signed(signed)
        self._desired_actions[:] = torch.where(enabled, targets, self._desired_actions)
        if self._target_filter is None:
            self._processed_actions[:] = self._desired_actions

    def _force_closing(self):
        return self._close_requested

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._raw_actions[ids] = 0
        self._close_requested[ids] = False
        self._reset_joint_targets(q, env_ids)
        if self._position_mapping is not None:
            direction = self._close_command - self._open_command
            closed = ((q - self._open_command) * direction).sum(-1) / direction.square().sum().clamp_min(1e-8)
            self._position_mapping.reset(env_ids, closed)
        if self._force_drive is not None:
            self._force_drive.reset(env_ids)


@configclass
class BinaryGripperCfg(InterpolatedJointPositionActionCfg):
    class_type: type = BinaryGripper
    # Retained so existing experiment config files remain loadable. Binary
    # control intentionally ignores incremental target scale.
    delta_scale: float = 0.0
