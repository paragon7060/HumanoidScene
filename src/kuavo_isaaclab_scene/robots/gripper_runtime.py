"""Isaac Lab runtime integration for configurable wrist grippers.

Each hand remains an independently addressable articulation. A USD fixed
joint marked ``excludeFromArticulation`` attaches it to the corresponding
Kuavo wrist without merging the two gripper joint-name namespaces.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import Callable

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdPhysics
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.string as string_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

from .gripper_config import GripperSettings
from .gripper_action import DirectionalGripperMapping, GripperTargetFilter, interpolate_signed_gripper_action


_MOUNTED_GRIPPERS: dict[str, Articulation] = {}


class InterpolatedJointPositionAction(ActionTerm):
    """One signed scalar continuously controlling a multi-joint gripper pose."""

    def __init__(self, cfg: "InterpolatedJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._joint_ids, self._joint_names = self._asset.find_joints(
            cfg.joint_names, preserve_order=True
        )
        self._raw_actions = torch.zeros(self.num_envs, 1, device=self.device)
        self._processed_actions = torch.zeros(
            self.num_envs, len(self._joint_ids), device=self.device
        )
        self._open_command = self._resolve_command(cfg.open_command_expr, "open")
        self._close_command = self._resolve_command(cfg.close_command_expr, "close")
        self._position_mapping = (DirectionalGripperMapping(cfg.position_mapping, self.num_envs, self.device)
                                  if cfg.position_mapping is not None else None)
        self._desired_actions = self._processed_actions.clone()
        self._target_filter = None
        self._force_drive = None
        if cfg.target_filter is not None:
            initial = self._asset.data.joint_pos[:, self._joint_ids]
            self._target_filter = GripperTargetFilter(initial, self._close_command-self._open_command, cfg.target_filter)
            self._desired_actions[:] = initial
            self._processed_actions[:] = initial
            self._filter_dt = sim_utils.SimulationContext.instance().get_physics_dt()
        if cfg.close_force_n is not None:
            self._force_drive = GripperForceDrive(self, env)

    def _set_joint_targets(self, targets):
        self._desired_actions[:] = targets
        if self._target_filter is None:
            self._processed_actions[:] = targets

    def _reset_joint_targets(self, targets, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._desired_actions[ids] = targets
        self._processed_actions[ids] = targets
        if self._target_filter is not None:
            self._target_filter.reset(targets, env_ids)

    def _targets_from_signed(self, actions):
        if self._position_mapping is None:
            return interpolate_signed_gripper_action(actions, self._open_command, self._close_command)
        fraction = self._position_mapping.process(actions)
        return self._open_command + fraction * (self._close_command - self._open_command)

    def _resolve_command(self, expressions: dict[str, float], label: str) -> torch.Tensor:
        command = torch.zeros(len(self._joint_ids), device=self.device)
        indices, names, values = string_utils.resolve_matching_names_values(
            expressions, self._joint_names
        )
        if len(indices) != len(self._joint_ids):
            missing = set(self._joint_names) - set(names)
            raise ValueError(f"Could not resolve {label} command for gripper joints: {missing}")
        command[indices] = torch.tensor(values, device=self.device)
        return command

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions[:] = actions
        self._set_joint_targets(self._targets_from_signed(self._raw_actions))

    def apply_actions(self) -> None:
        if self._target_filter is not None:
            self._processed_actions[:] = self._target_filter.advance(self._desired_actions, self._filter_dt)
        self._asset.set_joint_position_target(
            self._processed_actions, joint_ids=self._joint_ids
        )
        if self._force_drive is not None:
            self._force_drive.apply(self._force_closing())

    def _force_closing(self):
        return self._raw_actions < 0

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        ids = slice(None) if env_ids is None else env_ids
        self._raw_actions[ids] = 1.0
        if self._target_filter is None:
            self._reset_joint_targets(self._open_command, env_ids)
        else:
            q = self._asset.data.joint_pos[ids][:, self._joint_ids]
            self._reset_joint_targets(q, env_ids)
            self._desired_actions[ids] = self._open_command
        if self._position_mapping is not None:
            self._position_mapping.reset(env_ids)
        if self._force_drive is not None:
            self._force_drive.reset(env_ids)


@configclass
class InterpolatedJointPositionActionCfg(ActionTermCfg):
    class_type: type = InterpolatedJointPositionAction
    joint_names: list[str] = MISSING
    open_command_expr: dict[str, float] = MISSING
    close_command_expr: dict[str, float] = MISSING
    position_mapping: dict | None = None
    target_filter: dict | None = None
    close_force_n: float | None = None
    force_side: str | None = None
    force_sensor_names: tuple[str, str] | None = None


class ForceBinaryGripperAction(InterpolatedJointPositionAction):
    """VR open/close intent; both directions use force from command onset."""

    def process_actions(self, actions):
        super().process_actions(torch.where(actions < 0, -1., 1.))


@configclass
class ForceBinaryGripperActionCfg(InterpolatedJointPositionActionCfg):
    class_type: type = ForceBinaryGripperAction


class GripperForceDrive:
    """Explicit force feedforward with zero stiffness and implicit viscosity.

    Split the original torque budget equally between external torque and the
    damping drive. Their combined magnitude cannot exceed the original cap.
    """

    def __init__(self, action, env):
        from .gripper_force import JawForceServo

        self.action, self.env, self.asset = action, env, action._asset
        self.ids = action._joint_ids
        drive = self.asset.cfg.spawn.joint_drive_props
        if drive is None or drive.drive_type != "force":
            raise ValueError("Normal-force control requires PhysX force drives, not acceleration drives")
        if len(self.ids) != 2 or action.cfg.force_side not in ("left", "right"):
            raise ValueError("VR force control requires the two closed-claw drivers")
        self.stiffness = self.asset.data.joint_stiffness[:, self.ids].clone()
        self.damping = self.asset.data.joint_damping[:, self.ids].clone()
        self.effort_limits = self.asset.data.joint_effort_limits[:, self.ids].clone()
        self.joint_limits = self.asset.data.joint_pos_limits[:, self.ids].clone()
        direction = (action._close_command - action._open_command).sign()
        from .twofinger_linkage import DRIVER_OPEN_MIN
        bounds = torch.stack((torch.where(direction > 0, DRIVER_OPEN_MIN, 0.),
                              torch.where(direction > 0, 0., -DRIVER_OPEN_MIN)), -1)
        self.force_limits = self.joint_limits.clone()
        self.force_limits[..., 0] = torch.maximum(self.force_limits[..., 0], bounds[:, 0])
        self.force_limits[..., 1] = torch.minimum(self.force_limits[..., 1], bounds[:, 1])
        if (self.force_limits[..., 0] >= self.force_limits[..., 1]).any():
            raise ValueError("Claw force mode conflicts with live joint limits")
        self.servo = JawForceServo(action.cfg.force_side, env.num_envs, env.device,
                                  action.cfg.close_force_n, direction)
        self.enabled = torch.zeros(env.num_envs, 1, dtype=torch.bool, device=env.device)
        self.engaged = torch.zeros_like(self.enabled)
        self.dt = sim_utils.SimulationContext.instance().get_physics_dt()
        prefix = action.cfg.force_side[0]
        self.base_id = self.asset.find_bodies(f"{prefix}_twofinger_base")[0][0]
        by_name = dict(zip((f"{prefix}_f_bar_1_joint", f"{prefix}_b_bar_1_joint"), action.cfg.force_sensor_names))
        self.sensors = [env.scene[by_name[name]] for name in action._joint_names]
        self.measured_force = torch.zeros(env.num_envs, 2, device=env.device)
        print(f"[GRIP FORCE] {action.cfg.force_side}: close={action.cfg.close_force_n:g} N total "
              "(half per jaw), from command onset; open=same force reversed; original torque budget retained.", flush=True)

    def _mode(self, engaged, env_ids):
        stiffness = torch.where(engaged, 0., self.stiffness[env_ids])
        damping = torch.where(engaged, self.servo.damping, self.damping[env_ids])
        self.asset.write_joint_stiffness_to_sim(
            stiffness, joint_ids=self.ids, env_ids=env_ids)
        self.asset.write_joint_damping_to_sim(
            damping, joint_ids=self.ids, env_ids=env_ids)
        self.asset.write_joint_effort_limit_to_sim(
            torch.where(engaged, self.effort_limits[env_ids] * .5, self.effort_limits[env_ids]),
            joint_ids=self.ids, env_ids=env_ids)
        # Constant torque must stop at the CAD's closed/open configuration;
        # the donor's wider URDF limits would let the jaws cross through q=0.
        self.asset.write_joint_position_limit_to_sim(
            torch.where(engaged[..., None], self.force_limits[env_ids], self.joint_limits[env_ids]),
            joint_ids=self.ids, env_ids=env_ids)
        # Isaac's gain writers update PhysX/data, not actuator model buffers.
        # Keep implicit torque telemetry consistent with the live drive gains.
        for actuator in self.asset.actuators.values():
            indices = (list(range(self.asset.num_joints)) if isinstance(actuator.joint_indices, slice)
                       else list(actuator.joint_indices))
            for column, joint in enumerate(self.ids):
                if joint in indices:
                    local = indices.index(joint)
                    actuator.stiffness[env_ids, local] = stiffness[:, column]
                    actuator.damping[env_ids, local] = damping[:, column]

    def apply(self, closing):
        from isaaclab.utils.math import quat_apply
        changed = torch.nonzero((~self.engaged).flatten(), as_tuple=False).flatten()
        if changed.numel():
            self._mode(torch.ones(len(changed), 1, dtype=torch.bool, device=self.env.device), changed)
            self.engaged[changed] = True
        self.enabled[:] = closing
        axis = torch.zeros(self.env.num_envs, 3, device=self.env.device)
        axis[:, 0] = 1
        axis = quat_apply(self.asset.data.body_link_quat_w[:, self.base_id], axis)
        for index, sensor in enumerate(self.sensors):
            # Box bodies only: contacts with the other jaw, robot or rack must
            # not masquerade as object squeeze force.
            # Regulate squeeze along the closing axis. A top-edge contact that
            # merely supports finger weight is not jaw pressure.
            self.measured_force[:, index] = (sensor.data.force_matrix_w[:, 0]
                * axis[:, None]).sum(-1).abs().sum(-1)
        self.servo.advance(self.asset.data.joint_pos[:, self.ids],
            self.measured_force, closing, self.dt, self.effort_limits * .5,
            self.action._open_command)
        self.asset.set_joint_velocity_target(torch.zeros_like(self.servo.torque_request), joint_ids=self.ids)
        self.asset.set_joint_effort_target(self.servo.torque_request, joint_ids=self.ids)

    def reset(self, env_ids=None):
        ids = (torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None
               else torch.as_tensor(env_ids, device=self.env.device))
        self._mode(torch.zeros(len(ids), 1, dtype=torch.bool, device=self.env.device), ids)
        self.enabled[ids] = False
        self.engaged[ids] = False
        self.measured_force[ids] = 0
        self.servo.reset(ids)
        zeros = torch.zeros(len(ids), 2, device=self.env.device)
        self.asset.set_joint_velocity_target(zeros, joint_ids=self.ids, env_ids=ids)
        self.asset.set_joint_effort_target(zeros, joint_ids=self.ids, env_ids=ids)


class FilteredBinaryJointPositionAction(mdp.BinaryJointPositionAction):
    """Keep Isaac's binary sign/bool convention and filter only the PD target."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        initial = self._asset.data.joint_pos[:, self._joint_ids]
        self._desired_actions = initial.clone()
        self._processed_actions[:] = initial
        self._target_filter = GripperTargetFilter(initial, self._close_command-self._open_command, cfg.target_filter)
        self._filter_dt = sim_utils.SimulationContext.instance().get_physics_dt()

    def process_actions(self, actions):
        super().process_actions(actions)
        self._desired_actions[:] = self._processed_actions
        self._processed_actions[:] = self._target_filter.current

    def apply_actions(self):
        self._processed_actions[:] = self._target_filter.advance(self._desired_actions, self._filter_dt)
        super().apply_actions()

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        q = self._asset.data.joint_pos[ids][:, self._joint_ids]
        self._desired_actions[ids] = q
        self._processed_actions[ids] = q
        self._target_filter.reset(q, env_ids)


@configclass
class FilteredBinaryJointPositionActionCfg(mdp.BinaryJointPositionActionCfg):
    class_type: type = FilteredBinaryJointPositionAction
    target_filter: dict = MISSING


class MountedGripper(Articulation):
    """Articulation that exposes its pre-physics config to the mount spawner."""

    def __init__(self, cfg: ArticulationCfg):
        super().__init__(cfg)
        side = cfg.prim_path.rsplit("/", 1)[-1].lower()
        _MOUNTED_GRIPPERS[side] = self


@configclass
class MountedGripperCfg(ArticulationCfg):
    class_type: type = MountedGripper


def spawn_gripper_group(
    prim_path: str,
    cfg: "GripperGroupSpawnerCfg",
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Create the shared Xform parent before child gripper articulations spawn."""
    del cfg, kwargs
    stage = sim_utils.get_current_stage()
    parent_expression, leaf = prim_path.rsplit("/", 1)
    env_paths = sim_utils.find_matching_prim_paths(parent_expression)
    if not env_paths:
        raise RuntimeError(f"No environment prims match {parent_expression!r}.")

    first_group = None
    for env_path in env_paths:
        group_path = f"{env_path}/{leaf}"
        group = stage.GetPrimAtPath(group_path)
        if not group.IsValid():
            group = sim_utils.create_prim(
                group_path,
                "Xform",
                translation=translation,
                orientation=orientation,
                stage=stage,
            )
        first_group = first_group or group
    assert first_group is not None
    return first_group


@configclass
class GripperGroupSpawnerCfg(SpawnerCfg):
    func: Callable = spawn_gripper_group


def build_gripper_group_cfg(settings: GripperSettings) -> AssetBaseCfg | None:
    if not settings.active_sides or settings.integrated:
        return None
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Grippers",
        spawn=GripperGroupSpawnerCfg(),
    )


def _expand_isaac_tokens(path: str) -> str:
    return path.replace("${ISAAC_NUCLEUS_DIR}", ISAAC_NUCLEUS_DIR).replace(
        "${NUCLEUS_ASSET_ROOT_DIR}", NUCLEUS_ASSET_ROOT_DIR
    )


def build_gripper_articulation_cfg(
    settings: GripperSettings,
    side: str,
) -> MountedGripperCfg | None:
    if side not in settings.active_sides or settings.integrated:
        return None
    return MountedGripperCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Grippers/{side.title()}",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_expand_isaac_tokens(settings.usd_path_for(side)),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                enable_gyroscopic_forces=False,
                angular_damping=0.01,
                max_linear_velocity=25.0,
                max_angular_velocity=100.0,
                max_depenetration_velocity=2.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                # The challenge claw is a four-bar linkage represented as a
                # tree articulation in PhysX. Internal collisions would fight
                # the synchronized joint targets; object contacts stay active.
                enabled_self_collisions=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=2,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
                fix_root_link=False,
            ),
        ),
        # The mount spawner replaces this pose before physics initialization.
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=dict(settings.default_joint_pos),
            joint_vel={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=list(settings.joint_names),
                effort_limit_sim=settings.actuator.effort_limit_sim,
                stiffness=settings.actuator.stiffness,
                damping=settings.actuator.damping,
                friction=settings.actuator.friction,
            )
        },
        soft_joint_pos_limit_factor=1.0,
    )


def build_gripper_action_cfg(
    settings: GripperSettings,
    side: str,
    *,
    continuous: bool = False,
):
    if side not in settings.active_sides:
        return None
    filter_settings = settings.sides[side].target_filter
    cfg_type = (
        InterpolatedJointPositionActionCfg
        if continuous
        else (FilteredBinaryJointPositionActionCfg if filter_settings is not None else mdp.BinaryJointPositionActionCfg)
    )
    mapping_kwargs = ({"position_mapping": settings.sides[side].position_mapping}
                      if cfg_type is InterpolatedJointPositionActionCfg else {})
    if filter_settings is not None:
        mapping_kwargs["target_filter"] = filter_settings
    return cfg_type(
        asset_name=settings.asset_name_for(side),
        joint_names=list(settings.joint_names_for(side)),
        open_command_expr=settings.command_for(side, settings.open_command),
        close_command_expr=settings.command_for(side, settings.close_command),
        **mapping_kwargs,
    )


def _matrix_pose(matrix: Gf.Matrix4d) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    translation = matrix.ExtractTranslation()
    quaternion = matrix.ExtractRotationQuat()
    imaginary = quaternion.GetImaginary()
    return (
        (float(translation[0]), float(translation[1]), float(translation[2])),
        (
            float(quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _disable_world_root_joints(root_prim: Usd.Prim) -> None:
    for prim in Usd.PrimRange(root_prim):
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        if not joint.GetBody0Rel().GetTargets() or not joint.GetBody1Rel().GetTargets():
            joint.CreateJointEnabledAttr().Set(False)


def _align_and_attach(
    stage: Usd.Stage,
    env_path: str,
    group_path: str,
    settings: GripperSettings,
    side: str,
    *,
    update_default: bool,
) -> None:
    side_cfg = settings.sides[side]
    robot_root_path = f"{env_path}/Kuavo"
    base_body_path = f"{robot_root_path}/{side_cfg.robot_mount_body}"
    hand_root_path = f"{env_path}/Grippers/{side.title()}"
    hand_mount_path = f"{hand_root_path}/{settings.attachment_mount_body_for(side)}"
    base_body = stage.GetPrimAtPath(base_body_path)
    hand_root = stage.GetPrimAtPath(hand_root_path)
    hand_mount = stage.GetPrimAtPath(hand_mount_path)
    missing = [
        path
        for path, prim in ((base_body_path, base_body), (hand_root_path, hand_root), (hand_mount_path, hand_mount))
        if not prim.IsValid()
    ]
    if missing:
        raise RuntimeError(
            f"Cannot attach {side} gripper; missing USD prim(s): {', '.join(missing)}. "
            "Check robot_mount_body and attachment_mount_body in grippers.json."
        )

    base_mount_path = f"{base_body_path}/KuavoGripperMount"
    if not stage.GetPrimAtPath(base_mount_path).IsValid():
        sim_utils.create_prim(
            base_mount_path,
            "Xform",
            translation=side_cfg.robot_mount_pos,
            orientation=side_cfg.robot_mount_rot,
            stage=stage,
        )
    base_mount = stage.GetPrimAtPath(base_mount_path)

    # Match NVIDIA Robot Assembler's frame alignment before creating the
    # constraint. Deriving root-to-mount from world matrices also supports a
    # future hand whose configured mount is nested below the articulation root.
    mount_world = omni.usd.get_world_transform_matrix(base_mount)
    current_hand_world = omni.usd.get_world_transform_matrix(hand_root)
    current_mount_world = omni.usd.get_world_transform_matrix(hand_mount)
    root_to_mount = current_hand_world.GetInverse() * current_mount_world
    desired_hand_world = root_to_mount.GetInverse() * mount_world
    parent_world = omni.usd.get_world_transform_matrix(hand_root.GetParent())
    desired_hand_local = parent_world.GetInverse() * desired_hand_world
    omni.kit.commands.execute(
        "TransformPrimCommand",
        path=hand_root.GetPath(),
        new_transform_matrix=desired_hand_local,
    )
    _disable_world_root_joints(hand_root)

    fixed_joint = UsdPhysics.FixedJoint.Define(stage, f"{group_path}/{side.title()}FixedJoint")
    fixed_joint.CreateBody0Rel().SetTargets([Sdf.Path(base_mount_path)])
    fixed_joint.CreateBody1Rel().SetTargets([Sdf.Path(hand_mount_path)])
    fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed_joint.CreateExcludeFromArticulationAttr().Set(True)

    filter_api = UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(robot_root_path))
    filter_api.CreateFilteredPairsRel().AddTarget(Sdf.Path(hand_root_path))

    if update_default:
        asset = _MOUNTED_GRIPPERS.get(side)
        if asset is None:
            raise RuntimeError(f"Mounted gripper registry has no {side!r} articulation.")
        env_world = omni.usd.get_world_transform_matrix(stage.GetPrimAtPath(env_path))
        desired_in_env = env_world.GetInverse() * desired_hand_world
        position, rotation = _matrix_pose(desired_in_env)
        asset.cfg.init_state.pos = position
        asset.cfg.init_state.rot = rotation


def spawn_gripper_attachments(
    prim_path: str,
    cfg: "GripperAttachmentSpawnerCfg",
    translation=None,
    orientation=None,
    **kwargs,
) -> Usd.Prim:
    """Create one fixed-joint pair per already-spawned environment."""
    del translation, orientation, kwargs
    stage = sim_utils.get_current_stage()
    parent_expression, leaf = prim_path.rsplit("/", 1)
    env_paths = sim_utils.find_matching_prim_paths(parent_expression)
    if not env_paths:
        raise RuntimeError(f"No environment prims match {parent_expression!r}.")
    first_group = None
    for index, env_path in enumerate(env_paths):
        group_path = f"{env_path}/{leaf}"
        # AssetBaseCfg creates an XformPrimView after spawning. create_prim()
        # authors Isaac Lab's standard translate/orient/scale op sequence.
        group = sim_utils.create_prim(group_path, "Xform", stage=stage)
        first_group = first_group or group
        for side in cfg.settings.active_sides:
            _align_and_attach(
                stage,
                env_path,
                group_path,
                cfg.settings,
                side,
                update_default=index == 0,
            )
    assert first_group is not None
    return first_group


@configclass
class GripperAttachmentSpawnerCfg(SpawnerCfg):
    func: Callable = spawn_gripper_attachments
    settings: GripperSettings = MISSING


def build_gripper_attachment_cfg(settings: GripperSettings) -> AssetBaseCfg | None:
    if not settings.active_sides or settings.integrated:
        return None
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GripperAttachments",
        spawn=GripperAttachmentSpawnerCfg(settings=settings),
    )
