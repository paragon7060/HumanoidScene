"""Isaac Lab runtime integration for configurable wrist grippers.

Each hand remains an independently addressable articulation.  A USD fixed
joint marked ``excludeFromArticulation`` attaches it to the corresponding
Kuavo wrist without merging the two Allegro joint-name namespaces.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Callable

import omni.kit.commands
import omni.usd
from pxr import Gf, Sdf, Usd, UsdPhysics

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.envs import mdp
from isaaclab.sim import SpawnerCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, NUCLEUS_ASSET_ROOT_DIR

from .gripper_config import GripperSettings


_MOUNTED_GRIPPERS: dict[str, Articulation] = {}


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
    if not settings.active_sides:
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
    if side not in settings.active_sides:
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
                enabled_self_collisions=True,
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


def build_gripper_action_cfg(settings: GripperSettings, side: str):
    if side not in settings.active_sides:
        return None
    return mdp.BinaryJointPositionActionCfg(
        asset_name=f"{side}_gripper",
        joint_names=list(settings.joint_names),
        open_command_expr=dict(settings.open_command),
        close_command_expr=dict(settings.close_command),
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
    if not settings.active_sides:
        return None
    return AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/GripperAttachments",
        spawn=GripperAttachmentSpawnerCfg(settings=settings),
    )
