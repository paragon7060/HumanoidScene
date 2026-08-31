"""Manager-based Kuavo configuration specialized for Quest/OpenXR teleoperation."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from .gripper_runtime import build_gripper_action_cfg
from .teleop_ik import PersistentTeleopIKAction
from .teleop_body_action import TeleopBodyActionCfg
from .manager_env import (
    GRIPPER_SETTINGS,
    KuavoRobustWorkcellEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .workcell_layout import offset as layout_offset, rotation as layout_rotation


LEFT_ARM_JOINTS = [f"zarm_l{index}_joint" for index in range(1, 8)]
RIGHT_ARM_JOINTS = [f"zarm_r{index}_joint" for index in range(1, 8)]
HEAD_JOINTS = ["zhead_1_joint", "zhead_2_joint"]


@configclass
class TeleopActionsCfg:
    left_arm = DifferentialInverseKinematicsActionCfg(
        class_type=PersistentTeleopIKAction,
        asset_name="robot",
        joint_names=LEFT_ARM_JOINTS,
        body_name="zarm_l7_end_effector",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.035},
        ),
        scale=1.0,
        debug_vis=False,
    )
    right_arm = DifferentialInverseKinematicsActionCfg(
        class_type=PersistentTeleopIKAction,
        asset_name="robot",
        joint_names=RIGHT_ARM_JOINTS,
        body_name="zarm_r7_end_effector",
        controller=DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=True,
            ik_method="dls",
            ik_params={"lambda_val": 0.035},
        ),
        scale=1.0,
        debug_vis=False,
    )
    head = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=HEAD_JOINTS,
        scale=1.0,
        use_default_offset=True,
        preserve_order=True,
    )
    left_gripper = build_gripper_action_cfg(GRIPPER_SETTINGS, "left")
    right_gripper = build_gripper_action_cfg(GRIPPER_SETTINGS, "right")
    body = TeleopBodyActionCfg()


@configclass
class TeleopRewardsCfg(RewardsCfg):
    progress = None
    cargo_retention = None
    tote_stability = None
    obstacle_proximity = None
    action_rate = None
    joint_velocity = None
    success = None


@configclass
class TeleopTerminationsCfg(TerminationsCfg):
    time_out = None
    cargo_spill = None
    tote_drop = None
    human_or_robot_contact = None
    success = None


@configclass
class KuavoQuestTeleopEnvCfg(KuavoRobustWorkcellEnvCfg):
    """Single-environment bimanual IK scene with a robot-head XR anchor."""

    actions: TeleopActionsCfg = TeleopActionsCfg()
    rewards: TeleopRewardsCfg = TeleopRewardsCfg()
    terminations: TeleopTerminationsCfg = TeleopTerminationsCfg()
    xr: XrCfg = XrCfg(
        anchor_pos=layout_offset("robot", (0.0, 0.0, 0.55)),
        anchor_rot=layout_rotation("robot"),
        near_plane=0.08,
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        self.episode_length_s = 3600.0
        self.curriculum = None
        # Stronger simulation servo reduces gravity sag and lag; keep the
        # existing effort cap. These gains are not intended for real hardware.
        self.scene.robot.actuators["arms"].stiffness = 800.0
        self.scene.robot.actuators["arms"].damping = 50.0
        self.scene.robot.actuators["height_axis"].stiffness = 8000.0
        self.scene.robot.actuators["height_axis"].damping = 200.0
        # A fully extended elbow is singular for upward motion. Start with
        # spare elbow travel while preserving the tool's neutral orientation.
        self.scene.robot.init_state.joint_pos.pop("zarm_.*_joint", None)
        for side in ("l", "r"):
            for index in range(1, 8):
                self.scene.robot.init_state.joint_pos[f"zarm_{side}{index}_joint"] = {
                    1: .25, 4: -.65, 7: .40,
                }.get(index, 0.0)
        if GRIPPER_SETTINGS.integrated:
            self.scene.robot.init_state.joint_pos.pop("[lr]_[fb]_bar_[13]_joint", None)
        for side in GRIPPER_SETTINGS.active_sides:
            asset_cfg = self.scene.robot if GRIPPER_SETTINGS.integrated else getattr(
                self.scene, GRIPPER_SETTINGS.asset_name_for(side)
            )
            asset_cfg.init_state.joint_pos.update(GRIPPER_SETTINGS.command_for(side, GRIPPER_SETTINGS.open_command))
        # Collection reads sensors/state directly. Training observations add
        # object transforms, joint accelerations and duplicate camera copies.
        self.observations.policy = None
        self.observations.vision = None
        # Keep the physical Kuavo head and wrist cameras for the Quest camera
        # compositor. The waist camera is unnecessary for teleoperation.
        self.scene.waist_camera = None
        # Collection does not need the safety-worker / background AMR actors.
        # Keep the factory USD and its materials untouched.
        self.scene.moving_human = None
        self.scene.moving_robot = None
        self.events.reset_movers = None
        self.events.move_movers = None
        # The conveyor deck never moves. Keep its collider, but do not register
        # it for the dynamic-body reset that writes unsupported velocities to
        # kinematic bodies (PhysX errors on every reset).
        deck = self.scene.conveyor_surface
        self.scene.conveyor_surface = AssetBaseCfg(
            prim_path=deck.prim_path, spawn=deck.spawn,
            init_state=AssetBaseCfg.InitialStateCfg(pos=deck.init_state.pos, rot=deck.init_state.rot),
        )
        # Sensors stay in the scene, but the collector reads their tensors
        # directly so they do not need to be duplicated in observation dicts.
        self.teleop_devices = DevicesCfg(
            devices={
                "quest_handtracking": OpenXRDeviceCfg(
                    sim_device=self.sim.device,
                    xr_cfg=self.xr,
                )
            }
        )


def set_domain_randomization(cfg: KuavoQuestTeleopEnvCfg, enabled: bool) -> None:
    """Enable the robustness scene events, or reduce resets to deterministic ones."""
    if enabled:
        return
    cfg.events.robot_material = None
    cfg.events.robot_arm_mass = None
    cfg.events.actuator_gains = None
    cfg.events.left_gripper_material = None
    cfg.events.right_gripper_material = None
    cfg.events.left_gripper_gains = None
    cfg.events.right_gripper_gains = None
    cfg.events.tote_physics = None
    cfg.events.cargo_physics = None
    cfg.events.gravity = None
    cfg.events.reset_flap_friction = None
    cfg.events.reset_workcell = None
    cfg.events.reset_movers = None
    cfg.events.lighting = None
    cfg.events.move_movers = None
    cfg.events.cargo_disturbance = None
