"""Manager-based Kuavo configuration specialized for Quest/OpenXR teleoperation."""

from __future__ import annotations

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.devices import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg, XrCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .gripper_runtime import build_gripper_action_cfg
from .teleop_ik import PersistentTeleopIKAction
from .teleop_body_action import TeleopBodyActionCfg
from .manager_env import (
    GRIPPER_SETTINGS,
    KuavoRobustWorkcellEnvCfg,
    ROBOT_MODEL,
    RobustWorkcellSceneCfg,
    RewardsCfg,
    TerminationsCfg,
)
from .workcell_layout import offset as layout_offset, rotation as layout_rotation


LEFT_ARM_JOINTS = [f"zarm_l{index}_joint" for index in range(1, 8)]
RIGHT_ARM_JOINTS = [f"zarm_r{index}_joint" for index in range(1, 8)]
HEAD_JOINTS = ["zhead_1_joint", "zhead_2_joint"]


def _browser_eye_camera(name: str, lateral_offset_m: float) -> CameraCfg:
    """Create one RGB eye used by the lightweight Quest browser preview."""
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Kuavo/{ROBOT_MODEL.head_camera_body}/{name}",
        update_period=1.0 / 30.0,
        height=512,
        width=512,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            # 24 mm aperture / 12 mm focal length gives a 90 degree
            # horizontal FOV, which is closer to an HMD eye than the
            # physical Kuavo camera's narrow lens.
            focal_length=12.0,
            focus_distance=2.0,
            horizontal_aperture=24.0,
            clipping_range=(0.08, 8.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.08, lateral_offset_m, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


@configclass
class QuestTeleopSceneCfg(RobustWorkcellSceneCfg):
    """Workcell plus a virtual stereo pair used only by browser preview."""

    xr_left_eye_camera = _browser_eye_camera("XrLeftEyeCamera", 0.032)
    xr_right_eye_camera = _browser_eye_camera("XrRightEyeCamera", -0.032)


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

    scene: QuestTeleopSceneCfg = QuestTeleopSceneCfg(
        num_envs=1,
        env_spacing=5.0,
        replicate_physics=True,
    )
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
        from .robot_model import resolve_robot_model
        robot_model = resolve_robot_model()
        if robot_model.name == "s200062":
            from .teleop_inertials import spawn_teleop_robot
            # Shared hand/box physics is already configured by manager_env.
            # Only the kinematic base's wheel contact exception is Quest-only.
            self.scene.robot.spawn.func = spawn_teleop_robot
        self.scene.num_envs = 1
        self.scene.env_spacing = 5.0
        self.episode_length_s = 3600.0
        self.curriculum = None
        # Stronger simulation servo reduces gravity sag and lag; keep the
        # existing effort cap. These gains are not intended for real hardware.
        self.scene.robot.actuators["arms"].stiffness = 800.0
        self.scene.robot.actuators["arms"].damping = 50.0
        if robot_model.has_wheel_base:
            self.scene.robot.actuators["height_axis"].stiffness = 8000.0
            self.scene.robot.actuators["height_axis"].damping = 200.0
        # Give waist yaw its own servo; head gains remain unchanged.
        yaw = self.scene.robot.actuators["upper_body"].copy()
        yaw.joint_names_expr = ["waist_yaw_joint"]
        yaw.stiffness, yaw.damping = 800.0, 50.0
        self.scene.robot.actuators["waist_yaw"] = yaw
        self.scene.robot.actuators["upper_body"].joint_names_expr = ["zhead_.*_joint"]
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
