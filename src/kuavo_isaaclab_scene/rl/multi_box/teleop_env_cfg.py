"""Single-cell manager environment for Quest inspection of multi-box v2."""

from dataclasses import replace

from isaaclab.devices.openxr import XrCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ..managers.actions import AllJointActionsCfg
from ..managers.observations import ObservationsCfg
from ..managers.rewards import RewardsCfg
from ..managers.terminations import TerminationsCfg
from ..tasks.specs import TaskSpec
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.base_drive import apply_base_drive
from ...robots.initial_states import load_initial_state, merge_joint_position_defaults
from ...robots.robot_model import resolve_robot_model
from ...workcell.workcell_layout import offset as layout_offset, rotation as layout_rotation
from .scene.events import SceneEventsCfg
from .scene.scene_cfg import build_scene
from .spec import MultiBoxSpec


@configclass
class TeleopObservationsCfg(ObservationsCfg):
    policy = None


@configclass
class TeleopRewardsCfg(RewardsCfg):
    navigation = None
    approach_reaching = None
    reaching = None
    lift = None
    carrying = None
    placement = None
    button_reach = None
    stability = None
    action_rate = None
    joint_speed = None
    time_cost = None
    stage_completed = None
    success = None
    failure = None


@configclass
class TeleopTerminationsCfg(TerminationsCfg):
    success = None
    unsafe = None
    time_out = None


def _prepared_state_name() -> str:
    model = resolve_robot_model()
    return {
        "leju-twofinger": "s63_leju_ready_01",
        "s56_twofinger": "s56_twofinger_ready_01",
    }.get(model.integrated_gripper_preset, "quest_ready_02")


def _apply_prepared_robot_state(scene: InteractiveSceneCfg) -> str:
    """Make the model-specific prepared pose part of every default reset."""
    model = resolve_robot_model()
    hand = resolve_gripper_settings()
    name = _prepared_state_name()
    state = load_initial_state(name, robot_model=model.name, gripper=hand.name)
    robot = state["assets"]["robot"]
    scene.robot.init_state.joint_pos = merge_joint_position_defaults(
        scene.robot.init_state.joint_pos, robot.get("joint_positions", {}))
    if "root_pose" in robot:
        pose = robot["root_pose"]
        scene.robot.init_state.pos = tuple(pose[:3])
        scene.robot.init_state.rot = tuple(pose[3:])
    return name


@configclass
class MultiBoxTeleopEnvCfg(ManagerBasedRLEnvCfg):
    """One randomized v2 workcell driven by the existing Quest RL controls."""

    multi_box: MultiBoxSpec = MultiBoxSpec()
    task: TaskSpec = TaskSpec(
        # This facade selects the same physical-limit-preserving incremental
        # action behavior as mode 1. It does not supply v2 reward semantics.
        name="pick_place", control_mode="whole-body", active_arm="both",
        action_space="all-joints", grasp_mode="flap_top", episode_length_s=3600.0,
    )
    scene: InteractiveSceneCfg = None
    actions: AllJointActionsCfg = AllJointActionsCfg()
    observations: TeleopObservationsCfg = TeleopObservationsCfg()
    rewards: TeleopRewardsCfg = TeleopRewardsCfg()
    terminations: TeleopTerminationsCfg = TeleopTerminationsCfg()
    events: SceneEventsCfg = SceneEventsCfg()
    commands = None
    curriculum = None
    decimation: int = 4
    episode_length_s: float = 3600.0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**22,
            gpu_max_rigid_patch_count=2**20,
        ),
    )
    xr: XrCfg = XrCfg(
        anchor_pos=layout_offset("robot", (0.0, 0.0, 0.55)),
        anchor_rot=layout_rotation("robot"),
        near_plane=0.08,
    )
    prepared_state_name: str = ""

    def __post_init__(self):
        self.multi_box.validate()
        if self.multi_box.skill != "full":
            raise ValueError("Quest multi-box inspection uses the full 1-12 box scene.")
        self.scene, _ = build_scene(self.multi_box, num_envs=1, env_spacing=8.0)
        from .debug.contact_sensors import add_quest_contact_sensors
        add_quest_contact_sensors(self.scene)
        self.prepared_state_name = _apply_prepared_robot_state(self.scene)

        # Match the mode-1 whole-body action response used by rl_pick_place.py.
        self.actions.base.velocity_limits = (0.15, 0.15, 0.50)
        self.actions.base.acceleration_limits = (0.30, 0.30, 0.80)
        self.actions.upper_body.scale = {"waist_yaw_joint": 0.01, "zarm_.*_joint": 0.02}
        if self.actions.height is not None:
            self.actions.height.scale = 0.015
        self.actions.head.scale = 0.01
        # Use the same base model selected by the shared Quest CLI as regular
        # teleop and the RL environments.  In particular, mode 2 must not fall
        # back to the old root-pose teleport while mode 1 uses the new floating
        # base, since that would invalidate carry/slip debugging between modes.
        if apply_base_drive(self.scene.robot, self.actions, "base"):
            self.sim.physx.enable_external_forces_every_iteration = True
            print("[BASE] Dynamic base: floating root tracked by a PD wrench.", flush=True)

        # Retain the same body-servo support used by whole-body mode 1 for
        # models that do not provide the shared gravity-compensated writer.
        compensated = getattr(self.scene.robot.class_type, "gravity_compensation_enabled", False)
        if not compensated:
            actuators = self.scene.robot.actuators
            if "height_axis" in actuators:
                actuators["height_axis"].stiffness = 8000.0
                actuators["height_axis"].damping = 200.0
            if "upper_body" in actuators:
                actuators["upper_body"].stiffness = 800.0
                actuators["upper_body"].damping = 50.0

        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, -3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.num_rerenders_on_reset = 0


def build_quest_multi_box_cfg(args) -> MultiBoxTeleopEnvCfg:
    """Apply collector runtime options without enabling unfinished RL semantics."""
    box_count = getattr(args, "rl_shadow_box_count", None)
    count_range = ((box_count, box_count) if box_count is not None
                   else MultiBoxSpec().full_spawn_count_range)
    spec = replace(
        MultiBoxSpec(), strategy="end-to-end", skill="full", episode_seconds=None,
        full_spawn_count_range=count_range,
    )
    cfg = MultiBoxTeleopEnvCfg(multi_box=spec)
    cfg.seed = args.seed
    cfg.sim.device = args.device
    from ...robots.claw_assets.vr import configure_binary_gripper_control
    configure_binary_gripper_control(
        cfg,
        getattr(args, "gripper_close_force", None),
        contact_feedback=False,
        command_gate=None,
    )
    quality = args.render_quality == "quality"
    cfg.sim.render.antialiasing_mode = "DLSS"
    cfg.sim.render.dlss_mode = 2 if quality else 0
    cfg.sim.render.enable_reflections = quality
    cfg.sim.render.enable_translucency = quality
    cfg.sim.render.enable_global_illumination = quality
    cfg.sim.render.enable_ambient_occlusion = quality
    cfg.sim.render.samples_per_pixel = 2 if quality else 1
    return cfg
