"""Manager-based environment assembly. Existing teleop/eval configs stay independent."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from ..tasks.specs import TaskSpec
from ..scenes.scene_cfg import SCENE_PROFILE, build_scene
from ..managers.actions import ActionsCfg, ArmsOnlyActionsCfg
from ..managers.observations import ObservationsCfg, FlapPickObservationsCfg
from ..managers.commands import CommandsCfg
from ..managers.rewards import RewardsCfg, FlapPickRewardsCfg
from ..managers.events import EventsCfg
from ..managers.terminations import TerminationsCfg
from ..managers.curriculum import CurriculumCfg
from ..managers.recorders import RecordersCfg


@configclass
class WorkcellRLEnvCfg(ManagerBasedRLEnvCfg):
    scene_profile: str = SCENE_PROFILE
    task: TaskSpec = TaskSpec()
    num_envs: int = 8
    env_spacing: float = 8.0
    cameras: bool = False
    scene: InteractiveSceneCfg = None
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    recorders: RecordersCfg = RecordersCfg()
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=4,
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**22, gpu_max_rigid_patch_count=2**20))

    def __post_init__(self):
        self.task.validate()
        self.decimation = 4
        self.episode_length_s = self.task.episode_length_s
        self.scene, geometry = build_scene(self.task, self.num_envs, self.env_spacing, self.cameras)
        if self.task.control_mode == "arms-only":
            self.actions = ArmsOnlyActionsCfg()
            if self.task.active_arm != "both":
                side = self.task.active_arm
                inactive = "left" if side == "right" else "right"
                if getattr(self.actions, inactive + "_gripper").asset_name != "robot":
                    raise ValueError("Single-arm flap pick needs an integrated robot gripper for the inactive-hand lock.")
                self.actions.upper_body.active_arm = side
                self.actions.upper_body.joint_names = [f"zarm_{side[0]}{i}_joint" for i in range(1, 8)]
                setattr(self.actions, inactive + "_gripper", None)
            self.scene.robot.spawn.articulation_props.fix_root_link = True
        if self.task.grasp_mode == "flap_top":
            self.observations = FlapPickObservationsCfg()
            self.rewards = FlapPickRewardsCfg()
            if not self.task.collision_constraints_enabled:
                self.rewards.collision = None
            # Sample contacts at every physics substep, including impacts that
            # have ended before the next policy action.
            self.scene.lazy_sensor_update = False
            for name, sensor in vars(self.scene).items():
                if name.startswith("obstacle_contact_"):
                    sensor.history_length = self.decimation
        self.commands.workcell.task = self.task
        self.commands.workcell.geometry = geometry
        self.events.flap_friction.params["asset_names"] = self.task.box_names
        self.curriculum.reset_difficulty.params["ramp_steps"] = self.task.curriculum_steps
        if not self.task.randomization:
            self.events.arm_mass = self.events.arm_gains = self.events.flap_friction = None
            self.observations.policy.enable_corruption = False
            self.curriculum = None
        # Recorder also preserves terminal metrics across Isaac Lab auto-resets.
        self.viewer.eye = (3.0, -3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.rerender_on_reset = self.cameras
