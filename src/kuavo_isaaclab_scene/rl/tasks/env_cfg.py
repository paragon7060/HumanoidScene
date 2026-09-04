"""Manager-based environment assembly. Existing teleop/eval configs stay independent."""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from .specs import TaskSpec
from .scene_cfg import build_scene
from ..managers.actions import ActionsCfg
from ..managers.observations import ObservationsCfg
from ..managers.commands import CommandsCfg
from ..managers.rewards import RewardsCfg
from ..managers.events import EventsCfg
from ..managers.terminations import TerminationsCfg
from ..managers.curriculum import CurriculumCfg
from ..managers.recorders import RecordersCfg


@configclass
class WorkcellRLEnvCfg(ManagerBasedRLEnvCfg):
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
        if self.num_envs < 1 or self.env_spacing < 5.0:
            raise ValueError("Use num_envs >= 1 and env_spacing >= 5 m to separate workcells.")
        self.decimation = 4
        self.episode_length_s = self.task.episode_length_s
        self.scene, geometry = build_scene(self.task, self.num_envs, self.env_spacing, self.cameras)
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
