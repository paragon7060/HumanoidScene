"""Independent assembly; reuses asset factories without altering existing configs."""
from dataclasses import replace
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CommandTermCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.utils import configclass
from ..tasks.specs import TaskSpec
from ..scenes.scene_cfg import build_scene
from ..scenes.layout import box_spawn_plan
from ...robots.robot_model import resolve_robot_model
from ...robots.gripper_config import resolve_gripper_settings
from ._legacy_spec import MultiBoxSpec, validate_shelves
from ._legacy_state import MultiBoxCommandCfg
from .managers.actions import ActionsCfg, RightArmActionsCfg
from .managers.observations import ObservationsCfg
from .managers.rewards import RewardsCfg
from .managers.terminations import TerminationsCfg
from .managers.events import EventsCfg
from .managers.curriculum import CurriculumCfg
from .reset_states import RecordersCfg


@configclass
class CommandsCfg:
    workcell: CommandTermCfg = MultiBoxCommandCfg()


@configclass
class MultiBoxEnvCfg(ManagerBasedRLEnvCfg):
    multi_box: MultiBoxSpec = MultiBoxSpec()
    task: TaskSpec = None
    scene = None
    num_envs: int = 256
    env_spacing: float = 8.
    decimation: int = 4
    episode_length_s: float = 120.
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    commands: CommandsCfg = CommandsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    recorders: RecordersCfg = RecordersCfg()
    sim: SimulationCfg = SimulationCfg(dt=1/120, render_interval=4,
        physx=PhysxCfg(gpu_max_rigid_contact_count=2**22, gpu_max_rigid_patch_count=2**20))

    def __post_init__(self):
        self.multi_box.validate()
        model = resolve_robot_model()
        if not model.has_wheel_base or resolve_gripper_settings().name != "s200062_integrated":
            raise ValueError("Initial four-box baseline requires s200062 + s200062_integrated.")
        validate_shelves(box_spawn_plan(), self.multi_box)
        # TaskSpec here supplies asset/contact geometry settings only. Existing
        # stationary-pick validation/managers are intentionally not invoked.
        self.task = replace(TaskSpec(), name="full", box_names=self.multi_box.box_names,
            grasp_mode="flap_top", cargo_per_box=0, prefill_count=0, randomization=False,
            reset_settle_seconds=0., max_tilt=self.multi_box.max_tilt)
        self.scene, geometry = build_scene(self.task, self.num_envs, self.env_spacing, False)
        self.scene.robot.spawn.articulation_props.fix_root_link = True
        if self.multi_box.action_space == "right-arm":
            self.actions = RightArmActionsCfg()
        self.scene.lazy_sensor_update = False
        # No collision-failure baseline; remove unused high-dimensional sensors.
        for name in tuple(vars(self.scene)):
            if name.startswith("obstacle_contact_") or name == "robot_contact":
                setattr(self.scene, name, None)
        for i, name in enumerate(self.multi_box.box_names):
            box = getattr(self.scene, name)
            box.spawn.activate_contact_sensors = True
            body = geometry[name].body_path
            setattr(self.scene, f"belt_contact_{i}", ContactSensorCfg(
                prim_path=box.prim_path + ("/" + body if body != "." else ""),
                update_period=0., filter_prim_paths_expr=[self.scene.conveyor_surface.prim_path]))
        self.commands.workcell.geometry = geometry
        self.episode_length_s = self.multi_box.episode_seconds
        self.viewer.eye = (-3., -3., 2.2)
        self.viewer.lookat = (.1, .3, 1.1)
