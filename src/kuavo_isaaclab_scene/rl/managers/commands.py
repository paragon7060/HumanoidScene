"""Task goal, thresholds, contact definition and phase manager configuration."""

from isaaclab.managers import CommandTermCfg
from isaaclab.utils import configclass
from ..mdp.commands import WorkcellCommand
from ..tasks.specs import TaskSpec


@configclass
class WorkcellCommandCfg(CommandTermCfg):
    class_type: type = WorkcellCommand
    resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
    task: TaskSpec = TaskSpec()
    geometry: dict = {}
    collision_force: float = 180.0
    collision_reporting: str = "filtered"
    post_step_measurement: bool = True


@configclass
class CommandsCfg:
    workcell: WorkcellCommandCfg = WorkcellCommandCfg()
