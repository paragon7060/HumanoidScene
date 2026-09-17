"""Independent v2 scene assembly; no legacy TaskSpec or static box capture."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

from ..spec import MultiBoxSpec
from .assets import add_box_pool, add_randomizable_workcell
from ...envs.parallel_cfg import ParallelEnvCfg
from ...scenes.robot import build_robot_cfg
from ...scenes.workcell import add_light


@configclass
class MultiBoxSceneCfg(InteractiveSceneCfg):
    pass


def build_scene(spec: MultiBoxSpec, num_envs=8, env_spacing=8.0, *, parallel=None):
    parallel = parallel or ParallelEnvCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene = MultiBoxSceneCfg(**parallel.scene_kwargs())
    add_randomizable_workcell(scene, parallel)
    scene.robot = build_robot_cfg()
    # Whole-body actions drive the virtual base joints; the imported floating
    # root itself must not fall under gravity at scene startup.
    scene.robot.spawn.articulation_props.fix_root_link = True
    pool = add_box_pool(scene, spec)
    add_light(scene)
    scene.lazy_sensor_update = False
    return scene, pool
