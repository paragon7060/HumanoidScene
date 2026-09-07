"""Build a homogeneous minimal RL scene directly, never a general scene subclass."""

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from ..envs.parallel_cfg import ParallelEnvCfg
from .robot import build_robot_cfg
from .workcell import add_workcell, add_light
from .boxes import add_boxes
from .sensors import add_contacts, add_cameras

SCENE_PROFILE = "minimal_rl_v1"


@configclass
class MinimalRLSceneCfg(InteractiveSceneCfg):
    """Contains only explicitly assembled task assets and optional sensors."""


def build_scene(spec, num_envs=8, env_spacing=8., cameras=False, *, parallel=None):
    parallel = parallel or ParallelEnvCfg(num_envs=num_envs, env_spacing=env_spacing)
    scene = MinimalRLSceneCfg(**parallel.scene_kwargs())
    add_workcell(scene, parallel)
    scene.robot = build_robot_cfg()
    geometry = add_boxes(scene, spec)
    add_contacts(scene, spec, geometry)
    if cameras:
        add_cameras(scene)
    add_light(scene)
    return scene, geometry
