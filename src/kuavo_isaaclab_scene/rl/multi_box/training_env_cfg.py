"""Multi-box v2 training assembly, isolated from the legacy four-box task."""

from __future__ import annotations

from dataclasses import replace

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from ..managers.actions import AllJointActionsCfg
from ..tasks.specs import TaskSpec
from ...robots.base_drive import apply_base_drive
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.robot_model import resolve_robot_model
from .debug.contact_sensors import add_multi_box_contact_sensors
from .managers.v2_observations import V2ObservationsCfg
from .managers.v2_grasp import V2GraspRewardsCfg, V2GraspTerminationsCfg
from .scene.events import SceneEventsCfg
from .scene.scene_cfg import build_scene
from .spec import MultiBoxSpec
from .teleop_env_cfg import _apply_prepared_robot_state


@configclass
class MultiBoxGraspAssemblyEnvCfg(ManagerBasedRLEnvCfg):
    """Bootable v2 grasp scene and deployable actor-observation assembly.

    This config is consumed only by the v2 grasp PPO entrypoint. Keeping this
    boundary explicit prevents the legacy four-box managers from silently
    entering v2 training.
    """

    multi_box: MultiBoxSpec = replace(
        MultiBoxSpec(), strategy="staged", skill="grasp", episode_seconds=30.0)
    task: TaskSpec = TaskSpec(
        name="pick_place", control_mode="whole-body", active_arm="both",
        action_space="all-joints", grasp_mode="flap_top", episode_length_s=30.0,
        obstacle_contact_force=0.1,
    )
    scene = None
    num_envs: int = 64
    env_spacing: float = 8.0
    actions: AllJointActionsCfg = AllJointActionsCfg()
    observations: V2ObservationsCfg = V2ObservationsCfg()
    rewards: V2GraspRewardsCfg = V2GraspRewardsCfg()
    terminations: V2GraspTerminationsCfg = V2GraspTerminationsCfg()
    events: SceneEventsCfg = SceneEventsCfg()
    commands = None
    curriculum = None
    decimation: int = 4
    episode_length_s: float = 30.0
    sim: SimulationCfg = SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physx=PhysxCfg(
            gpu_max_rigid_contact_count=2**22,
            # 1664 roller environments exceeded 2**20 patches during reset.
            gpu_max_rigid_patch_count=2**21,
        ),
    )
    prepared_state_name: str = ""
    validate_randomized_resets: bool = True

    def __post_init__(self):
        self.multi_box.validate()
        if (self.multi_box.strategy, self.multi_box.skill) != ("staged", "grasp"):
            raise ValueError("The first v2 training assembly is staged grasp only.")
        if self.multi_box.episode_seconds is None:
            raise ValueError("Grasp learning needs a finite rollout timeout.")
        model = resolve_robot_model()
        hand = resolve_gripper_settings()
        if model.name != "s63" or hand.name != "leju-twofinger":
            raise ValueError("Multi-box v2 grasp assembly requires S63 + Leju two-finger.")

        self.scene, _ = build_scene(
            self.multi_box, num_envs=self.num_envs, env_spacing=self.env_spacing)
        # These sensors are privileged training inputs and are never included
        # in the deployable actor observation.
        add_multi_box_contact_sensors(self.scene)
        self.prepared_state_name = _apply_prepared_robot_state(self.scene)

        self.actions.base.velocity_limits = (0.15, 0.15, 0.50)
        self.actions.base.acceleration_limits = (0.30, 0.30, 0.80)
        self.actions.upper_body.scale = {
            "waist_yaw_joint": 0.01,
            "zarm_.*_joint": 0.02,
        }
        if self.actions.height is not None:
            self.actions.height.scale = 0.015
        self.actions.head.scale = 0.01
        if apply_base_drive(self.scene.robot, self.actions, "base"):
            self.sim.physx.enable_external_forces_every_iteration = True

        from ...robots.claw_assets.vr import configure_binary_gripper_control
        configure_binary_gripper_control(
            self, contact_feedback=False, command_gate="multi_box_reset")
        self.episode_length_s = float(self.multi_box.episode_seconds)
        self.sim.render_interval = self.decimation
        self.viewer.eye = (3.0, -3.0, 2.5)
        self.viewer.lookat = (0.0, 0.0, 1.0)
        self.num_rerenders_on_reset = 0
