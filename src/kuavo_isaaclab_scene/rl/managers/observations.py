"""Policy observations/noise; add a separate critic group here if needed."""

from isaaclab.envs import mdp
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg
from ..mdp import observations


@configclass
class PolicyObservationsCfg(ObservationGroupCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=AdditiveUniformNoiseCfg(n_min=-0.003, n_max=0.003))
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
    goal = ObsTerm(func=mdp.generated_commands, params={"command_name": "workcell"})
    objects = ObsTerm(func=observations.objects)
    cargo = ObsTerm(func=observations.cargo_state)
    task_state = ObsTerm(func=observations.task_state)
    actuator_state = ObsTerm(func=observations.actuator_state)
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = True


@configclass
class ObservationsCfg:
    policy: PolicyObservationsCfg = PolicyObservationsCfg()


@configclass
class FlapPickPolicyCfg(ObservationGroupCfg):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel, scale=0.1)
    box_pose = ObsTerm(func=observations.box_pose_only)
    box_velocity = ObsTerm(func=observations.box_velocity)
    box_rest_relation = ObsTerm(func=observations.box_rest_relation)
    hand_flap = ObsTerm(func=observations.hand_flap_relation)
    reaching_history = ObsTerm(func=observations.reaching_history)
    contacts_and_hold = ObsTerm(func=observations.flap_pick_state)
    actuator_state = ObsTerm(func=observations.actuator_state)
    last_action = ObsTerm(func=mdp.last_action)

    def __post_init__(self):
        self.concatenate_terms = True
        self.enable_corruption = False


@configclass
class FlapPickObservationsCfg:
    policy: FlapPickPolicyCfg = FlapPickPolicyCfg()
