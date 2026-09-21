"""Shared action ordering for every skill and the end-to-end policy."""
from isaaclab.utils import configclass
from ...mdp.actions import PlanarDrive, PlanarDriveCfg, JointDeltaTargets, JointDeltaTargetsCfg
from ...managers.actions import hand_action
from ...mdp.actions import ArmsOnlyJointTargets, ArmsOnlyJointTargetsCfg
from ...action_spaces import RIGHT_ARM_JOINTS, HEAD_JOINTS


def gate(env, value):
    return value * env.command_manager.get_term("workcell").ready[:, None]


class MobileBase(PlanarDrive):
    def process_actions(self, actions):
        super().process_actions(gate(self._env, actions))


class FullJointTargets(JointDeltaTargets):
    def process_actions(self, actions):
        super().process_actions(gate(self._env, actions))


class RightArmTargets(ArmsOnlyJointTargets):
    def process_actions(self, actions):
        super().process_actions(gate(self._env, actions))


def gripper(side):
    return hand_action(side, command_gate="ready")


@configclass
class ActionsCfg:
    # Kinematic planar base abstraction: 3 velocity commands, not wheel torque control.
    base = PlanarDriveCfg(class_type=MobileBase)
    torso = JointDeltaTargetsCfg(class_type=FullJointTargets, asset_name="robot",
        joint_names=["knee_joint", "leg_joint", "waist_pitch_joint", "waist_yaw_joint"],
        scale=.015, preserve_order=True)
    arms = JointDeltaTargetsCfg(class_type=FullJointTargets, asset_name="robot",
        joint_names=[f"zarm_{s}{i}_joint" for s in "lr" for i in range(1, 8)],
        scale=.02, preserve_order=True)
    head = JointDeltaTargetsCfg(class_type=FullJointTargets, asset_name="robot",
        joint_names=list(HEAD_JOINTS), scale=.015, preserve_order=True)
    left_gripper = gripper("left")
    right_gripper = gripper("right")


@configclass
class RightArmActionsCfg:
    arms = ArmsOnlyJointTargetsCfg(class_type=RightArmTargets, asset_name="robot",
        joint_names=list(RIGHT_ARM_JOINTS), active_arm="right", scale=.02, preserve_order=True)
    right_gripper = gripper("right")
