"""Edit action joint order, scales, base limits and hand speed here."""

from isaaclab.utils import configclass
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.claw_assets.package import default_close_force_n
from ...robots.claw_assets.linkage import TWO_FINGER_PRESETS
from ...robots.robot_model import resolve_robot_model
from ..mdp.actions import PlanarDriveCfg, JointDeltaTargetsCfg, BinaryGripperCfg, ArmsOnlyJointTargetsCfg
from ..mdp.body_lock import ARM_JOINT_NAMES
from ..action_spaces import HEAD_JOINTS


def hand_action(side):
    hand = resolve_gripper_settings()
    force = ({"close_force_n": default_close_force_n(), "force_side": side}
             if hand.name in TWO_FINGER_PRESETS else {})
    return BinaryGripperCfg(asset_name=hand.asset_name_for(side),
        joint_names=list(hand.joint_names_for(side)),
        open_command_expr=hand.command_for(side, hand.open_command),
        close_command_expr=hand.command_for(side, hand.close_command),
        position_mapping=hand.sides[side].position_mapping,
        target_filter=hand.sides[side].target_filter,
        # Package value is total squeeze; the force model divides it equally
        # between the two symmetric jaws. Other gripper packages keep binary
        # position control without assuming this claw's linkage geometry.
        **force)


@configclass
class ActionsCfg:
    base = PlanarDriveCfg()
    upper_body = JointDeltaTargetsCfg(
        asset_name="robot", joint_names=["waist_yaw_joint", *[f"zarm_{s}{i}_joint" for s in "lr" for i in range(1, 8)]],
        scale=0.035, preserve_order=True)
    height = (JointDeltaTargetsCfg(asset_name="robot",
        joint_names=["knee_joint", "leg_joint", "waist_pitch_joint"],
        scale=0.015, preserve_order=True)
        if resolve_robot_model().has_wheel_base else None)
    left_gripper = hand_action("left")
    right_gripper = hand_action("right")


@configclass
class ArmsOnlyActionsCfg:
    # No dummy base/head/waist channels: the policy really has 14 + 1 + 1 actions.
    upper_body = ArmsOnlyJointTargetsCfg(asset_name="robot",
        joint_names=list(ARM_JOINT_NAMES), scale=0.035, preserve_order=True)
    left_gripper = hand_action("left")
    right_gripper = hand_action("right")


@configclass
class AllJointActionsCfg(ActionsCfg):
    head = JointDeltaTargetsCfg(asset_name="robot", joint_names=list(HEAD_JOINTS),
                               scale=.015, preserve_order=True)
