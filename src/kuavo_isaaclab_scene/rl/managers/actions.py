"""Edit action joint order, scales, base limits and hand speed here."""

from isaaclab.utils import configclass
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.robot_model import resolve_robot_model
from ..mdp.actions import PlanarDriveCfg, JointDeltaTargetsCfg, IncrementalGripperCfg


def hand_action(side):
    hand = resolve_gripper_settings()
    return IncrementalGripperCfg(asset_name=hand.asset_name_for(side),
        joint_names=list(hand.joint_names_for(side)),
        open_command_expr=hand.command_for(side, hand.open_command),
        close_command_expr=hand.command_for(side, hand.close_command))


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
