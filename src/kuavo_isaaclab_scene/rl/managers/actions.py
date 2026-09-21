"""Edit action joint order, scales, base limits and hand speed here."""

from isaaclab.utils import configclass
from ...robots.gripper_config import resolve_gripper_settings
from ...robots.claw_assets.package import default_close_force_n
from ...robots.claw_assets.linkage import TWO_FINGER_PRESETS
from ...robots.claw_assets.vr import build_binary_gripper_action_cfg
from ...robots.robot_model import resolve_robot_model
from ..mdp.actions import PlanarDriveCfg, JointDeltaTargetsCfg, ArmsOnlyJointTargetsCfg
from ..mdp.body_lock import ARM_JOINT_NAMES
from ..action_spaces import HEAD_JOINTS


def hand_action(side, *, command_gate="settling"):
    hand = resolve_gripper_settings()
    force = default_close_force_n() if hand.name in TWO_FINGER_PRESETS else None
    return build_binary_gripper_action_cfg(
        hand,
        side,
        command_gate=command_gate,
        # Package value is total squeeze; the force model divides it equally
        # between the two symmetric jaws. Other gripper packages keep binary
        # position control without assuming this claw's linkage geometry.
        force_n=force,
    )


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
