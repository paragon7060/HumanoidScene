"""Shared CLI vocabulary and joint order for every online RL runner."""

from dataclasses import replace

ACTION_SPACES = ("right-arm", "all-joints")
RIGHT_ARM_JOINTS = tuple(f"zarm_r{i}_joint" for i in range(1, 8))
HEAD_JOINTS = ("zhead_1_joint", "zhead_2_joint")


def add_action_space_argument(parser, default=None):
    parser.add_argument("--action-space", choices=ACTION_SPACES, default=default,
        help="right-arm: 7 right arm joints + right gripper; all-joints: planar base, "
             "torso, both arms, head and both grippers. Omit to keep the task config.")


def select_task_action_space(spec, choice):
    if choice is None:
        return spec
    if choice not in ACTION_SPACES:
        raise ValueError(f"Unknown action space: {choice}")
    if choice == "right-arm":
        if spec.name in ("approach_rack", "carry", "full"):
            raise ValueError(f"{spec.name} requires base motion; select --action-space all-joints.")
        return replace(spec, action_space=choice, control_mode="arms-only", active_arm="right",
                       grasp_hand="right", required_grasp_hands=1)
    return replace(spec, action_space=choice, control_mode="whole-body", active_arm="both")
