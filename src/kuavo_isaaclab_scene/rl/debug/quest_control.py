"""Quest Cartesian goals -> existing RL incremental actions, without changing drives."""

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from ...teleop.teleop_ik import PersistentTeleopIKAction
from ...teleop.teleop_mapping import ScaledControllerMapper
from ...teleop.urdf_arm_ik import UrdfArm
from ...teleop.teleop_servo import arm_response_profile


def normalized_delta(target, current, scale):
    if not torch.isfinite(target).all() or not torch.isfinite(current).all():
        raise ValueError("Non-finite Quest joint target")
    if not torch.isfinite(torch.as_tensor(scale)).all() or (torch.as_tensor(scale) <= 0).any():
        raise ValueError("Quest reward inspection requires positive action scales")
    return ((target - current) / scale).clamp(-1, 1)


class QuestRLControl:
    def __init__(self, env, model, args):
        self.env, self.robot = env, env.scene["robot"]
        from ...robots.end_effector import get_end_effector_frames
        self.frames = get_end_effector_frames(self.robot)
        self.sides = ("left", "right") if env.cfg.task.active_arm == "both" else (env.cfg.task.active_arm,)
        if list(env.action_manager.active_terms) != ["upper_body", *[s + "_gripper" for s in self.sides]]:
            raise ValueError("Reward inspection requires arm deltas followed by active gripper actions")
        self.upper = env.action_manager.get_term("upper_body")
        self.mapper = ScaledControllerMapper(position_gain=args.position_gain, tool_forward_sign=model.tool_forward_sign)
        self.torso = self.robot.find_bodies("waist_yaw_link")[0][0]
        self.solvers, self.columns = {}, {}
        for side, letter in (("left", "l"), ("right", "r")):
            if side not in self.sides:
                continue
            cfg = DifferentialInverseKinematicsActionCfg(
                class_type=PersistentTeleopIKAction, asset_name="robot",
                joint_names=[f"zarm_{letter}{i}_joint" for i in range(1, 8)],
                body_name=f"zarm_{letter}7_end_effector", scale=1.,
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                       ik_method="dls"), debug_vis=False)
            solver = PersistentTeleopIKAction(cfg, env)
            solver.response = arm_response_profile(args.arm_response, "scaled", "controllers")
            solver.orientation_weight = args.arm_orientation_weight
            solver.configure_urdf(UrdfArm(model.urdf_path, side))
            self.solvers[side] = solver
            self.columns[side] = [self.upper._joint_ids.index(i) for i in solver._joint_ids]
        self.reset()

    def reset(self):
        self.mapper.reset()
        for solver in self.solvers.values():
            solver.reset()
            solver.hold_current_pose()

    def pose(self, body=None):
        value = self.robot.data.root_pose_w[0] if body is None else self.robot.data.body_pose_w[0, body]
        return value.detach().cpu().numpy()

    def action(self, packets):
        action = torch.zeros((1, self.env.action_manager.total_action_dim), device=self.env.device)
        for hand_index, side in enumerate(self.sides):
            solver = self.solvers[side]
            tcp = self.frames.center_pose_w[0, 0 if side == "left" else 1].detach().cpu().numpy()
            goal = self.mapper.target(side, packets[side], tcp, self.pose(),
                                      following=True, reference_pose_w=self.pose(self.torso))
            # Standalone IK only computes a target. Never call apply_actions():
            # the unchanged RL action manager is the only articulation writer.
            columns = self.columns[side]
            solver._joint_command[:] = self.upper.processed_actions[:, columns]
            solver.process_actions(torch.as_tensor(np.asarray(goal), device=self.env.device,
                                                  dtype=torch.float32).unsqueeze(0))
            scale = self.upper._scale
            if isinstance(scale, torch.Tensor):
                scale = scale[:, columns]
            action[:, columns] = normalized_delta(solver._joint_command,
                                                   self.upper.processed_actions[:, columns], scale)
            gripper = self.env.action_manager.get_term(side + "_gripper")
            desired = torch.full_like(gripper._signed_target, -1. if packets[side][1, 2] >= .5 else 1.)
            index = self.upper.action_dim + hand_index
            action[:, index:index + 1] = normalized_delta(
                desired, gripper._signed_target, gripper.cfg.delta_scale)
        return action
