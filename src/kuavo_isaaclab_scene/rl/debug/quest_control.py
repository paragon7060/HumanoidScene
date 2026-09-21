"""Quest Cartesian goals -> existing RL actions, without changing drives."""

import numpy as np
import torch
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from ...teleop.teleop_ik import PersistentTeleopIKAction
from ...teleop.teleop_body import BODY_JOINTS, TeleopBodyMapper
from ...teleop.teleop_mapping import (
    AbsoluteControllerMapper,
    BimanualTeleopMapper,
    ScaledControllerMapper,
    TeleopMappingCfg,
)
from ...teleop.urdf_arm_ik import UrdfArm
from ...teleop.teleop_servo import arm_response_profile


def normalized_delta(target, current, scale):
    if not torch.isfinite(target).all() or not torch.isfinite(current).all():
        raise ValueError("Non-finite Quest joint target")
    if not torch.isfinite(torch.as_tensor(scale)).all() or (torch.as_tensor(scale) <= 0).any():
        raise ValueError("Quest reward inspection requires positive action scales")
    return ((target - current) / scale).clamp(-1, 1)


class QuestRLControl:
    def __init__(self, env, model, args, xr):
        self.env, self.robot = env, env.scene["robot"]
        from ...robots.end_effector import get_end_effector_frames
        self.frames = get_end_effector_frames(self.robot)
        self.xr = xr
        self.sides = ("left", "right") if env.cfg.task.active_arm == "both" else (env.cfg.task.active_arm,)
        self.term_slices = {}
        offset = 0
        for name in env.action_manager.active_terms:
            term = env.action_manager.get_term(name)
            self.term_slices[name] = slice(offset, offset + term.action_dim)
            offset += term.action_dim
        required = {"upper_body", *(side + "_gripper" for side in self.sides)}
        if env.cfg.task.control_mode == "whole-body":
            required.add("base")
            if model.has_wheel_base:
                required.add("height")
        missing = required - self.term_slices.keys()
        if missing:
            raise ValueError(f"Reward inspection is missing action terms: {sorted(missing)}")
        nonbinary = [name for name in required if name.endswith("_gripper")
                     and not hasattr(env.action_manager.get_term(name), "_close_requested")]
        if nonbinary:
            raise ValueError(
                f"Reward inspection requires shared binary gripper terms: {sorted(nonbinary)}")
        self.upper = env.action_manager.get_term("upper_body")
        self.body_mapper = None
        if env.cfg.task.control_mode == "whole-body":
            self.base = env.action_manager.get_term("base")
            self.height = env.action_manager.get_term("height") if "height" in self.term_slices else None
            self.body_mapper = TeleopBodyMapper(model.urdf_path, has_wheel_base=model.has_wheel_base)
            self.body_joint_ids = (
                self.robot.find_joints(BODY_JOINTS, preserve_order=True)[0]
                if model.has_wheel_base else []
            )
            self.waist_column = (
                self.upper._joint_ids.index(self.body_joint_ids[3])
                if self.body_joint_ids else None
            )
        self.relative_mapping = args.controller_mapping == "relative"
        if self.relative_mapping:
            self.mapper = None
            self.relative_mapper = BimanualTeleopMapper(TeleopMappingCfg(
                position_gain=args.position_gain,
                rotation_gain=args.rotation_gain,
            ))
        elif args.controller_mapping == "absolute":
            self.mapper = AbsoluteControllerMapper(tool_forward_sign=model.tool_forward_sign,
                                                   orientation_mode=args.absolute_orientation)
            self.relative_mapper = None
        else:
            self.mapper = ScaledControllerMapper(position_gain=args.position_gain, tool_forward_sign=model.tool_forward_sign)
            self.relative_mapper = None
        self.torso = self.robot.find_bodies("waist_yaw_link")[0][0]
        self.solvers, self.columns = {}, {}
        for side, letter in (("left", "l"), ("right", "r")):
            if side not in self.sides:
                continue
            cfg = DifferentialInverseKinematicsActionCfg(
                class_type=PersistentTeleopIKAction, asset_name="robot",
                joint_names=[f"zarm_{letter}{i}_joint" for i in range(1, 8)],
                body_name=f"zarm_{letter}7_end_effector", scale=1.,
                controller=DifferentialIKControllerCfg(
                    command_type="pose", use_relative_mode=self.relative_mapping,
                                                       ik_method="dls"), debug_vis=False)
            solver = PersistentTeleopIKAction(cfg, env)
            solver.response = arm_response_profile(args.arm_response, args.controller_mapping, "controllers")
            solver.orientation_weight = args.arm_orientation_weight
            solver.configure_urdf(UrdfArm(model.urdf_path, side))
            self.solvers[side] = solver
            self.columns[side] = [self.upper._joint_ids.index(i) for i in solver._joint_ids]
        self.reset()

    def reset(self):
        mapper = getattr(self, "mapper", None)
        relative_mapper = getattr(self, "relative_mapper", None)
        if mapper is not None:
            mapper.reset()
        if relative_mapper is not None:
            relative_mapper.reset()
        self.last_body_command = None
        for solver in self.solvers.values():
            solver.reset()
            solver.hold_current_pose()
        if self.body_mapper is not None:
            # The RL manager owns the commanded posture. Re-capturing measured
            # joints on settling/pause/recenter would adopt gravity sag as a
            # new target even though no body command was given.
            joints = (
                torch.cat((self.height.processed_actions[0],
                           self.upper.processed_actions[0, self.waist_column:self.waist_column + 1]))
                .detach().cpu().numpy()
                if self.body_joint_ids else None
            )
            self.body_mapper.reset(joints)

    def pose(self, body=None):
        value = self.robot.data.root_pose_w[0] if body is None else self.robot.data.body_pose_w[0, body]
        return value.detach().cpu().numpy()

    def action(self, packets):
        action = torch.zeros((1, self.env.action_manager.total_action_dim), device=self.env.device)
        upper_slice = self.term_slices["upper_body"]
        relative_actions = None
        relative_mapper = getattr(self, "relative_mapper", None)
        if relative_mapper is not None:
            root_quat = self.robot.data.root_quat_w[0].detach().cpu().numpy()
            relative_actions = relative_mapper.advance_controllers(
                packets["left"], packets["right"], None, root_quat).action
        for side in self.sides:
            solver = self.solvers[side]
            tcp = self.frames.center_pose_w[0, 0 if side == "left" else 1].detach().cpu().numpy()
            # Standalone IK only computes a target. Never call apply_actions():
            # the unchanged RL action manager is the only articulation writer.
            columns = self.columns[side]
            solver._joint_command[:] = self.upper.processed_actions[:, columns]
            if relative_mapper is not None:
                index = 0 if side == "left" else 6
                solver.process_actions(torch.as_tensor(
                    relative_actions[index:index + 6], device=self.env.device,
                    dtype=torch.float32).unsqueeze(0))
            else:
                goal = self.mapper.target(
                    side, packets[side], tcp, self.pose(), following=True,
                    aim_pose=self.xr.controller_aim_pose(side),
                    reference_pose_w=self.pose(self.torso))
                solver.process_actions(torch.as_tensor(
                    np.asarray(goal), device=self.env.device,
                    dtype=torch.float32).unsqueeze(0))
            scale = self.upper._scale
            if isinstance(scale, torch.Tensor):
                scale = scale[:, columns]
            action[:, [upper_slice.start + column for column in columns]] = normalized_delta(
                solver._joint_command, self.upper.processed_actions[:, columns], scale
            )
            # Match every Quest/RL execution path: 0 opens and 1 closes.
            action[:, self.term_slices[side + "_gripper"]] = (
                1.0 if packets[side][1, 2] >= .5 else 0.0
            )
        if self.body_mapper is not None:
            body = torch.as_tensor(
                self.body_mapper.advance(packets["left"], packets["right"], self.env.step_dt, enabled=True),
                device=self.env.device,
            ).unsqueeze(0)
            self.last_body_command = body[0].detach().cpu().tolist()
            action[:, self.term_slices["base"]] = (body[:, :3] / self.base._scale).clamp(-1, 1)
            if self.height is not None:
                action[:, self.term_slices["height"]] = normalized_delta(
                    body[:, 3:6], self.height.processed_actions, self.height._scale
                )
            if self.waist_column is not None:
                column = self.waist_column
                scale = self.upper._scale
                if isinstance(scale, torch.Tensor):
                    scale = scale[:, column:column + 1]
                action[:, upper_slice.start + column:upper_slice.start + column + 1] = normalized_delta(
                    body[:, 6:7], self.upper.processed_actions[:, column:column + 1], scale
                )
        return action
