"""Isaac adapter for the deployable v2 robot proprioception contract."""

from __future__ import annotations

import torch

from ....robots.end_effector import get_end_effector_frames
from ....robots.gripper_config import resolve_gripper_settings
from .robot_proprio import (
    ACTUATED_BODY_JOINTS, closure_fraction, kinematic_base_twist_world,
    robot_state_from_sensors,
)


class IsaacRobotProprioAdapter:
    def __init__(self, env):
        self.env = env
        self.robot = env.scene["robot"]
        self.tcp = get_end_effector_frames(self.robot)
        joint_ids, resolved = self.robot.find_joints(
            list(ACTUATED_BODY_JOINTS), preserve_order=True)
        if tuple(resolved) != ACTUATED_BODY_JOINTS:
            raise ValueError("Isaac robot is missing or reordered a v2 actuated body joint.")
        self.joint_ids = joint_ids
        settings = resolve_gripper_settings()
        if not settings.integrated or settings.active_sides != ("left", "right"):
            raise ValueError("V2 robot proprioception requires integrated bilateral grippers.")
        self.gripper_ids = []
        self.open_commands = []
        self.closed_commands = []
        for side in settings.active_sides:
            names = settings.joint_names_for(side)
            ids, resolved = self.robot.find_joints(list(names), preserve_order=True)
            if tuple(resolved) != names or len(ids) != 2:
                raise ValueError(f"V2 requires two actuated {side} gripper driver joints.")
            self.gripper_ids.append(ids)
            opened = settings.command_for(side, settings.open_command)
            closed = settings.command_for(side, settings.close_command)
            self.open_commands.append(torch.tensor(
                [opened[name] for name in names], device=env.device))
            self.closed_commands.append(torch.tensor(
                [closed[name] for name in names], device=env.device))

    def read(self):
        data = self.robot.data
        gripper_position = torch.stack([
            closure_fraction(data.joint_pos[:, ids], opened, closed)
            for ids, opened, closed in zip(
                self.gripper_ids, self.open_commands, self.closed_commands, strict=True)
        ], dim=-1)
        # BinaryGripper.raw_actions holds the commanded close bit after reset
        # gating. It is a controller command, not a contact/success signal.
        gripper_command = torch.cat([
            self.env.action_manager.get_term(f"{side}_gripper").raw_actions
            for side in ("left", "right")
        ], dim=-1)
        if gripper_command.shape != gripper_position.shape:
            raise ValueError("Expected one binary gripper command per hand.")
        # Floating roots report real dynamics; only a teleported kinematic root
        # needs its commanded local velocity converted into a world twist.
        base = self.env.action_manager.get_term("base")
        base_command = base.processed_actions
        base_twist_world = (data.root_vel_w if base.cfg.dynamic else
                            kinematic_base_twist_world(data.root_quat_w, base_command))
        # Logical PD targets are controller telemetry, available on the real
        # robot too. Do not expose gravity-biased PhysX drive targets instead.
        targets = data.joint_pos[:, self.joint_ids].clone()
        columns = {joint: i for i, joint in enumerate(self.joint_ids)}
        for name in ("upper_body", "height", "head"):
            if name not in self.env.action_manager.active_terms:
                continue
            term = self.env.action_manager.get_term(name)
            commanded = getattr(term, "_joint_targets", term.processed_actions)
            indices = [columns[joint] for joint in term._joint_ids]
            targets[:, indices] = commanded
        return robot_state_from_sensors(
            joint_pos=data.joint_pos[:, self.joint_ids],
            joint_vel=data.joint_vel[:, self.joint_ids],
            base_pose_world=data.root_pose_w,
            base_twist_world=base_twist_world,
            tcp_pose_world=self.tcp.center_pose_w,
            gripper_position=gripper_position,
            gripper_command=gripper_command,
            joint_targets=targets, base_command=base_command,
        )
