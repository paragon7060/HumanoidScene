"""Batched, pose/proprio-only rack entry exploration using the shared IK servo.

This guide proposes actions for collection; contact checks and task success
are still measured by the unmodified environment. The deployed SAC actor
does not require this guide or any simulator contact truth.
"""

from __future__ import annotations

import torch

from .guided_exploration import RELATION_START, ASSIGNMENT_START
from ..demo_replay import _rotation_matrix
from ..spec import MAX_BOXES


def target_token(observation):
    mask_start = ASSIGNMENT_START + 2
    target = observation[:, mask_start + MAX_BOXES:mask_start + 2 * MAX_BOXES]
    ids = target.argmax(-1)
    tokens = observation[:, 86:RELATION_START].reshape(-1, MAX_BOXES, 22)
    rows = torch.arange(len(observation), device=observation.device)
    valid = (target.sum(-1) > 0.5) & (observation[rows, mask_start + ids] > 0.5)
    return tokens[rows, ids], valid


def entry_geometry(observation, front_y):
    """Assigned flap centers and the front plane, all expressed in robot base."""
    tcp = observation[:, 50:68].reshape(-1, 2, 9)
    rotation = _rotation_matrix(tcp[..., 3:])
    relation = observation[:, RELATION_START:ASSIGNMENT_START].reshape(-1, 2, 2, 9)
    left = observation[:, ASSIGNMENT_START:ASSIGNMENT_START + 2].argmax(-1)
    assignments = torch.stack((left, 1 - left), -1)
    rows = torch.arange(len(observation), device=observation.device)[:, None]
    hands = torch.arange(2, device=observation.device)[None]
    center = tcp[..., :3] + torch.matmul(
        rotation, relation[rows, hands, assignments, :3, None]).squeeze(-1)
    rack = observation[:, 68:77]
    rack_rotation = _rotation_matrix(rack[:, 3:])
    outward = rack_rotation[..., 1]
    front = rack[:, :3] + outward * front_y
    offset = ((front[:, None] - center) * outward[:, None]).sum(-1).clamp_min(0)
    stage = center + offset[..., None] * outward[:, None]
    return tcp, center, stage, outward


def successful_demo_grasp_offsets(demonstrations, front_y):
    """Offline grasp offsets selected only from physical demo pinch frames."""
    demo = demonstrations["actor_obs"]
    token, _ = target_token(demo)
    rotation = _rotation_matrix(token[:, 15:21])
    _, centers, _, _ = entry_geometry(demo, front_y)
    offsets = (rotation.transpose(-1, -2)[:, None]
               @ (demo[:, 50:68].reshape(-1, 2, 9)[..., :3] - centers)[..., None]).squeeze(-1)
    privileged = demonstrations["critic_obs"][:, demo.shape[1]:]
    close = demonstrations["action"][:, 20:22] > 0
    pinching = privileged[:, 35:37] > 0.5
    goals = []
    for hand in range(2):
        ids = torch.where(close[:, hand] & pinching[:, hand])[0]
        if not len(ids):
            raise ValueError("Demo grasp goals require physical pinch annotations for each hand")
        goals.append(offsets[ids[-1], hand])
    return torch.stack(goals)


class KinematicGraspExplorer:
    """Use successful demo wrist orientations and a shared bounded IK servo."""

    def __init__(self, env, demonstrations, *, grasp_goal="center", lift_distance_m=0.025,
                 base_clearance_m=0.65, torso_forward_m=0.0):
        from isaaclab.controllers import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        from isaaclab.utils.math import quat_from_matrix
        from ....teleop.teleop_ik import PersistentTeleopIKAction
        from ....workcell.workcell_layout import RACK_RAW_BOUNDS_M
        from ....workcell.workcell_layout import scale as workcell_scale
        from ..metrics.potentials import FRONT_STAGE_CLEARANCE_M

        self.env = env
        if grasp_goal not in ("center", "demo", "center-to-demo") or not 0.008 <= lift_distance_m <= 0.15 \
                or not 0.4 <= base_clearance_m <= 0.8 or not 0 <= torso_forward_m <= 0.15:
            raise ValueError("Invalid IK grasp goal or wrist lift distance")
        self.grasp_goal = grasp_goal
        self.lift_distance_m = lift_distance_m
        self.base_clearance_m, self.torso_forward_m = base_clearance_m, torso_forward_m
        self.front_y = RACK_RAW_BOUNDS_M[1][1] * workcell_scale("rack")[1] + FRONT_STAGE_CLEARANCE_M
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.close_ticks = torch.zeros_like(self.phase)
        self.lift_goal = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.slices = {}
        offset = 0
        for name in env.action_manager.active_terms:
            width = env.action_manager.get_term(name).action_dim
            self.slices[name] = slice(offset, offset + width)
            offset += width
        self.upper = env.action_manager.get_term("upper_body")
        self.solvers, self.columns = [], []
        for letter in "lr":
            solver = PersistentTeleopIKAction(DifferentialInverseKinematicsActionCfg(
                asset_name="robot", joint_names=[f"zarm_{letter}{i}_joint" for i in range(1, 8)],
                body_name=f"zarm_{letter}7_end_effector", scale=1.0,
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                      ik_method="dls")), env)
            solver.orientation_weight = 0.5
            self.solvers.append(solver)
            self.columns.append([self.upper._joint_ids.index(i) for i in solver._joint_ids])
        demo = demonstrations["actor_obs"].to(env.device)
        tokens, _ = target_token(demo)
        box_rotation = _rotation_matrix(tokens[:, 15:21])
        tcp_rotation = _rotation_matrix(demo[:, 50:68].reshape(-1, 2, 9)[..., 3:])
        orientations = box_rotation.transpose(-1, -2)[:, None] @ tcp_rotation
        close = demonstrations["action"].to(env.device)[:, 20:22] > 0
        # Select a real successful-close frame per hand; averaging rotations
        # or mixing the two demos can yield an invalid wrist orientation.
        self.relative_rotation = torch.stack([
            orientations[torch.where(close[:, hand])[0][-1], hand] for hand in range(2)])
        self.goal_offset = torch.zeros(2, 3, device=env.device)
        if grasp_goal in ("demo", "center-to-demo"):
            # Retarget the successful physical grasp location, not just the
            # wrist rotation, onto each new perceived neutral flap center.
            # Simulator contact annotations are used offline to choose a demo
            # frame; the live guide still consumes only deployable geometry.
            self.goal_offset = successful_demo_grasp_offsets(
                {key: demonstrations[key].to(env.device)
                 for key in ("actor_obs", "critic_obs", "action")}, self.front_y)
        self.quat_from_matrix = quat_from_matrix

    @torch.no_grad()
    def act(self, observation):
        tcp, centers, stage, outward = entry_geometry(observation, self.front_y)
        tokens, valid = target_token(observation)
        box_rotation = _rotation_matrix(tokens[:, 15:21])
        offset = (box_rotation[:, None] @ self.goal_offset[None, ..., None]).squeeze(-1)
        if self.grasp_goal == "demo":
            centers = centers + offset
        # Preserve the same front plane after translating the in-flap goal.
        front_offset = ((stage - centers) * outward[:, None]).sum(-1).clamp_min(0)
        stage = centers + front_offset[..., None] * outward[:, None]
        target_rotation = box_rotation[:, None] @ self.relative_rotation[None]
        orientation = self.quat_from_matrix(target_rotation.reshape(-1, 3, 3)).reshape(-1, 2, 4)
        stage_error = (stage - tcp[..., :3]).norm(dim=-1)
        current_rotation = _rotation_matrix(tcp[..., 3:])
        relative = current_rotation.transpose(-1, -2) @ target_rotation
        angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        reached_stage = ((stage_error < 0.07) & (angle < 0.35)).all(-1)
        self.phase = torch.where((self.phase == 0) & reached_stage, 1, self.phase)
        center_error = (centers - tcp[..., :3]).norm(dim=-1)
        # SAC may visit states outside the demonstration tube. Restage rather
        # than teaching insertion while a hand has drifted far from the box.
        recover = (self.phase > 0) & (center_error.amax(-1) > 0.3)
        self.phase[recover] = 0
        self.close_ticks[recover] = 0
        close = (self.phase[:, None] > 0) & (center_error < 0.035)
        self.close_ticks = torch.where(close.all(-1), self.close_ticks + 1, torch.zeros_like(self.close_ticks))
        begin_lift = (self.phase == 1) & (self.close_ticks >= 15)
        self.lift_goal[begin_lift] = centers[begin_lift]
        if self.grasp_goal == "center-to-demo":
            self.lift_goal[begin_lift] += offset[begin_lift]
        self.lift_goal[begin_lift, :, 2] += self.lift_distance_m
        self.phase = torch.where(begin_lift, 2, self.phase)
        target = torch.where((self.phase == 0)[:, None, None], stage, centers).clone()
        target = torch.where((self.phase == 2)[:, None, None], self.lift_goal, target)
        action = torch.zeros_like(self.env.action_manager.action)
        for hand, solver in enumerate(self.solvers):
            columns = self.columns[hand]
            solver._joint_command[:] = self.upper.processed_actions[:, columns]
            solver.process_actions(torch.cat((target[:, hand], orientation[:, hand]), -1))
            scale = self.upper._scale[:, columns]
            delta = (solver._joint_command - self.upper.processed_actions[:, columns]) / scale
            action[:, [self.slices["upper_body"].start + c for c in columns]] = delta.clamp(-1, 1)
            action[:, self.slices[("left_gripper", "right_gripper")[hand]]] = torch.where(
                close[:, hand, None] | (self.phase == 2)[:, None], 1.0, -1.0)
        # Place the base opposite the selected box while retaining clearance
        # from the rack front; IK then controls the arm approach independently.
        center = stage.mean(1) + outward * self.base_clearance_m
        base = self.env.action_manager.get_term("base")
        action[:, self.slices["base"]][:, :2] = (0.5 * center[:, :2] / base._scale[:2]).clamp(-0.5, 0.5)
        action[:, self.slices["base"]][:, 2] = torch.atan2(-outward[:, 1], -outward[:, 0]).clamp(-0.2, 0.2)
        # Height motion provides bounded extra reach for either shelf without
        # pitching the torso. X remains available to the learned policy.
        action[:, self.slices["height"]][:, 1] = (2 * (target[..., 2] - tcp[..., 2]).mean(-1)).clamp(-0.3, 0.3)
        action[self.phase > 0, self.slices["base"]] = 0
        action[self.phase > 0, self.slices["height"]] = 0
        if self.torso_forward_m:
            torso = self.env.action_manager.get_term("height")
            error = torso._origin_xz[:, 0] + self.torso_forward_m - torso.processed_actions[:, 0]
            assist = (self.phase < 2) & (center_error.amax(-1) > 0.05)
            action[:, self.slices["height"].start] = torch.where(
                assist, (2 * error).clamp(-0.3, 0.3), 0.0)
        return torch.where(valid[:, None], action, torch.zeros_like(action))

    def reset(self, done):
        self.phase[done] = self.close_ticks[done] = 0
        ids = torch.where(done)[0]
        if len(ids):
            for solver in self.solvers:
                solver.reset(ids)
