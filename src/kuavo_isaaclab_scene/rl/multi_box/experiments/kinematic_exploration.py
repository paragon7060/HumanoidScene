"""Batched rack entry exploration using the shared IK servo.

This guide proposes actions for collection; contact checks and task success
are still measured by the unmodified environment. The training teacher uses
existing privileged contact evidence to confirm its lift handoff. The SAC actor
does not require this guide or any simulator contact truth.
"""

from __future__ import annotations

import torch

from .guided_exploration import RELATION_START, ASSIGNMENT_START
from ..demo_replay import _rotation_matrix
from ..spec import MAX_BOXES


IK_LIFT_CONFIRMATION = "physical_opposing_pinch_3ticks"


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


def assigned_flap_rotations(observation):
    """Panel orientation in the base frame from the actor's perceived relations."""
    tcp = observation[:, 50:68].reshape(-1, 2, 9)
    relations = observation[:, RELATION_START:ASSIGNMENT_START].reshape(-1, 2, 2, 9)
    left = observation[:, ASSIGNMENT_START:ASSIGNMENT_START + 2].argmax(-1)
    assignment = torch.stack((left, 1 - left), -1)
    rows = torch.arange(len(observation), device=observation.device)[:, None]
    hands = torch.arange(2, device=observation.device)[None]
    return _rotation_matrix(tcp[..., 3:]) @ _rotation_matrix(
        relations[rows, hands, assignment, 3:])


def successful_demo_grasp_offsets(demonstrations, front_y):
    """Offline grasp offsets selected only from physical demo pinch frames."""
    demo = demonstrations["actor_obs"]
    rotation = assigned_flap_rotations(demo)
    _, centers, _, _ = entry_geometry(demo, front_y)
    offsets = (rotation.transpose(-1, -2)
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


def retarget_grasp_goal(centers, stage, outward, box_rotation, goal_offset, grasp_goal):
    """Keep neutral observation anchors, but aim closing at the physical VR pose."""
    # Nominal callers supply [env,3,3]; articulated perception supplies the
    # separate [env,hand,3,3] panel frames. Do not rotate a bent-flap offset
    # using the unchanged box frame.
    rotation = box_rotation[:, None] if box_rotation.ndim == 3 else box_rotation
    offset = (rotation @ goal_offset[None, ..., None]).squeeze(-1)
    goals = centers + offset if grasp_goal == "demo" else centers
    front_offset = ((stage - goals) * outward[:, None]).sum(-1).clamp_min(0)
    return goals, goals + front_offset[..., None] * outward[:, None], offset


def observed_close_ticks(previous_ticks, proposed_close, observation):
    """Advance lift readiness only after the executed controller starts closing.

    A correction query is hypothetical: its positive jaw label does not mean
    the SAC policy executed that label. Commands and measured closure are
    deployable telemetry, not simulator pinch or success annotations.
    """
    observed = (observation[:, 48:50] > 0.5) & (observation[:, 46:48] > 0.5)
    closing = proposed_close.all(-1) & observed.all(-1)
    return torch.where(closing, previous_ticks + 1, torch.zeros_like(previous_ticks))


def confirmed_pinch_ticks(previous_ticks, hand_pinching, flap_index):
    """Empty closed jaws and two hands on one flap cannot start teacher lift."""
    valid = (flap_index >= 0) & (flap_index < 2)
    opposing = flap_index[:, 0] != flap_index[:, 1]
    confirmed = (hand_pinching & valid).all(-1) & opposing
    return torch.where(confirmed, previous_ticks + 1, torch.zeros_like(previous_ticks))


class KinematicGraspExplorer:
    """Use successful demo wrist orientations and a shared bounded IK servo."""

    def __init__(self, env, demonstrations, *, grasp_goal="demo", lift_distance_m=0.025,
                 base_clearance_m=0.65, torso_forward_m=0.0, orientation_mode="full"):
        from isaaclab.controllers import DifferentialIKControllerCfg
        from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
        from isaaclab.utils.math import quat_from_matrix
        from ....teleop.teleop_ik import PersistentTeleopIKAction
        from ....workcell.workcell_layout import RACK_RAW_BOUNDS_M
        from ....workcell.workcell_layout import scale as workcell_scale
        from ..metrics.potentials import FRONT_STAGE_CLEARANCE_M
        from ..state.schema import ACTUATED_BODY_JOINTS

        self.env = env
        if orientation_mode not in ("full", "closing-axis"):
            raise ValueError("Unsupported IK orientation mode")
        self.orientation_mode = orientation_mode
        axes = None
        if orientation_mode == "closing-axis":
            from ....robots.end_effector import closed_closing_axes
            axes = closed_closing_axes()
        if grasp_goal not in ("center", "demo", "center-to-demo") or not 0.008 <= lift_distance_m <= 0.15 \
                or not 0.4 <= base_clearance_m <= 0.8 or not 0 <= torso_forward_m <= 0.15:
            raise ValueError("Invalid IK grasp goal or wrist lift distance")
        self.grasp_goal = grasp_goal
        self.lift_distance_m = lift_distance_m
        self.base_clearance_m, self.torso_forward_m = base_clearance_m, torso_forward_m
        self.front_y = RACK_RAW_BOUNDS_M[1][1] * workcell_scale("rack")[1] + FRONT_STAGE_CLEARANCE_M
        self.phase = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.close_ticks = torch.zeros_like(self.phase)
        self.lost_contact_ticks = torch.zeros_like(self.phase)
        self.lift_goal = torch.zeros(env.num_envs, 2, 3, device=env.device)
        self.lift_rotation = torch.eye(3, device=env.device).expand(
            env.num_envs, 2, 3, 3).clone()
        self.slices = {}
        offset = 0
        for name in env.action_manager.active_terms:
            width = env.action_manager.get_term(name).action_dim
            self.slices[name] = slice(offset, offset + width)
            offset += width
        self.upper = env.action_manager.get_term("upper_body")
        self.solvers, self.columns = [], []
        self.velocity_columns = []
        for letter in "lr":
            solver = PersistentTeleopIKAction(DifferentialInverseKinematicsActionCfg(
                asset_name="robot", joint_names=[f"zarm_{letter}{i}_joint" for i in range(1, 8)],
                body_name=f"zarm_{letter}7_end_effector", scale=1.0,
                controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False,
                                                      ik_method="dls")), env)
            solver.orientation_weight = 0.5
            if axes is not None:
                solver.orientation_axis = torch.tensor(
                    axes["left" if letter == "l" else "right"], device=env.device)
            self.solvers.append(solver)
            self.columns.append([self.upper._joint_ids.index(i) for i in solver._joint_ids])
            self.velocity_columns.append([ACTUATED_BODY_JOINTS.index(name) for name in solver._joint_names])
        demo = demonstrations["actor_obs"].to(env.device)
        tokens, _ = target_token(demo)
        box_rotation = _rotation_matrix(tokens[:, 15:21])
        tcp_rotation = _rotation_matrix(demo[:, 50:68].reshape(-1, 2, 9)[..., 3:])
        orientations = box_rotation.transpose(-1, -2)[:, None] @ tcp_rotation
        if env.cfg.multi_box.flap_pose_source == "articulated":
            # Native articulated demonstrations must calibrate in each panel
            # frame. An explicitly allowed nominal prior remains approximate.
            orientations = assigned_flap_rotations(demo).transpose(-1, -2) @ tcp_rotation
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
        panel_rotation = box_rotation[:, None]
        if self.env.cfg.multi_box.flap_pose_source == "articulated":
            panel_rotation = assigned_flap_rotations(observation)
            valid &= observation[:, ASSIGNMENT_START:ASSIGNMENT_START + 2].sum(-1) > .5
        centers, stage, _ = retarget_grasp_goal(
            centers, stage, outward, panel_rotation, self.goal_offset, self.grasp_goal)
        target_rotation = panel_rotation @ self.relative_rotation[None]
        orientation = self.quat_from_matrix(target_rotation.reshape(-1, 3, 3)).reshape(-1, 2, 4)
        stage_error = (stage - tcp[..., :3]).norm(dim=-1)
        current_rotation = _rotation_matrix(tcp[..., 3:])
        relative = current_rotation.transpose(-1, -2) @ target_rotation
        angle = torch.acos(((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
        if self.orientation_mode == "closing-axis":
            axes = torch.stack([solver.orientation_axis for solver in self.solvers])
            current_axes = (current_rotation @ axes[None, ..., None]).squeeze(-1)
            target_axes = (target_rotation @ axes[None, ..., None]).squeeze(-1)
            angle = torch.acos((current_axes * target_axes).sum(-1).abs().clamp(0, 1))
        reached_stage = ((stage_error < 0.07) & (angle < 0.35)).all(-1)
        self.phase = torch.where((self.phase == 0) & reached_stage, 1, self.phase)
        center_error = (centers - tcp[..., :3]).norm(dim=-1)
        # SAC may visit states outside the demonstration tube. Restage rather
        # than teaching insertion while a hand has drifted far from the box.
        recover = (self.phase > 0) & (center_error.amax(-1) > 0.3)
        self.phase[recover] = 0
        self.close_ticks[recover] = 0
        close = (self.phase[:, None] > 0) & (center_error < 0.035)
        # A closed fraction only reports jaw travel. Actual probes showed
        # completely empty closed jaws passing that test and receiving a
        # positive lift label. Reuse the environment's existing contact cache;
        # no new sensor, reward, success predicate or actor feature is added.
        grasp = self.env._multi_box_privileged_grasp_step
        pinching = grasp.pinch.hand_pinching & ~(
            grasp.invalid_box_pose | grasp.invalid_flap_pose)[:, None]
        close |= pinching  # Preserve an actual grasp despite estimated-goal error.
        self.close_ticks = confirmed_pinch_ticks(
            self.close_ticks, pinching, grasp.pinch.hand_flap_index)
        self.phase = torch.where((self.phase == 0) & pinching.any(-1), 1, self.phase)
        losing = (self.phase == 2) & (self.close_ticks == 0)
        self.lost_contact_ticks = torch.where(
            losing, self.lost_contact_ticks + 1, torch.zeros_like(self.lost_contact_ticks))
        self.phase[self.lost_contact_ticks >= 15] = 1
        begin_lift = (self.phase == 1) & (self.close_ticks >= 3)
        # Lift from the measured successful capture, not a nominal point that
        # can pull the hands sideways out of a bent flap.
        self.lift_goal[begin_lift] = tcp[begin_lift, :, :3]
        self.lift_rotation[begin_lift] = current_rotation[begin_lift]
        self.lift_goal[begin_lift, :, 2] += self.lift_distance_m
        self.phase = torch.where(begin_lift, 2, self.phase)
        target = torch.where((self.phase == 0)[:, None, None], stage, centers).clone()
        hold_pinch = pinching & (self.phase < 2)[:, None]
        target = torch.where(hold_pinch[..., None], tcp[..., :3], target)
        target = torch.where((self.phase == 2)[:, None, None], self.lift_goal, target)
        target_rotation = torch.where(hold_pinch[..., None, None], current_rotation, target_rotation)
        target_rotation = torch.where((self.phase == 2)[:, None, None, None],
                                      self.lift_rotation, target_rotation)
        orientation = self.quat_from_matrix(target_rotation.reshape(-1, 3, 3)).reshape(-1, 2, 4)
        action = torch.zeros_like(self.env.action_manager.action)
        for hand, solver in enumerate(self.solvers):
            columns = self.columns[hand]
            solver._joint_command[:] = self.upper.processed_actions[:, columns]
            # A hypothetical label was not executed by SAC. Bound the next
            # servo step around measured velocity, rather than its ghost momentum.
            solver._joint_velocity[:] = observation[:, 20:40][:, self.velocity_columns[hand]]
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
            assist = (self.phase < 2) & ~pinching.any(-1) & (center_error.amax(-1) > 0.05)
            action[:, self.slices["height"].start] = torch.where(
                assist, (2 * error).clamp(-0.3, 0.3), 0.0)
        return torch.where(valid[:, None], action, torch.zeros_like(action))

    def reset(self, done):
        self.phase[done] = self.close_ticks[done] = 0
        self.lost_contact_ticks[done] = 0
        ids = torch.where(done)[0]
        if len(ids):
            for solver in self.solvers:
                solver.reset(ids)
