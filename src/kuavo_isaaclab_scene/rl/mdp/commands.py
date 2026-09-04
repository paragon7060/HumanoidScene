"""Goal/phase state for all skills, updated once per control step by manager terms.

Transitions never move task objects. Contacts, lift, support, release and dwell
must all come from simulation. Terminations run before rewards in Isaac Lab;
refresh() makes every manager see the same transition and terminal snapshot.
"""

import math
import torch
from isaaclab.managers import CommandTerm
from isaaclab.utils.math import quat_mul, quat_conjugate
from .geometry import rotate, unrotate, yaw, wrap_angle, projected_half_size, slot_offsets
from ..tasks.specs import PHASES


class WorkcellCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.spec = cfg.task
        self.robot = env.scene["robot"]
        self.boxes = [env.scene[n] for n in self.spec.box_names]
        self.n = len(self.boxes)
        self.ids = torch.arange(self.num_envs, device=self.device)
        self.center_offset = torch.tensor([cfg.geometry[n].center for n in self.spec.box_names], device=self.device)
        self.half_size = torch.tensor([cfg.geometry[n].half_size for n in self.spec.box_names], device=self.device)
        self.tool_ids, _ = self.robot.find_bodies(list(self.spec.tool_bodies), preserve_order=True)
        if len(self.tool_ids) != 2:
            raise ValueError("Exactly two tool bodies are required.")
        self.phase = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reward_phase = self.phase.clone()
        self.active_box = self.phase.clone()
        self.reward_box = self.phase.clone()
        self.slot = self.phase.clone()
        self.done_boxes = torch.zeros(self.num_envs, self.n, dtype=torch.bool, device=self.device)
        self.initial_z = torch.zeros(self.num_envs, self.n, device=self.device)
        self.dwell = torch.zeros(self.num_envs, device=self.device)
        self.belt_time = torch.zeros_like(self.dwell)
        self.belt_running = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = self.belt_running.clone()
        self.failure = self.belt_running.clone()
        self.transition = self.belt_running.clone()
        self.last_step = torch.full_like(self.phase, -1)
        self.goal = torch.zeros(self.num_envs, 3, device=self.device)  # world x, y, yaw
        self.metrics = {name: torch.zeros(self.num_envs, device=self.device)
                        for name in ("success", "boxes_placed", "phase", "cargo_retained")}
        self._measure()
        self._goals()

    @property
    def command(self):
        delta = torch.zeros(self.num_envs, 3, device=self.device)
        delta[:, :2] = self.goal[:, :2] - self.robot.data.root_pos_w[:, :2]
        delta = unrotate(self.robot.data.root_quat_w, delta)
        heading = wrap_angle(self.goal[:, 2] - yaw(self.robot.data.root_quat_w))
        return torch.cat((delta[:, :2], heading[:, None],
            torch.nn.functional.one_hot(self.phase, len(PHASES)).float(),
            torch.nn.functional.one_hot(self.active_box, self.n).float(),
            self.free_slots.float()), dim=-1)

    def _resample_command(self, env_ids):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.phase[ids] = 0 if self.spec.name == "full" else PHASES.index(self.spec.name)
        self.reward_phase[ids] = self.phase[ids]
        self.active_box[ids] = 0
        self.done_boxes[ids] = False
        self.dwell[ids] = 0
        self.belt_time[ids] = 0
        self.belt_running[ids] = False
        self.success[ids] = False
        self.failure[ids] = False
        self.transition[ids] = False
        self.last_step[ids] = self._env.common_step_counter
        self._measure()
        self.initial_z[ids] = self.centers[ids, :, 2] - self._env.scene.env_origins[ids, None, 2]
        saved = getattr(self._env, "_rl_reset_metadata", {})
        for env_id in ids.tolist():
            if env_id in saved:
                record = saved.pop(env_id)
                self.active_box[env_id] = record["active_box"]
                self.initial_z[env_id] = torch.tensor(record["initial_z"], device=self.device)
        self._measure()
        if self.spec.name == "press_button" and not self.supported[ids].all():
            raise ValueError("press_button reset bank must have ALL selected boxes supported on the conveyor.")
        if self.spec.name in ("carry", "place"):
            height = self.centers[self.ids, self.active_box, 2] - self._env.scene.env_origins[:, 2]
            if not (height[ids] > self.initial_z[ids, self.active_box[ids]] + self.spec.lift_height * 0.5).all():
                raise ValueError("Reset bank is not a lifted-box state for this task/geometry.")
        self._goals()

    def _measure(self):
        self.poses = torch.stack([b.data.root_pose_w for b in self.boxes], 1)
        self.velocities = torch.stack([b.data.root_vel_w for b in self.boxes], 1)
        self.centers = self.poses[..., :3] + rotate(self.poses[..., 3:], self.center_offset[None].expand(self.num_envs, -1, -1))
        up = torch.zeros_like(self.centers); up[..., 2] = 1
        self.upright = rotate(self.poses[..., 3:], up)[..., 2]
        belt = self._env.scene["conveyor_surface"]
        self.belt_pose = belt.data.root_pose_w
        bq = self.belt_pose[:, None, 3:].expand(-1, self.n, -1)
        local = unrotate(bq, self.centers - self.belt_pose[:, None, :3])
        relative_q = quat_mul(quat_conjugate(bq), self.poses[..., 3:])
        self.belt_half = projected_half_size(relative_q, self.half_size[None].expand(self.num_envs, -1, -1))
        # Full oriented box footprint must fit, not just its center.
        self.supported = ((local[..., :2].abs() + self.belt_half[..., :2]
                           <= torch.tensor((1.275, 0.34), device=self.device) - self.spec.clearance).all(-1)
                          & ((local[..., 2] - self.belt_half[..., 2] - 0.015).abs() < self.spec.support_tolerance)
                          & (self.upright > math.cos(self.spec.max_tilt)))
        # Reject box-on-box overlap and placements into an already occupied area.
        pair = (local[:, :, None, :2] - local[:, None, :, :2]).abs()
        ext = self.belt_half[:, :, None, :2] + self.belt_half[:, None, :, :2] + self.spec.clearance
        overlap = (pair < ext).all(-1)
        overlap &= ~torch.eye(self.n, dtype=torch.bool, device=self.device)[None]
        self.supported &= ~(overlap & self.supported[:, None, :]).any(-1)
        offsets = slot_offsets(self.spec.slot_count, self.spec.slot_pitch, self.device)
        self.slots_w = self.belt_pose[:, None, :3] + rotate(
            self.belt_pose[:, None, 3:].expand(-1, self.spec.slot_count, -1), offsets[None].expand(self.num_envs, -1, -1))
        active_half = self.belt_half[self.ids, self.active_box, :2]
        slot_delta = (offsets[None, :, None, :2] - local[:, None, :, :2]).abs()
        occupied = (slot_delta < active_half[:, None, None, :] + self.belt_half[:, None, :, :2] + self.spec.clearance).all(-1)
        occupied &= (local[..., 2].abs() < 0.5)[:, None, :]
        occupied[self.ids, :, self.active_box] = False
        self.free_slots = ~occupied.any(-1)
        self.free_slots &= (offsets[None, :, :2].abs() + active_half[:, None]
                           < torch.tensor((1.275, 0.34), device=self.device) - self.spec.clearance).all(-1)
        for i in range(self.spec.prefill_count):
            foreign = self._env.scene[f"prefill_{i}"]
            foreign_local = unrotate(self.belt_pose[:, 3:], foreign.data.root_pos_w - self.belt_pose[:, :3])
            fhalf = torch.tensor((0.15, 0.12), device=self.device)
            self.free_slots &= ~(((offsets[None, :, :2] - foreign_local[:, None, :2]).abs()
                                  < active_half[:, None] + fhalf + self.spec.clearance).all(-1)
                                 & (foreign_local[:, None, 2].abs() < 0.5))
            self.supported &= ~(((local[..., :2] - foreign_local[:, None, :2]).abs()
                                 < self.belt_half[..., :2] + fhalf + self.spec.clearance).all(-1)
                                & (foreign_local[:, None, 2].abs() < 0.5))
        body_pos = self.robot.data.body_link_pos_w[:, self.tool_ids]
        body_q = self.robot.data.body_link_quat_w[:, self.tool_ids]
        tool_offset = torch.tensor(self.spec.tool_offset, device=self.device).expand_as(body_pos)
        self.tools = body_pos + rotate(body_q, tool_offset)
        target = self.centers[self.ids, self.active_box]
        target_q = self.poses[self.ids, self.active_box, 3:]
        half = self.half_size[self.active_box]
        grips = torch.zeros(self.num_envs, 2, 3, device=self.device)
        grips[:, 0, 0] = half[:, 0] + 0.01
        grips[:, 1, 0] = -half[:, 0] - 0.01
        self.grips = target[:, None] + rotate(target_q[:, None].expand(-1, 2, -1), grips)
        self.reach_distance = torch.linalg.vector_norm(self.tools - self.grips, dim=-1).mean(-1)
        forces = []
        for index in range(4):
            matrix = self._env.scene[f"grasp_contact_{index}"].data.force_matrix_w
            if matrix is None or matrix.shape[2] != self.n:
                raise RuntimeError("Filtered finger→box contacts are missing; check USD Body paths and sensor filters.")
            forces.append(matrix[self.ids, 0, self.active_box].norm(dim=-1))
        self.contact_force = torch.stack(forces, -1)
        pairs = (self.contact_force > self.spec.grasp_force).reshape(self.num_envs, 2, 2).all(-1)
        nearby = (self.tools - target[:, None]).norm(dim=-1) < half.norm(dim=-1)[:, None] + self.spec.grasp_distance
        self.grasped = (pairs & nearby).sum(-1) >= self.spec.required_grasp_hands
        self.released = (self.contact_force < self.spec.grasp_force).all(-1) & ~nearby.any(-1)
        self.cargo_ok = torch.ones(self.num_envs, self.n, dtype=torch.bool, device=self.device)
        for box_id, name in enumerate(self.spec.box_names):
            for item in range(self.spec.cargo_per_box):
                cargo = self._env.scene[f"cargo_{name}_{item}"]
                point = unrotate(self.poses[:, box_id, 3:], cargo.data.root_pos_w - self.poses[:, box_id, :3])
                delta = (point - self.center_offset[box_id]).abs()
                self.cargo_ok[:, box_id] &= (delta < self.half_size[box_id] - self.spec.cargo_radius * 0.5).all(-1)
        button = self._env.scene["button_station"]
        button_ids, _ = button.find_bodies("Plunger")
        self.button_point = button.data.body_link_pos_w[:, button_ids[0]]
        button_joints, _ = button.find_joints("ButtonJoint")
        self.button_pressed = button.data.joint_pos[:, button_joints[0]] >= self.spec.button_travel

    def _goals(self):
        target = self.centers[self.ids, self.active_box]
        nominal = self.robot.data.default_root_state[:, :3] + self._env.scene.env_origins
        delta = target[:, :2] - nominal[:, :2]
        direction = delta / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        rack_goal = target[:, :2] - direction * self.spec.approach_distance
        distance = (self.slots_w[..., :2] - self.robot.data.root_pos_w[:, None, :2]).norm(dim=-1)
        distance = distance.masked_fill(~self.free_slots, torch.inf)
        self.slot = distance.argmin(-1)
        self.slot_goal = self.slots_w[self.ids, self.slot]
        delta_belt = self.slot_goal[:, :2] - nominal[:, :2]
        belt_direction = delta_belt / delta_belt.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        belt_goal = self.slot_goal[:, :2] - belt_direction * self.spec.approach_distance
        delta_button = self.button_point[:, :2] - nominal[:, :2]
        button_direction = delta_button / delta_button.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        button_goal = self.button_point[:, :2] - button_direction * self.spec.approach_distance
        self.goal[:, :2] = torch.where((self.phase >= 2)[:, None], belt_goal, rack_goal)
        self.goal[:, :2] = torch.where((self.phase == 4)[:, None], button_goal, self.goal[:, :2])
        facing = torch.where((self.phase >= 2)[:, None], belt_direction, direction)
        facing = torch.where((self.phase == 4)[:, None], button_direction, facing)
        self.goal[:, 2] = torch.atan2(facing[:, 1], facing[:, 0])
        self.nav_distance = (self.goal[:, :2] - self.robot.data.root_pos_w[:, :2]).norm(dim=-1)
        self.heading_error = wrap_angle(self.goal[:, 2] - yaw(self.robot.data.root_quat_w)).abs()

    def refresh(self):
        update = self.last_step != self._env.common_step_counter
        if not update.any():
            return
        self.last_step[update] = self._env.common_step_counter
        self.reward_phase[:] = self.phase
        self.reward_box[:] = self.active_box
        self.transition[:] = False
        self._measure()
        self._goals()
        target = self.centers[self.ids, self.active_box]
        lifted = target[:, 2] - self._env.scene.env_origins[:, 2] > self.initial_z[self.ids, self.active_box] + self.spec.lift_height
        upright = self.upright[self.ids, self.active_box] > math.cos(self.spec.max_tilt)
        navigated = (self.nav_distance < self.spec.navigation_tolerance) & (self.heading_error < self.spec.heading_tolerance)
        held = self.grasped & lifted & upright
        velocity = self.velocities[self.ids, self.active_box]
        settled = (velocity[:, :3].norm(dim=-1) < self.spec.settle_speed) & (velocity[:, 3:].norm(dim=-1) < self.spec.settle_angular_speed)
        placed = self.supported[self.ids, self.active_box] & settled & self.released
        condition = torch.where(self.phase == 0, navigated,
                    torch.where(self.phase == 1, held,
                    torch.where(self.phase == 2, held & navigated & self.free_slots.any(-1),
                    torch.where(self.phase == 3, placed,
                                self.supported.all(-1) & self.button_pressed
                                & ((self.tools - self.button_point[:, None]).norm(dim=-1).amin(-1)
                                   < self.spec.button_hand_distance)))))
        condition &= self.cargo_ok.all(-1)
        self.dwell[update] = torch.where(condition[update], self.dwell[update] + self._env.step_dt, 0.0)
        reached = (self.dwell >= self.spec.hold_seconds) & update & ~self.success
        self.transition[:] = reached & ~self.belt_running
        button_done = reached & (self.phase == 4)
        self.belt_running |= button_done
        self.belt_time += self.belt_running.float() * self._env.step_dt
        if self.spec.name == "full":
            placed_ids = (reached & (self.phase == 3)).nonzero().flatten()
            self.done_boxes[placed_ids, self.active_box[placed_ids]] = True
            next_box = (~self.done_boxes).long().argmax(-1)
            more = reached & (self.phase == 3) & ~self.done_boxes.all(-1)
            self.active_box[more] = next_box[more]
            advancing = reached & (self.phase < 4)
            self.phase[advancing] += 1
            self.phase[more] = 0
            self.dwell[reached] = 0
            self.success |= self.belt_running & (self.belt_time >= self.spec.conveyor_run_seconds)
        elif self.spec.name == "press_button":
            self.success |= self.belt_running & (self.belt_time >= self.spec.conveyor_run_seconds)
        else:
            self.success |= reached
        floor_drop = (self.centers[..., 2] - self._env.scene.env_origins[:, None, 2] < 0.12).any(-1)
        outside = (self.robot.data.root_pos_w[:, :2] - self._env.scene.env_origins[:, :2]).abs().amax(-1) > 3.0
        collision = self._env.scene["robot_contact"].data.net_forces_w.norm(dim=-1).amax(-1) > self.cfg.collision_force
        grace = self._env.episode_length_buf > 3
        self.failure |= (floor_drop | outside | ((~self.cargo_ok.all(-1) | collision) & grace)) & update
        self.success &= ~self.failure
        self.metrics["success"][:] = self.success.float()
        self.metrics["boxes_placed"][:] = self.supported.float().sum(-1)
        self.metrics["phase"][:] = self.phase.float()
        self.metrics["cargo_retained"][:] = self.cargo_ok.float().mean(-1)

    def _update_command(self):
        self.refresh()
        # Rewards used the previous phase's cached measurements. Refresh target
        # contacts/poses now so next-step observations match the new active box.
        self._measure()
        self._goals()

    def _update_metrics(self):
        pass


def task(env):
    command = env.command_manager.get_term("workcell")
    command.refresh()
    return command
