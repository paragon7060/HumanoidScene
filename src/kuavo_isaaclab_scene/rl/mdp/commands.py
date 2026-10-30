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
        from ...robots.end_effector import get_end_effector_frames
        self.endeffector_center = get_end_effector_frames(self.robot)
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
        self.initial_centers = torch.zeros(self.num_envs, self.n, 3, device=self.device)
        self.initial_quats = torch.zeros(self.num_envs, self.n, 4, device=self.device)
        self.initial_quats[..., 0] = 1
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
        self.flap_grasp = None
        self.reach_progress = None
        self.flap_progress = None
        self.settling = None
        if self.spec.reset_settle_seconds and not self.spec.reset_bank:
            from .settling import ResetSettling
            self.settling = ResetSettling(self.num_envs, self.device, self.spec)
        if self.spec.grasp_mode == "flap_top":
            from .flap_grasp import FlapGrasp
            self.flap_grasp = FlapGrasp(self)
            from .reach_progress import ReachProgress
            self.reach_progress = ReachProgress(self.num_envs, self.device)
            from .flap_progress import FlapProgress
            self.flap_progress = FlapProgress(self.num_envs, self.device)
            self.metrics.update({name: torch.zeros(self.num_envs, device=self.device)
                                 for name in ("grasp_left", "grasp_right", "lift_height", "hold_fraction",
                                              "left_target_distance", "right_target_distance")})
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
        if self.settling is not None:
            self.settling.reset(ids)
        if self.flap_grasp is not None:
            self.flap_grasp.reset(ids)
        self._measure()
        self.initial_z[ids] = self.centers[ids, :, 2] - self._env.scene.env_origins[ids, None, 2]
        self.initial_centers[ids] = self.centers[ids] - self._env.scene.env_origins[ids, None]
        self.initial_quats[ids] = self.poses[ids, :, 3:]
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
        if self.reach_progress is not None:
            self.reach_progress.reset(ids)
            update = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            update[ids] = True
            enabled = self.phase == 1
            if self.settling is not None:
                enabled &= self.settling.ready
            self.reach_progress.advance(self.hand_target_distance, self.active_box, enabled, update,
                                        held=self.hand_grasp_flags)
            self.flap_progress.reset(ids)
            self.flap_progress.advance(self.grasp_alignment, self.hand_target_distance,
                self.nearest_flap_index, self.hand_grasp_flags, self.grasped,
                (self.centers[self.ids, self.active_box, 2] - self._env.scene.env_origins[:, 2]
                 - self.initial_z[self.ids, self.active_box]) / self.spec.lift_height,
                self.active_box, enabled, update, prime=True)

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
        if self.endeffector_center.definition:
            # One stable TCP definition across IK, reward, observations and recording.
            self.tools = self.endeffector_center.center_pose_w[..., :3]
        else:
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
            filters_per_box = len(self.spec.grasp_flaps) if self.flap_grasp is not None else 1
            if matrix is None or matrix.shape[2] != self.n * filters_per_box:
                raise RuntimeError("Filtered finger→box contacts are missing; check USD Body paths and sensor filters.")
            matrix = matrix[:, 0].reshape(self.num_envs, self.n, filters_per_box, 3)
            forces.append(matrix[self.ids, self.active_box].sum(-2).norm(dim=-1))
        self.contact_force = torch.stack(forces, -1)
        pairs = (self.contact_force > self.spec.grasp_force).reshape(self.num_envs, 2, 2).all(-1)
        nearby = (self.tools - target[:, None]).norm(dim=-1) < half.norm(dim=-1)[:, None] + self.spec.grasp_distance
        self.grasped = (pairs & nearby).sum(-1) >= self.spec.required_grasp_hands
        self.released = (self.contact_force < self.spec.grasp_force).all(-1) & ~nearby.any(-1)
        if self.flap_grasp is not None:
            self.flap_grasp.measure()
            from .collisions import obstacle_forces
            self.obstacle_forces = obstacle_forces(self._env)
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
        if self.settling is not None:
            completed = self.settling.advance(self.velocities, update, self._env.step_dt)
            self.initial_z[completed] = (self.centers[..., 2]
                - self._env.scene.env_origins[:, None, 2])[completed]
            self.initial_centers[completed] = (self.centers - self._env.scene.env_origins[:, None])[completed]
            self.initial_quats[completed] = self.poses[completed, :, 3:]
        if self.reach_progress is not None:
            enabled = self.reward_phase == 1
            if self.settling is not None:
                enabled &= self.settling.ready
            self.reach_progress.advance(self.hand_target_distance, self.reward_box, enabled, update,
                                        held=self.hand_grasp_flags)
            self.flap_progress.advance(self.grasp_alignment, self.hand_target_distance,
                self.nearest_flap_index, self.hand_grasp_flags, self.grasped,
                (self.centers[self.ids, self.reward_box, 2] - self._env.scene.env_origins[:, 2]
                 - self.initial_z[self.ids, self.reward_box]) / self.spec.lift_height,
                self.reward_box, enabled, update)
        target = self.centers[self.ids, self.active_box]
        lifted = target[:, 2] - self._env.scene.env_origins[:, 2] > self.initial_z[self.ids, self.active_box] + self.spec.lift_height
        upright = self.upright[self.ids, self.active_box] > math.cos(self.spec.max_tilt)
        navigated = (self.nav_distance < self.spec.navigation_tolerance) & (self.heading_error < self.spec.heading_tolerance)
        held = self.grasped & lifted & upright
        velocity = self.velocities[self.ids, self.active_box]
        settled = (velocity[:, :3].norm(dim=-1) < self.spec.settle_speed) & (velocity[:, 3:].norm(dim=-1) < self.spec.settle_angular_speed)
        # Pick success does not require a motion/contact-residual limit. Keep
        # the existing extra stability requirements for the later carry stage.
        carry_held = held
        if self.flap_grasp is not None:
            carry_held = held & settled & (self.unexpected_finger_force.amax(-1) < self.spec.unexpected_contact_limit)
        self.pick_checks = {
            "grasp": self.grasped, "height": lifted, "tilt": upright,
        }
        placed = self.supported[self.ids, self.active_box] & settled & self.released
        condition = torch.where(self.phase == 0, navigated,
                    torch.where(self.phase == 1, held,
                    torch.where(self.phase == 2, carry_held & navigated & self.free_slots.any(-1),
                    torch.where(self.phase == 3, placed,
                                self.supported.all(-1) & self.button_pressed
                                & ((self.tools - self.button_point[:, None]).norm(dim=-1).amin(-1)
                                   < self.spec.button_hand_distance)))))
        condition &= self.cargo_ok.all(-1)
        if self.settling is not None:
            condition &= self.settling.ready
            self.pick_checks["initial_wait"] = self.settling.ready
        if self.spec.cargo_per_box:
            self.pick_checks["cargo"] = self.cargo_ok.all(-1)
        self.dwell[update] = torch.where(condition[update], self.dwell[update] + self._env.step_dt, 0.0)
        self.pick_checks["hold"] = self.dwell >= self.spec.hold_seconds
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
        if self.flap_grasp is not None:
            collision = self.obstacle_forces.amax(-1) > self.spec.obstacle_contact_force
        else:
            collision = self._env.scene["robot_contact"].data.net_forces_w.norm(dim=-1).amax(-1) > self.cfg.collision_force
        grace = self._env.episode_length_buf > 3
        collision_failure = collision if self.flap_grasp is not None else collision & grace
        collision_failure = collision_failure & self.spec.collision_constraints_enabled
        self.failure_checks = {"floor_drop": floor_drop, "outside": outside,
                               "cargo_lost": ~self.cargo_ok.all(-1) & grace}
        if self.spec.collision_constraints_enabled:
            self.failure_checks["obstacle_collision"] = collision_failure
        if self.settling is not None and self.spec.reset_settle_timeout > 0:
            self.failure_checks["settle_timeout"] = self.settling.failed
        self.failure |= (floor_drop | outside | collision_failure | (~self.cargo_ok.all(-1) & grace)) & update
        if self.settling is not None and self.spec.reset_settle_timeout > 0:
            self.failure |= self.settling.failed & update
        self.success &= ~self.failure
        self.transition &= ~self.failure
        self.metrics["success"][:] = self.success.float()
        self.metrics["boxes_placed"][:] = self.supported.float().sum(-1)
        self.metrics["phase"][:] = self.phase.float()
        self.metrics["cargo_retained"][:] = self.cargo_ok.float().mean(-1)
        if self.flap_grasp is not None:
            self.metrics["grasp_left"][:] = self.hand_grasp_flags[:, 0].float()
            self.metrics["grasp_right"][:] = self.hand_grasp_flags[:, 1].float()
            self.metrics["lift_height"][:] = target[:, 2] - self._env.scene.env_origins[:, 2] - self.initial_z[self.ids, self.active_box]
            self.metrics["hold_fraction"][:] = (self.dwell / self.spec.hold_seconds).clamp(0, 1)
            self.metrics["left_target_distance"][:] = self.hand_target_distance[:, 0]
            self.metrics["right_target_distance"][:] = self.hand_target_distance[:, 1]

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
