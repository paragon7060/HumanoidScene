"""Temporary v1 state term used only while the existing runner migrates to v2."""
from types import SimpleNamespace
from itertools import product
import math
import torch
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_conjugate, quat_mul
from ..mdp.geometry import rotate, unrotate, projected_half_size, slot_offsets
from ..mdp.flap_grasp import FlapGrasp
from ...robots.end_effector import get_end_effector_frames
from .kernels import placement_mask, advance_placement
from ._legacy_spec import SKILLS


class MultiBoxCommand(CommandTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.spec = env.cfg.multi_box
        self.robot = env.scene["robot"]
        self.boxes = [env.scene[n] for n in self.spec.box_names]
        self.ids = torch.arange(env.num_envs, device=env.device)
        self.tcp = get_end_effector_frames(self.robot)
        if not self.tcp.definition:
            raise ValueError("Four-box training requires calibrated endeffector centers.")
        self.offset = torch.tensor([cfg.geometry[n].center for n in self.spec.box_names], device=env.device)
        self.half = torch.tensor([cfg.geometry[n].half_size for n in self.spec.box_names], device=env.device)
        self.adapters, self.grasps = [], []
        for i in range(4):
            adapter = SimpleNamespace(spec=env.cfg.task, cfg=cfg, robot=self.robot, boxes=self.boxes,
                ids=self.ids, n=4, num_envs=env.num_envs, device=env.device, _env=env,
                active_box=torch.full((env.num_envs,), i, device=env.device, dtype=torch.long),
                endeffector_center=self.tcp)
            self.adapters.append(adapter)
            self.grasps.append(FlapGrasp(adapter))
        zeros = lambda *shape: torch.zeros(env.num_envs, *shape, device=env.device)
        self.initial_centers = zeros(4, 3)
        self.placement_time = zeros(4)
        self.paid = zeros(4).bool()
        self.credit = zeros(4).bool()
        self.elapsed = zeros()
        self.skill_time = zeros()
        self.target = zeros().long()
        self.route = zeros().long()
        self.last_step = torch.full_like(self.target, -1)
        self.success = zeros().bool()
        self.failure = zeros().bool()
        self.ready = zeros().bool()
        self.complete = zeros(4).bool()
        self.base_distance = zeros()
        self.dual_time = zeros()
        self.metrics = {k: zeros() for k in ("placed", "success", "base_distance", "dual_carry_seconds")}
        self._rack_boundary()
        self.measure()
        self.previous_base = self.robot.data.root_pos_w.clone()

    def _rack_boundary(self):
        from pxr import Usd, UsdGeom, Gf
        from ...workcell.workcell_layout import rotation
        q = torch.tensor([rotation("rack")], device=self.device)
        self.outward = rotate(q, torch.tensor([self.spec.rack_outward_local], device=self.device))[0]
        prim = self._env.scene.stage.GetPrimAtPath("/World/envs/env_0/Workcell/Racks/Rack/Visual")
        bound = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"]).ComputeWorldBound(prim).ComputeAlignedRange()
        corners = torch.tensor([tuple(Gf.Vec3d(*v)) for v in product(*[
            (bound.GetMin()[i], bound.GetMax()[i]) for i in range(3)])], device=self.device)
        self.rack_plane = ((corners - self._env.scene.env_origins[0]) * self.outward).sum(-1).max()

    @property
    def command(self):
        return torch.cat((torch.nn.functional.one_hot(self.target, 4),
                          torch.nn.functional.one_hot(self.route, 4)), -1).float()

    def _resample_command(self, env_ids):
        ids = self.ids[env_ids] if isinstance(env_ids, slice) else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for g in self.grasps:
            g.reset(ids)
        self.measure()
        self.initial_centers[ids] = self.centers[ids] - self._env.scene.env_origins[ids, None]
        self.target[ids] = torch.randint(4, (len(ids),), device=self.device) if self.spec.skill == "pick" else 0
        self.route[ids] = SKILLS.index(self.spec.skill) if self.spec.skill in SKILLS else 0
        self.placement_time[ids] = 0
        self.paid[ids] = False
        self.complete[ids] = False
        self.credit[ids] = False
        self.elapsed[ids] = self.skill_time[ids] = 0
        self.base_distance[ids] = self.dual_time[ids] = 0
        self.success[ids] = self.failure[ids] = self.ready[ids] = False
        self.previous_base[ids] = self.robot.data.root_pos_w[ids]
        for i in ids.tolist():
            saved = getattr(self._env, "_multi_box_restore", {}).pop(i, None)
            if saved:
                self.target[i] = saved["target"]
                self.initial_centers[i] = torch.tensor(saved["initial_centers"], device=self.device)
                self.paid[i] = torch.tensor(saved["paid"], device=self.device)
                for name, fields in saved["targets"].items():
                    term = self._env.action_manager.get_term(name)
                    for key, value in fields.items():
                        getattr(term, key)[i] = torch.tensor(value, device=self.device)
                # Rebuild physical contact after reset; never restore synthetic grasp flags.
        self.last_step[ids] = self._env.common_step_counter

    def measure(self):
        self.poses = torch.stack([b.data.root_pose_w for b in self.boxes], 1)
        self.velocities = torch.stack([b.data.root_vel_w for b in self.boxes], 1)
        self.centers = self.poses[..., :3] + rotate(self.poses[..., 3:], self.offset[None].expand(self.num_envs, -1, -1))
        for a, g in zip(self.adapters, self.grasps):
            a.tools = self.tcp.center_pose_w[..., :3]
            g.measure()
        self.hand_box_grasp = torch.stack([a.hand_grasp_flags for a in self.adapters], -1)
        self.held = self.hand_box_grasp.any(1)
        self.distances = torch.stack([a.hand_target_distance for a in self.adapters], -1)
        self.alignment = torch.stack([a.grasp_alignment for a in self.adapters], -1)
        self.forces = torch.stack([a.contact_force for a in self.adapters], 1)
        belt = self._env.scene["conveyor_surface"]
        bq = belt.data.root_quat_w[:, None].expand(-1, 4, -1)
        self.local = unrotate(bq, self.centers - belt.data.root_pos_w[:, None])
        self.projected = projected_half_size(quat_mul(quat_conjugate(bq), self.poses[..., 3:]),
                                             self.half[None].expand(self.num_envs, -1, -1))
        up = torch.zeros_like(self.centers); up[..., 2] = 1
        self.upright = rotate(self.poses[..., 3:], up)[..., 2]
        support = torch.stack([self._env.scene[f"belt_contact_{i}"].data.force_matrix_w.norm(dim=-1).flatten(1).sum(-1)
                               for i in range(4)], -1)
        s = self.spec
        # Require both calibrated moving jaw tips to clear the box volume, so
        # lifting/supporting the body without a flap latch is not called release.
        tips = self.tcp.tips_w.reshape(self.num_envs, 4, 3)
        pair_q = self.poses[:, :, None, 3:].expand(-1, -1, 4, -1)
        local_tips = unrotate(pair_q, tips[:, None] - self.centers[:, :, None])
        tip_clearance = (local_tips.abs() - self.half[None, :, None]).clamp_min(0).norm(dim=-1)
        self.released = (~self.held & (self.forces.amax(-1) < .2)
                         & (tip_clearance.amin(-1) > s.release_distance))
        self.valid_placement = placement_mask(self.local, self.projected,
            torch.tensor((1.275, .34), device=self.device), support,
            self.released,
            self.velocities[..., :3].norm(dim=-1), self.velocities[..., 3:].norm(dim=-1), self.upright,
            clearance=s.clearance, tolerance=s.support_tolerance, min_force=s.support_force,
            max_speed=s.placement_speed, max_angular=s.placement_angular_speed, min_up=math.cos(s.max_tilt))
        # Each box can target any vacant slot. Held boxes are not parked obstacles.
        slots = slot_offsets(4, .58, self.device)
        delta = (slots[None, None, :, None, :2] - self.local[:, None, None, :, :2]).abs()
        extents = self.half[None, :, None, None, :2] + self.projected[:, None, None, :, :2] + s.clearance
        occupied = (delta < extents).all(-1) & (~self.held[:, None, None, :])
        occupied &= (self.local[:, None, None, :, 2].abs() < .6)
        occupied &= ~torch.eye(4, device=self.device, dtype=torch.bool)[None, :, None, :]
        fits = (slots[None, :, :2].abs() + self.half[:, None, :2]
                <= torch.tensor((1.275, .34), device=self.device) - s.clearance).all(-1)
        self.free_slots = ~occupied.any(-1) & fits[None]
        distance = (self.local[:, :, None, :2] - slots[None, None, :, :2]).norm(dim=-1)
        choice = distance.masked_fill(~self.free_slots, torch.inf).argmin(-1)
        local_goal = slots[choice].clone()
        local_goal[..., 2] += self.half[None, :, 2]
        self.destinations = belt.data.root_pos_w[:, None] + rotate(bq, local_goal)
        self.destination_distance = (self.centers - self.destinations).norm(dim=-1)

    def conditions(self):
        s = self.spec
        height = self.centers[..., 2] - self._env.scene.env_origins[:, None, 2] - self.initial_centers[..., 2]
        pick = self.held & (height >= s.lift_height) & (self.upright > math.cos(s.max_tilt))
        # Entire oriented box must clear the rack's outward plane.
        direction = self.outward.expand(self.num_envs, 4, -1)
        extent = (unrotate(self.poses[..., 3:], direction).abs() * self.half[None]).sum(-1)
        near_face = ((self.centers - self._env.scene.env_origins[:, None]) * self.outward).sum(-1) - extent
        self.extraction_distance = (self.rack_plane + s.extraction_clearance - near_face).clamp_min(0)
        extract = pick & (near_face > self.rack_plane + s.extraction_clearance)
        carry = extract & self.free_slots.any(-1) & (self.destination_distance < s.carry_distance)
        return torch.stack((pick, extract, carry, self.complete), -1)

    def refresh(self):
        update = self.last_step != self._env.common_step_counter
        if not update.any():
            return
        self.last_step[update] = self._env.common_step_counter
        self.measure()
        from ..mdp.box_safety import box_safety_checks, failed
        self.failure_checks = box_safety_checks(self.centers, self.poses, self.velocities,
            self.initial_centers[..., 2], self._env.scene.env_origins, self.spec)
        self.box_safety_failure = failed(self.failure_checks)
        dt = self._env.step_dt
        self.elapsed[update] += dt
        just_ready = ~self.ready & (self.elapsed >= self.spec.settle_seconds) & update
        just_ready &= ~self.box_safety_failure
        if not self.spec.reset_bank:
            self.initial_centers[just_ready] = (self.centers - self._env.scene.env_origins[:, None])[just_ready]
        self.ready |= just_ready
        timer, paid, credit, complete = advance_placement(self.valid_placement & self.ready[:, None],
            self.placement_time, self.paid, dt, self.spec.placement_hold)
        self.credit[:] = False
        for dst, src in ((self.placement_time, timer), (self.paid, paid), (self.credit, credit), (self.complete, complete)):
            dst[update] = src[update]
        self.failure = ((self.centers[..., 2] - self._env.scene.env_origins[:, None, 2] < self.spec.failure_floor).any(-1)
            | ((self.robot.data.root_pos_w - self._env.scene.env_origins)[:, :2].norm(dim=-1) > self.spec.workspace_radius)) & self.ready
        self.failure |= self.box_safety_failure
        checks = self.conditions()
        condition = checks[self.ids, self.target, self.route] & self.ready
        self.skill_time[update] = torch.where(condition, self.skill_time + dt, 0.)[update]
        finished = self.skill_time >= self.spec.skill_hold
        self.success = (self.complete.all(-1) if self.spec.skill == "full" else finished) & ~self.failure
        if self.spec.strategy == "staged" and self.spec.skill == "full":
            advance = finished & update & ~self.success
            placed = advance & (self.route == 3)
            self.target[placed] = (~self.complete[placed]).long().argmax(-1)
            self.route[advance] = (self.route[advance] + 1) % 4
            # Reacquire after losing the target; physical state persists across switches.
            lost = (self.route > 0) & (self.route < 3) & ~self.held[self.ids, self.target]
            self.route[lost] = 0
            self.skill_time[advance | lost] = 0
        self.base_distance[update] += (self.robot.data.root_pos_w - self.previous_base)[:, :2].norm(dim=-1)[update]
        self.previous_base[update] = self.robot.data.root_pos_w[update]
        self.dual_time[update] += ((self.held.sum(-1) >= 2) * dt)[update]
        self.metrics["placed"][:] = self.complete.sum(-1)
        self.metrics["success"][:] = self.success
        self.metrics["base_distance"][:] = self.base_distance
        self.metrics["dual_carry_seconds"][:] = self.dual_time

    def _update_command(self):
        self.refresh()

    def _update_metrics(self):
        pass


@configclass
class MultiBoxCommandCfg(CommandTermCfg):
    class_type: type = MultiBoxCommand
    resampling_time_range: tuple = (1e9, 1e9)
    geometry: dict = None


def state(env):
    value = env.command_manager.get_term("workcell")
    value.refresh()
    return value
