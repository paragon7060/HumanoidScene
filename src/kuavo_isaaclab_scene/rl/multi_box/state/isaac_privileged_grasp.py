"""Vectorized privileged grasp measurements for multi-box v2 training.

This adapter reads exact Isaac contact and geometry tensors.  Its outputs are
reserved for reward, success, termination, and diagnostics; the deployable
actor observation has no dependency on this module.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ....robots.end_effector import get_end_effector_frames
from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from ....workcell.workcell_layout import scale as workcell_scale
from ...scenes.asset_geometry import box_geometry
from ..debug.contact_sensors import CONTACT_SENSOR_NAMES
from ..geometry import relative_pose
from ..geometry.pose import quat_apply
from ..geometry.rack import box_shelf_clearance_m
from ..metrics import GraspRawMetrics, grasp_potentials
from ..scene.spawn import BOX_TYPE_IDS, physical_asset_names
from ..success import (
    FingerFlapContacts,
    GraspSuccessInput,
    GraspSuccessResult,
    GraspSuccessTracker,
    PinchEvidence,
    RelativePoseStabilityTracker,
    classify_pinches,
)


MIN_JAW_FORCE_N = 5.0
FLAP_NAMES = ("flap_right", "flap_left")
FINGER_NAMES = ("l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger")


def _closest_flap_surface(
    points: torch.Tensor,
    centers: torch.Tensor,
    halves: torch.Tensor,
    normal_axes: torch.Tensor,
) -> torch.Tensor:
    delta = points - centers
    nearest = delta.clamp(-halves, halves)
    axis = normal_axes[..., None]
    side = torch.where(delta.gather(-1, axis) >= 0, 1.0, -1.0)
    return centers + nearest.scatter(-1, axis, side * halves.gather(-1, axis))


@dataclass(frozen=True)
class IsaacPrivilegedGraspStep:
    target_logical_id: torch.Tensor
    target_pool_id: torch.Tensor
    box_pose_world: torch.Tensor
    box_velocity_world: torch.Tensor
    lift_from_reset_m: torch.Tensor
    contacts: FingerFlapContacts
    pinch: PinchEvidence
    stable_hands: torch.Tensor
    rack_clearance_m: torch.Tensor
    raw: GraspRawMetrics
    potentials: dict[str, torch.Tensor]
    success: GraspSuccessResult
    bilateral_pinch_event: torch.Tensor
    success_event: torch.Tensor


class IsaacPrivilegedGraspAdapter:
    """Read exact batched grasp truth for one active box per environment."""

    def __init__(self, env):
        self.env = env
        self.num_envs = env.num_envs
        self.device = torch.device(env.device)
        self.names = physical_asset_names()
        self.robot = env.scene["robot"]
        self.tcp = get_end_effector_frames(self.robot)
        self.finger_ids, _ = self.robot.find_bodies(FINGER_NAMES, preserve_order=True)
        if len(self.finger_ids) != 4:
            raise ValueError("Multi-box v2 grasp requires four Leju finger links.")

        flap_ids = []
        flap_centers = []
        flap_halves = []
        flap_normal_axes = []
        for name in self.names:
            asset = env.scene[name]
            ids, _ = asset.find_bodies(FLAP_NAMES, preserve_order=True)
            if len(ids) != 2:
                raise ValueError(f"{name} is missing flap_right/flap_left rigid bodies.")
            geometry = box_geometry(getattr(env.cfg.scene, name), FLAP_NAMES)
            flaps = [geometry.flaps[key] for key in FLAP_NAMES]
            flap_ids.append(ids)
            flap_centers.append([value.center for value in flaps])
            flap_halves.append([value.half_size for value in flaps])
            flap_normal_axes.append([
                value.half_size.index(min(value.half_size)) for value in flaps
            ])
        self.flap_ids = flap_ids
        self.flap_centers = torch.tensor(
            flap_centers, dtype=torch.float32, device=self.device)
        self.flap_halves = torch.tensor(
            flap_halves, dtype=torch.float32, device=self.device)
        self.flap_normal_axes = torch.tensor(
            flap_normal_axes, dtype=torch.long, device=self.device)

        self.pose_stability = RelativePoseStabilityTracker(
            self.num_envs, self.device)
        self.success_tracker = GraspSuccessTracker(self.num_envs, self.device)
        self.initial_box_z = torch.zeros(self.num_envs, device=self.device)
        self.target_logical_id = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.target_pool_id = torch.full_like(self.target_logical_id, -1)
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)
        self.stability_armed = torch.zeros_like(self.initialized)
        self.bilateral_rewarded = torch.zeros_like(self.initialized)
        self.success_rewarded = torch.zeros_like(self.initialized)

    @staticmethod
    def _ids(env_ids, *, device, count):
        if env_ids is None:
            return torch.arange(count, dtype=torch.long, device=device)
        return torch.as_tensor(env_ids, dtype=torch.long, device=device)

    def reset(self, env_ids=None) -> None:
        ids = self._ids(env_ids, device=self.device, count=self.num_envs)
        self.pose_stability.reset(ids)
        self.success_tracker.reset(ids)
        self.initialized[ids] = False
        self.stability_armed[ids] = False
        self.bilateral_rewarded[ids] = False
        self.success_rewarded[ids] = False
        self.target_logical_id[ids] = -1
        self.target_pool_id[ids] = -1

    def _targets(self) -> tuple[torch.Tensor, torch.Tensor]:
        active = self.env._multi_box_active
        if active.shape != (self.num_envs, self.env.cfg.multi_box.max_boxes):
            raise ValueError("Unexpected multi-box active-mask shape.")
        if not bool((active.sum(-1) == 1).all()):
            raise ValueError("The staged grasp adapter requires one active box per environment.")
        logical = active.to(torch.long).argmax(-1)
        pool = self.env._multi_box_pool_ids.gather(1, logical[:, None]).squeeze(1)
        if bool(((pool < 0) | (pool >= len(self.names))).any()):
            raise ValueError("Active logical boxes must map to valid physical pool IDs.")
        return logical, pool

    def _selected_box_pose(self, pool: torch.Tensor) -> torch.Tensor:
        poses = torch.stack(
            [self.env.scene[name].data.root_pose_w for name in self.names], dim=1)
        rows = torch.arange(self.num_envs, device=self.device)
        return poses[rows, pool]

    def _selected_box_velocity(self, pool: torch.Tensor) -> torch.Tensor:
        velocities = torch.stack(
            [self.env.scene[name].data.root_vel_w for name in self.names], dim=1)
        rows = torch.arange(self.num_envs, device=self.device)
        return velocities[rows, pool]

    def _selected_flap_pose(self, pool: torch.Tensor) -> torch.Tensor:
        positions = torch.stack([
            self.env.scene[name].data.body_link_pos_w[:, ids]
            for name, ids in zip(self.names, self.flap_ids, strict=True)
        ], dim=1)
        quaternions = torch.stack([
            self.env.scene[name].data.body_link_quat_w[:, ids]
            for name, ids in zip(self.names, self.flap_ids, strict=True)
        ], dim=1)
        rows = torch.arange(self.num_envs, device=self.device)
        return torch.cat((positions[rows, pool], quaternions[rows, pool]), dim=-1)

    def _contacts(
        self,
        pool: torch.Tensor,
        flap_pose: torch.Tensor,
        centers: torch.Tensor,
        halves: torch.Tensor,
        axes: torch.Tensor,
    ) -> FingerFlapContacts:
        filter_count = len(self.names) * 2
        force_rows = []
        point_rows = []
        rows = torch.arange(self.num_envs, device=self.device)
        for sensor_name in CONTACT_SENSOR_NAMES:
            data = self.env.scene[sensor_name].data
            force = data.force_matrix_w
            point = data.contact_pos_w
            expected = (self.num_envs, 1, filter_count, 3)
            if force is None or point is None or force.shape != expected or point.shape != expected:
                raise RuntimeError(
                    f"{sensor_name} contact tensors must have shape {expected}; "
                    f"got force={None if force is None else tuple(force.shape)}, "
                    f"point={None if point is None else tuple(point.shape)}"
                )
            force_rows.append(force[:, 0].reshape(self.num_envs, len(self.names), 2, 3)[rows, pool])
            point_rows.append(point[:, 0].reshape(self.num_envs, len(self.names), 2, 3)[rows, pool])

        # Source sensor order is L-front, L-back, R-front, R-back.
        force_vector = torch.stack(force_rows, dim=1).reshape(
            self.num_envs, 2, 2, 2, 3).permute(0, 1, 3, 2, 4)
        point_world = torch.stack(point_rows, dim=1).reshape(
            self.num_envs, 2, 2, 2, 3).permute(0, 1, 3, 2, 4)
        force_n = force_vector.norm(dim=-1)

        expanded_flap = flap_pose[:, None, :, None].expand(-1, 2, -1, 2, -1)
        point_pose = torch.cat((
            point_world,
            flap_pose[:, None, :, None, 3:].expand(-1, 2, -1, 2, -1),
        ), dim=-1)
        point_local = relative_pose(expanded_flap, point_pose)[..., :3]
        delta = point_local - centers[:, None, :, None]
        margin = 0.004
        in_region = (
            torch.isfinite(point_local).all(-1)
            & (delta.abs() <= halves[:, None, :, None] + margin).all(-1)
            & (force_n > 0)
        )
        gather_axes = axes[:, None, :, None, None].expand(-1, 2, -1, 2, 1)
        signed_contact = delta.gather(-1, gather_axes).squeeze(-1)
        opposed = signed_contact[..., 0] * signed_contact[..., 1] < 0

        # PhysX may place both representative points on one face of a thin
        # convex flap.  In that case the physical jaw-link origins are the
        # unambiguous straddle check, as used by the validated VR probe.
        finger_world = self.robot.data.body_link_pos_w[:, self.finger_ids].reshape(
            self.num_envs, 2, 2, 3)
        finger_world = finger_world[:, :, None].expand(-1, -1, 2, -1, -1)
        finger_pose = torch.cat((
            finger_world,
            flap_pose[:, None, :, None, 3:].expand(-1, 2, -1, 2, -1),
        ), dim=-1)
        finger_local = relative_pose(expanded_flap, finger_pose)[..., :3]
        finger_delta = finger_local - centers[:, None, :, None]
        signed_finger = finger_delta.gather(-1, gather_axes).squeeze(-1)
        opposed |= signed_finger[..., 0] * signed_finger[..., 1] < 0
        return FingerFlapContacts(
            force_n=force_n,
            in_region=in_region,
            opposed=opposed,
            available=torch.ones(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _rack_clearance(
        self,
        box_pose: torch.Tensor,
        type_id: torch.Tensor,
        region_id: torch.Tensor,
    ) -> torch.Tensor:
        rack_pose = self.env.scene["rack"].data.root_pose_w
        shelf_gap = torch.empty(self.num_envs, device=self.device)
        type_names = {value: key for key, value in BOX_TYPE_IDS.items()}
        for candidate_type, box_name in type_names.items():
            for shelf in (2, 3):
                mask = (type_id == candidate_type) & (
                    torch.tensor(
                        [region.shelf for region in self.env.cfg.multi_box.rack_regions],
                        dtype=torch.long,
                        device=self.device,
                    )[region_id] == shelf
                )
                if bool(mask.any()):
                    shelf_gap[mask] = box_shelf_clearance_m(
                        box_pose[mask], rack_pose[mask], BOX_DIMENSIONS_M[box_name],
                        shelf=shelf, rack_scale=workcell_scale("rack"),
                    )
        moved_up = box_pose[:, 2] - self.initial_box_z
        return torch.minimum(moved_up, shelf_gap)

    def _raw_metrics(
        self,
        box_pose: torch.Tensor,
        flap_pose: torch.Tensor,
        centers: torch.Tensor,
        halves: torch.Tensor,
        axes: torch.Tensor,
        rack_clearance: torch.Tensor,
    ) -> GraspRawMetrics:
        tcp_pose = self.tcp.center_pose_w
        pair_pose = flap_pose[:, None].expand(-1, 2, -1, -1)
        tcp_for_flaps = torch.cat((
            tcp_pose[:, :, None, :3].expand(-1, -1, 2, -1),
            flap_pose[:, None, :, 3:].expand(-1, 2, -1, -1),
        ), dim=-1)
        tcp_local = relative_pose(pair_pose, tcp_for_flaps)[..., :3]
        nearest = _closest_flap_surface(
            tcp_local, centers[:, None], halves[:, None], axes[:, None])
        distance = (tcp_local - nearest).norm(dim=-1)
        direct = distance[:, 0, 0] + distance[:, 1, 1]
        swapped = distance[:, 0, 1] + distance[:, 1, 0]
        assignment = torch.where(
            (direct <= swapped)[:, None],
            torch.tensor((0, 1), device=self.device),
            torch.tensor((1, 0), device=self.device),
        )
        env_rows = torch.arange(self.num_envs, device=self.device)[:, None]
        hand_rows = torch.arange(2, device=self.device)[None]
        matched_distance = distance[env_rows, hand_rows, assignment].mean(-1)

        normals_local = torch.nn.functional.one_hot(axes, 3).to(tcp_pose.dtype)
        normals_world = quat_apply(flap_pose[..., 3:], normals_local)
        assigned_normal = normals_world[env_rows, assignment]
        fingers = self.robot.data.body_link_pos_w[:, self.finger_ids].reshape(
            self.num_envs, 2, 2, 3)
        closing = fingers[:, :, 0] - fingers[:, :, 1]
        closing = closing / closing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        alignment = torch.acos(
            (closing * assigned_normal).sum(-1).abs().clamp(0, 1)).mean(-1)

        assigned_pose = flap_pose[env_rows, assignment]
        finger_pose = torch.cat((
            fingers,
            assigned_pose[:, :, None, 3:].expand(-1, -1, 2, -1),
        ), dim=-1)
        finger_local = relative_pose(assigned_pose[:, :, None], finger_pose)[..., :3]
        assigned_center = centers[env_rows, assignment]
        assigned_half = halves[env_rows, assignment]
        assigned_axis = axes[env_rows, assignment]
        delta = finger_local - assigned_center[:, :, None]
        gather_axis = assigned_axis[:, :, None, None].expand(-1, -1, 2, 1)
        signed = delta.gather(-1, gather_axis).squeeze(-1)
        normal_outside = signed.amin(-1).clamp_min(0) + (-signed.amax(-1)).clamp_min(0)
        midpoint_delta = finger_local.mean(2) - assigned_center
        tangent_excess = (midpoint_delta.abs() - assigned_half).clamp_min(0)
        tangent_excess.scatter_(2, assigned_axis[..., None], 0.0)
        capture = torch.sqrt(
            normal_outside.square() + tangent_excess.square().sum(-1)).mean(-1)
        return GraspRawMetrics(
            matched_flap_distance_m=matched_distance,
            jaw_alignment_error_rad=alignment,
            capture_error_m=capture,
            proof_lift_m=rack_clearance,
        )

    def measure(self, dt: float) -> IsaacPrivilegedGraspStep:
        logical, pool = self._targets()
        box_pose = self._selected_box_pose(pool)
        box_velocity = self._selected_box_velocity(pool)
        changed = (~self.initialized) | (logical != self.target_logical_id) | (
            pool != self.target_pool_id)
        if bool(changed.any()):
            ids = changed.nonzero(as_tuple=False).flatten()
            self.pose_stability.reset(ids)
            self.success_tracker.reset(ids)
            self.stability_armed[ids] = False
            self.bilateral_rewarded[ids] = False
            self.success_rewarded[ids] = False
            self.initial_box_z[ids] = box_pose[ids, 2]
            self.target_logical_id[ids] = logical[ids]
            self.target_pool_id[ids] = pool[ids]
            self.initialized[ids] = True

        rows = torch.arange(self.num_envs, device=self.device)
        type_id = self.env._multi_box_box_type_ids[rows, logical]
        region_id = self.env._multi_box_region_ids[rows, logical]
        flap_pose = self._selected_flap_pose(pool)
        centers = self.flap_centers[pool]
        halves = self.flap_halves[pool]
        axes = self.flap_normal_axes[pool]
        contacts = self._contacts(pool, flap_pose, centers, halves, axes)
        pinch = classify_pinches(contacts, min_jaw_force_n=MIN_JAW_FORCE_N)
        rack_clearance = self._rack_clearance(box_pose, type_id, region_id)

        valid_flaps = ((pinch.hand_flap_index >= 0) & (pinch.hand_flap_index < 2)).all(-1)
        opposing = valid_flaps & (
            pinch.hand_flap_index[:, 0] != pinch.hand_flap_index[:, 1])
        ready = (
            pinch.hand_pinching.all(-1)
            & opposing
            & (rack_clearance >= self.success_tracker.config.proof_lift_m)
        )
        self.stability_armed |= ready
        track = pinch.hand_pinching.all(-1) & opposing & self.stability_armed
        hand_to_box = relative_pose(box_pose, self.tcp.center_pose_w)
        stable = self.pose_stability.update(
            hand_to_box, pinch.hand_pinching & track[:, None])
        success = self.success_tracker.update(GraspSuccessInput(
            hand_pinching=pinch.hand_pinching,
            hand_flap_index=pinch.hand_flap_index,
            relative_pose_stable=stable,
            rack_clearance_m=rack_clearance,
        ), dt)
        # Event rewards are episode-once latches. A policy must not farm the
        # pinch bonus by repeatedly losing and reacquiring the same box.
        bilateral_event = success.bilateral_pinch & ~self.bilateral_rewarded
        success_event = success.success & ~self.success_rewarded
        self.bilateral_rewarded |= success.bilateral_pinch
        self.success_rewarded |= success.success

        raw = self._raw_metrics(
            box_pose, flap_pose, centers, halves, axes, rack_clearance)
        return IsaacPrivilegedGraspStep(
            target_logical_id=logical,
            target_pool_id=pool,
            box_pose_world=box_pose,
            box_velocity_world=box_velocity,
            lift_from_reset_m=box_pose[:, 2] - self.initial_box_z,
            contacts=contacts,
            pinch=pinch,
            stable_hands=stable,
            rack_clearance_m=rack_clearance,
            raw=raw,
            potentials=grasp_potentials(raw),
            success=success,
            bilateral_pinch_event=bilateral_event,
            success_event=success_event,
        )
