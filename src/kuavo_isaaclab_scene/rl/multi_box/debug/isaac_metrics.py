"""Read-only Isaac scene adapter for multi-box v2 shadow reward metrics.

This module deliberately does not decide grasp/place success.  It measures
geometry and privileged box velocity for reward calibration while contact
sensors and deployable transition confidence are integrated separately.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch

from ....robots.end_effector import get_end_effector_frames
from ....robots.claw_assets import load_claw_config
from ....workcell.rack_box_layout import BOX_DIMENSIONS_M
from ...scenes.asset_geometry import box_geometry
from .contact_sensors import CONTACT_SENSOR_NAMES, BELT_CONTACT_SENSOR_NAMES
from ..geometry import relative_pose, unsigned_axis_angle_error
from ..geometry.belt import BELT_HALF_EXTENTS_XY
from ..geometry.pose import quat_apply
from ..geometry.rack import box_shelf_clearance_m
from ..geometry.pad_distance import pad_to_boxes_clearance_m
from ..metrics import (
    CarryRawMetrics,
    GRASP_APPROACH_REWARD_SCALE_M,
    GraspRawMetrics,
    PlaceRawMetrics,
    carry_potentials,
    grasp_reward_potentials,
    opposing_flap_reach_assignment,
    place_potentials,
)
from ..scene.spawn import BOX_TYPE_IDS, physical_asset_names
from ....workcell.workcell_layout import scale as workcell_scale
from ..success.contact import FingerFlapContacts


BELT_HALF_THICKNESS_M = 0.015
EXTRACTION_GOAL_M = 0.10


def _closest_box_surface(points: torch.Tensor, centers: torch.Tensor,
                         halves: torch.Tensor, normal_axes: torch.Tensor) -> torch.Tensor:
    """Closest point on either broad face of a thin flap AABB."""
    delta = points - centers
    nearest = delta.clamp(-halves, halves)
    axis = normal_axes[..., None]
    side = torch.where(delta.gather(-1, axis) >= 0, 1.0, -1.0)
    return centers + nearest.scatter(-1, axis, side * halves.gather(-1, axis))


def _body_footprint_corners_belt(
    body_pose: torch.Tensor,
    belt_pose: torch.Tensor,
    body_center: torch.Tensor,
    body_half_size: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the projected *physical bottom* corners and bottom height.

    Isaac's root pose is the rigid-link frame, not the bottom face of the
    cardboard body.  Using that pose directly made a box appear roughly one
    half-height above the belt.  Build the corners from the measured Body USD
    bounds instead, including the local center offset, then transform them to
    the belt frame.  Taking the minimum of the four z values also remains
    correct when the box is slightly tilted.
    """
    half_x, half_y, half_z = body_half_size
    cx, cy, cz = body_center
    local = torch.stack((
        torch.stack((cx - half_x, cy - half_y, cz - half_z)),
        torch.stack((cx - half_x, cy + half_y, cz - half_z)),
        torch.stack((cx + half_x, cy - half_y, cz - half_z)),
        torch.stack((cx + half_x, cy + half_y, cz - half_z)),
    ))
    world = body_pose[:3] + quat_apply(body_pose[3:].expand(4, -1), local)
    points = torch.cat((world, body_pose[3:].expand(4, -1)), dim=-1)
    belt_points = relative_pose(belt_pose.expand(4, -1), points)
    return belt_points[:, :2], belt_points[:, 2].amin() - BELT_HALF_THICKNESS_M


def _footprint_outside(corners: torch.Tensor, half_extents: torch.Tensor) -> torch.Tensor:
    excess = (corners.abs() - half_extents).clamp_min(0)
    return excess.norm(dim=-1).amax().reshape(1)


def _signed_aabb_clearance(corners: torch.Tensor, other_corners: list[torch.Tensor]) -> torch.Tensor:
    """Positive gap or negative overlap depth against the closest active box."""
    if not other_corners:
        return corners.new_tensor([1.0])
    low, high = corners.amin(0), corners.amax(0)
    center, half = (low + high) / 2, (high - low) / 2
    values = []
    for other in other_corners:
        other_low, other_high = other.amin(0), other.amax(0)
        other_center = (other_low + other_high) / 2
        other_half = (other_high - other_low) / 2
        separation = (center - other_center).abs() - (half + other_half)
        if bool((separation > 0).any()):
            values.append(separation.clamp_min(0).norm())
        else:
            values.append(-(-separation).amin())
    return torch.stack(values).amin().reshape(1)


@dataclass(frozen=True)
class IsaacMetricSnapshot:
    target_logical_id: int
    target_asset_name: str
    target_box_type: str
    target_region: str
    raw_by_phase: dict[str, object]
    potentials_by_phase: dict[str, dict[str, torch.Tensor]]
    diagnostics: dict[str, float]
    contacts: FingerFlapContacts
    hand_to_box_pose: torch.Tensor
    rack_clearance_m: torch.Tensor
    box_footprint_corners_belt: torch.Tensor
    box_bottom_height_m: torch.Tensor
    box_tilt_rad: torch.Tensor
    overlaps_other_belt_box: torch.Tensor
    belt_body_force_n: torch.Tensor
    belt_sensor_available: torch.Tensor
    gripper_box_distance_m: torch.Tensor

    def raw_scalars(self, phase: str) -> dict[str, torch.Tensor]:
        return asdict(self.raw_by_phase[phase])

    def log_scalars(self, phase: str) -> dict[str, object]:
        return {**self.raw_scalars(phase), **self.diagnostics}

    def contact_report(self) -> str:
        if not self.diagnostics.get("contact_adapter_available", 0.0):
            return "CONTACT: unavailable"
        lines = ["CONTACT RAW (no success threshold):"]
        for hand in ("l", "r"):
            for flap in ("right", "left"):
                front = self.diagnostics[f"contact_{hand}_{flap}_front_n"]
                back = self.diagnostics[f"contact_{hand}_{flap}_back_n"]
                valid = (
                    int(self.diagnostics[f"contact_{hand}_{flap}_front_region"]),
                    int(self.diagnostics[f"contact_{hand}_{flap}_back_region"]),
                )
                opposed = int(self.diagnostics[f"contact_{hand}_{flap}_opposed"])
                lines.append(
                    f"{hand.upper()} {flap:<5} F={front:.2f}/{back:.2f}N "
                    f"region={valid[0]}{valid[1]} opp={opposed}")
        return "\n".join(lines)


class IsaacMultiBoxMetricAdapter:
    """Measure one locked logical target in the single-environment Quest scene."""

    def __init__(self, env):
        if env.num_envs != 1:
            raise ValueError("Quest shadow metrics currently require exactly one environment.")
        self.env = env
        self.robot = env.scene["robot"]
        self.tcp = get_end_effector_frames(self.robot)
        self.names = physical_asset_names()
        self.type_names = {value: key for key, value in BOX_TYPE_IDS.items()}
        self.finger_ids, _ = self.robot.find_bodies(
            ["l_f_finger", "l_b_finger", "r_f_finger", "r_b_finger"],
            preserve_order=True,
        )
        if len(self.finger_ids) != 4:
            raise ValueError("Multi-box v2 shadow metrics require four two-finger links.")
        pad = load_claw_config()["contact"]["distal_pad"]
        self.pad_size_m = tuple(float(value) for value in pad["size_m"])
        self.pad_centers = torch.tensor(
            [pad["center_m"][jaw] for jaw in ("f", "b", "f", "b")],
            device=env.device,
        )
        self.flap_ids = []
        self.flap_centers = []
        self.flap_halves = []
        self.flap_normal_axes = []
        self.part_ids = []
        self.part_centers = []
        self.part_halves = []
        for name in self.names:
            asset = env.scene[name]
            body_ids, _ = asset.find_bodies(["flap_right", "flap_left"], preserve_order=True)
            if len(body_ids) != 2:
                raise ValueError(f"{name} is missing flap_right/flap_left rigid bodies.")
            geometry = box_geometry(
                getattr(env.cfg.scene, name), ("flap_right", "flap_left"))
            part_ids, _ = asset.find_bodies(
                ["Body", "flap_right", "flap_left"], preserve_order=True)
            if len(part_ids) != 3:
                raise ValueError(f"{name} is missing a body or flap rigid link.")
            parts = (geometry, geometry.flaps["flap_right"], geometry.flaps["flap_left"])
            self.part_ids.append(part_ids)
            self.part_centers.append(torch.tensor(
                [part.center for part in parts], device=env.device))
            self.part_halves.append(torch.tensor(
                [part.half_size for part in parts], device=env.device))
            flap_geometry = [geometry.flaps[key] for key in ("flap_right", "flap_left")]
            self.flap_ids.append(body_ids)
            self.flap_centers.append([value.center for value in flap_geometry])
            self.flap_halves.append([value.half_size for value in flap_geometry])
            self.flap_normal_axes.append([value.half_size.index(min(value.half_size))
                                          for value in flap_geometry])
        self.target_logical_id = -1
        self.initial_box_z = torch.zeros(env.cfg.multi_box.max_boxes, device=env.device)
        self.initial_rack_y = torch.zeros_like(self.initial_box_z)
        self.reset()

    def _active_ids(self) -> list[int]:
        return self.env._multi_box_active[0].nonzero(as_tuple=False).flatten().tolist()

    def _asset_for_logical(self, logical_id: int):
        pool_id = int(self.env._multi_box_pool_ids[0, logical_id].item())
        if pool_id < 0:
            raise ValueError(f"Logical box {logical_id} is inactive.")
        return pool_id, self.env.scene[self.names[pool_id]]

    def _rack_local_pose(self, world_pose: torch.Tensor) -> torch.Tensor:
        return relative_pose(self.env.scene["rack"].data.root_pose_w[0], world_pose)

    def _choose_nearest(self) -> int:
        active = self._active_ids()
        if not active:
            raise RuntimeError("The randomized v2 scene has no active boxes.")
        tcp_midpoint = self.tcp.center_pose_w[0, :, :3].mean(0)
        distances = []
        for logical_id in active:
            _, asset = self._asset_for_logical(logical_id)
            distances.append((asset.data.root_pos_w[0] - tcp_midpoint).norm())
        return active[int(torch.stack(distances).argmin().item())]

    def reset(self) -> None:
        if not hasattr(self.env, "_multi_box_active"):
            return
        for logical_id in self._active_ids():
            _, asset = self._asset_for_logical(logical_id)
            pose = asset.data.root_pose_w[0]
            self.initial_box_z[logical_id] = pose[2]
            self.initial_rack_y[logical_id] = self._rack_local_pose(pose)[1]
        self.target_logical_id = self._choose_nearest()

    def cycle_target(self, direction: int) -> int:
        active = self._active_ids()
        if not active:
            raise RuntimeError("The randomized v2 scene has no active boxes.")
        if self.target_logical_id not in active:
            self.target_logical_id = active[0]
        else:
            index = active.index(self.target_logical_id)
            self.target_logical_id = active[(index + direction) % len(active)]
        return self.target_logical_id

    def _grasp_metrics(self, pool_id: int, asset, box_pose: torch.Tensor):
        device = self.env.device
        flap_ids = self.flap_ids[pool_id]
        flap_pos = asset.data.body_link_pos_w[0, flap_ids]
        flap_quat = asset.data.body_link_quat_w[0, flap_ids]
        centers = torch.tensor(self.flap_centers[pool_id], device=device)
        halves = torch.tensor(self.flap_halves[pool_id], device=device)
        axes = torch.tensor(self.flap_normal_axes[pool_id], device=device, dtype=torch.long)

        tcp = self.tcp.center_pose_w[0, :, :3]
        pair_pose = torch.cat((
            flap_pos[None].expand(2, -1, -1),
            flap_quat[None].expand(2, -1, -1),
        ), dim=-1)
        tcp_pose = torch.cat((tcp[:, None].expand(-1, 2, -1),
                              flap_quat[None].expand(2, -1, -1)), dim=-1)
        tcp_local = relative_pose(pair_pose, tcp_pose)[..., :3]
        nearest = _closest_box_surface(
            tcp_local, centers[None], halves[None], axes[None])
        distance = (tcp_local - nearest).norm(dim=-1)
        _, assignment = opposing_flap_reach_assignment(
            distance[None], GRASP_APPROACH_REWARD_SCALE_M)
        assignment = assignment[0]
        hands = torch.arange(2, device=device)

        normals_local = torch.nn.functional.one_hot(axes, 3).to(tcp.dtype)
        normals_world = quat_apply(flap_quat, normals_local)
        assigned_normal = normals_world[assignment]
        fingers = (self.tcp.tips_w[0] if self.tcp.definition else
                   self.robot.data.body_link_pos_w[0, self.finger_ids].reshape(2, 2, 3))
        closing = fingers[:, 0] - fingers[:, 1]
        closing = closing / closing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        alignment_cos = (closing * assigned_normal).sum(-1).abs().clamp(0, 1)
        alignment_error = torch.acos(alignment_cos).mean().reshape(1)

        assigned_pose = torch.cat((flap_pos[assignment], flap_quat[assignment]), dim=-1)
        finger_pose = torch.cat((
            fingers,
            flap_quat[assignment, None].expand(-1, 2, -1),
        ), dim=-1)
        finger_local = relative_pose(assigned_pose[:, None], finger_pose)[..., :3]
        assigned_center = centers[assignment]
        assigned_half = halves[assignment]
        assigned_axis = axes[assignment]
        delta = finger_local - assigned_center[:, None]
        signed = delta.gather(
            -1, assigned_axis[:, None, None].expand(-1, 2, 1)).squeeze(-1)
        normal_outside = signed.amin(-1).clamp_min(0) + (-signed.amax(-1)).clamp_min(0)
        midpoint_delta = finger_local.mean(1) - assigned_center
        tangent_excess = (midpoint_delta.abs() - assigned_half).clamp_min(0)
        tangent_excess.scatter_(1, assigned_axis[:, None], 0.0)
        capture_by_hand = torch.sqrt(
            normal_outside.square() + tangent_excess.square().sum(-1))
        capture_error = capture_by_hand.mean().reshape(1)
        jaw_gap = (signed[:, 0] - signed[:, 1]).abs()
        thickness = 2.0 * assigned_half.gather(-1, assigned_axis[:, None]).squeeze(-1)
        proof_lift = (box_pose[2] - self.initial_box_z[self.target_logical_id]).reshape(1)
        raw = GraspRawMetrics(
            matched_flap_distance_m=distance[hands, assignment].mean().reshape(1),
            jaw_alignment_error_rad=alignment_error,
            capture_error_m=capture_error,
            proof_lift_m=proof_lift,
        )
        potentials = grasp_reward_potentials(
            distance[None], distance[hands, assignment][None],
            alignment_cos[None], capture_by_hand[None], jaw_gap[None],
            thickness[None], proof_lift,
        )
        return raw, potentials

    def _all_active_footprints(self, belt_pose: torch.Tensor):
        result = {}
        for logical_id in self._active_ids():
            pool_id, asset = self._asset_for_logical(logical_id)
            body_id = self.part_ids[pool_id][0]
            body_pose = torch.cat((
                asset.data.body_link_pos_w[0, body_id],
                asset.data.body_link_quat_w[0, body_id],
            ))
            corners, bottom = _body_footprint_corners_belt(
                body_pose, belt_pose,
                self.part_centers[pool_id][0], self.part_halves[pool_id][0],
            )
            result[logical_id] = (corners, bottom)
        return result

    def _contact_diagnostics(self, pool_id: int, asset) -> tuple[dict[str, float], FingerFlapContacts]:
        filter_count = len(self.names) * 2
        force_rows, point_rows = [], []
        try:
            for sensor_name in CONTACT_SENSOR_NAMES:
                data = self.env.scene[sensor_name].data
                if data.force_matrix_w is None or data.contact_pos_w is None:
                    raise RuntimeError("filtered contact tensors are unavailable")
                force = data.force_matrix_w[0, 0]
                point = data.contact_pos_w[0, 0]
                if force.shape != (filter_count, 3) or point.shape != (filter_count, 3):
                    raise RuntimeError(
                        f"unexpected contact tensor shapes force={force.shape}, point={point.shape}")
                force_rows.append(force.reshape(len(self.names), 2, 3)[pool_id])
                point_rows.append(point.reshape(len(self.names), 2, 3)[pool_id])
        except (KeyError, RuntimeError, TypeError, IndexError):
            shape = (1, 2, 2, 2)
            return {"contact_adapter_available": 0.0}, FingerFlapContacts(
                force_n=torch.zeros(shape, device=self.env.device),
                in_region=torch.zeros(shape, dtype=torch.bool, device=self.env.device),
                opposed=torch.zeros(shape[:3], dtype=torch.bool, device=self.env.device),
                available=torch.zeros(1, dtype=torch.bool, device=self.env.device),
            )

        # Source order is L-front, L-back, R-front, R-back.  Reorder to
        # [hand, candidate flap, jaw, xyz].
        force_vector = torch.stack(force_rows).reshape(2, 2, 2, 3).permute(0, 2, 1, 3)
        force_n = force_vector.norm(dim=-1)
        point_world = torch.stack(point_rows).reshape(2, 2, 2, 3).permute(0, 2, 1, 3)

        flap_ids = self.flap_ids[pool_id]
        flap_pos = asset.data.body_link_pos_w[0, flap_ids]
        flap_quat = asset.data.body_link_quat_w[0, flap_ids]
        centers = point_world.new_tensor(self.flap_centers[pool_id])
        halves = point_world.new_tensor(self.flap_halves[pool_id])
        axes = torch.tensor(self.flap_normal_axes[pool_id], device=self.env.device,
                            dtype=torch.long)
        flap_pose = torch.cat((flap_pos, flap_quat), dim=-1)[None, :, None].expand(2, -1, 2, -1)
        point_pose = torch.cat((point_world,
                                flap_quat[None, :, None].expand(2, -1, 2, -1)), dim=-1)
        point_local = relative_pose(flap_pose, point_pose)[..., :3]
        delta = point_local - centers[None, :, None]
        margin = 0.004
        in_region = torch.isfinite(point_local).all(-1) \
            & (delta.abs() <= halves[None, :, None] + margin).all(-1) \
            & (force_n > 0)
        signed_contact = delta.gather(
            -1, axes[None, :, None, None].expand(2, -1, 2, 1)).squeeze(-1)
        opposed = signed_contact[..., 0] * signed_contact[..., 1] < 0

        # Contact representatives on a thin collision shape can occasionally
        # share one face.  Report whether the physical jaw link origins still
        # straddle the flap, but never turn this diagnostic into success here.
        fingers = self.robot.data.body_link_pos_w[0, self.finger_ids].reshape(2, 2, 3)
        finger_world = fingers[:, None].expand(-1, 2, -1, -1)
        finger_pose = torch.cat((finger_world,
                                 flap_quat[None, :, None].expand(2, -1, 2, -1)), dim=-1)
        finger_local = relative_pose(flap_pose, finger_pose)[..., :3]
        finger_delta = finger_local - centers[None, :, None]
        signed_finger = finger_delta.gather(
            -1, axes[None, :, None, None].expand(2, -1, 2, 1)).squeeze(-1)
        opposed |= signed_finger[..., 0] * signed_finger[..., 1] < 0

        result = {"contact_adapter_available": 1.0}
        for hand_index, hand in enumerate(("l", "r")):
            for flap_index, flap in enumerate(("right", "left")):
                prefix = f"contact_{hand}_{flap}"
                result[f"{prefix}_front_n"] = float(force_n[hand_index, flap_index, 0].item())
                result[f"{prefix}_back_n"] = float(force_n[hand_index, flap_index, 1].item())
                result[f"{prefix}_front_region"] = float(
                    in_region[hand_index, flap_index, 0].item())
                result[f"{prefix}_back_region"] = float(
                    in_region[hand_index, flap_index, 1].item())
                result[f"{prefix}_opposed"] = float(opposed[hand_index, flap_index].item())
        return result, FingerFlapContacts(
            force_n=force_n.unsqueeze(0),
            in_region=in_region.unsqueeze(0),
            opposed=opposed.unsqueeze(0),
            available=torch.ones(1, dtype=torch.bool, device=self.env.device),
        )

    def _belt_body_force(self, pool_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        try:
            matrix = self.env.scene[BELT_CONTACT_SENSOR_NAMES[pool_id]].data.force_matrix_w
            if matrix is None or matrix.shape != (1, 1, 1, 3):
                raise RuntimeError("body-to-belt force matrix is unavailable")
        except (KeyError, RuntimeError, TypeError, IndexError):
            return (torch.zeros(1, device=self.env.device),
                    torch.zeros(1, dtype=torch.bool, device=self.env.device))
        return matrix[0, 0, 0].norm().reshape(1), torch.ones(
            1, dtype=torch.bool, device=self.env.device)

    def _pad_box_distance(self, pool_id: int, asset) -> torch.Tensor:
        finger_pos = self.robot.data.body_link_pos_w[0, self.finger_ids]
        finger_quat = self.robot.data.body_link_quat_w[0, self.finger_ids]
        pad_pos = finger_pos + quat_apply(finger_quat, self.pad_centers)
        pad_pose = torch.cat((pad_pos, finger_quat), dim=-1)
        part_ids = self.part_ids[pool_id]
        part_poses = torch.cat((asset.data.body_link_pos_w[0, part_ids],
                                asset.data.body_link_quat_w[0, part_ids]), dim=-1)
        distance = pad_to_boxes_clearance_m(
            pad_pose, self.pad_size_m, part_poses,
            self.part_centers[pool_id], self.part_halves[pool_id])
        return distance.reshape(1, 2, 2).amin(dim=-1)

    def measure(self) -> IsaacMetricSnapshot:
        if self.target_logical_id not in self._active_ids():
            self.reset()
        logical_id = self.target_logical_id
        pool_id, asset = self._asset_for_logical(logical_id)
        box_pose = asset.data.root_pose_w[0]
        belt_pose = self.env.scene["conveyor_surface"].data.root_pose_w[0]
        type_id = int(self.env._multi_box_box_type_ids[0, logical_id].item())
        box_type = self.type_names[type_id]
        region_id = int(self.env._multi_box_region_ids[0, logical_id].item())
        region = self.env.cfg.multi_box.region_names[region_id]

        grasp, grasp_values = self._grasp_metrics(pool_id, asset, box_pose)
        footprints = self._all_active_footprints(belt_pose)
        corners, bottom_height = footprints[logical_id]
        other = [value[0] for other_id, value in footprints.items()
                 if other_id != logical_id and -0.10 <= float(value[1]) <= 0.50]
        clearance = _signed_aabb_clearance(corners, other)
        belt_half = box_pose.new_tensor(BELT_HALF_EXTENTS_XY)
        outside = _footprint_outside(corners, belt_half)

        rack_local = self._rack_local_pose(box_pose)
        outward_progress = rack_local[1] - self.initial_rack_y[logical_id]
        extraction_remaining = (box_pose.new_tensor(EXTRACTION_GOAL_M) - outward_progress) \
            .clamp_min(0).reshape(1)
        carry = CarryRawMetrics(
            extraction_remaining_m=extraction_remaining,
            footprint_distance_to_belt_m=outside,
            free_space_clearance_m=clearance,
            box_bottom_height_m=bottom_height.reshape(1),
        )

        box_in_belt = relative_pose(belt_pose, box_pose)
        box_up_world = quat_apply(
            box_pose[3:].reshape(1, 4),
            box_pose.new_tensor([[0.0, 0.0, 1.0]]),
        )[0]
        box_tilt = torch.acos(box_up_world[2].clamp(-1.0, 1.0)).reshape(1)
        long_axis = quat_apply(box_in_belt[3:].reshape(1, 4),
                               box_pose.new_tensor([[1.0, 0.0, 0.0]]))[0]
        angle = unsigned_axis_angle_error(torch.atan2(long_axis[1], long_axis[0]).reshape(1))
        velocity = asset.data.root_vel_w[0]
        place = PlaceRawMetrics(
            footprint_outside_m=outside,
            long_axis_error_rad=angle,
            free_space_clearance_m=clearance,
            box_bottom_height_m=bottom_height.reshape(1),
            linear_speed_mps=velocity[:3].norm().reshape(1),
            angular_speed_radps=velocity[3:].norm().reshape(1),
        )
        raw = {"grasp": grasp, "carry": carry, "place": place}
        potentials = {
            "grasp": grasp_values,
            "carry": carry_potentials(carry),
            "place": place_potentials(place),
        }
        diagnostics = {
            "footprint_inside": float(outside.item() <= 1e-6),
            "box_tilt_deg": math.degrees(float(box_tilt.item())),
            "long_axis_error_deg": math.degrees(float(angle.item())),
        }
        contact_diagnostics, contacts = self._contact_diagnostics(pool_id, asset)
        diagnostics.update(contact_diagnostics)
        belt_body_force_n, belt_sensor_available = self._belt_body_force(pool_id)
        gripper_box_distance_m = self._pad_box_distance(pool_id, asset)
        diagnostics["belt_body_force_n"] = float(belt_body_force_n.item())
        diagnostics["belt_sensor_available"] = float(belt_sensor_available.item())
        diagnostics["gripper_box_distance_left_m"] = float(gripper_box_distance_m[0, 0].item())
        diagnostics["gripper_box_distance_right_m"] = float(gripper_box_distance_m[0, 1].item())
        hand_to_box_pose = relative_pose(
            box_pose, self.tcp.center_pose_w[0]).unsqueeze(0)
        shelf = self.env.cfg.multi_box.rack_regions[region_id].shelf
        shelf_gap = box_shelf_clearance_m(
            box_pose.unsqueeze(0), self.env.scene["rack"].data.root_pose_w,
            BOX_DIMENSIONS_M[box_type], shelf=shelf, rack_scale=workcell_scale("rack"))
        # A spawned box begins with a small shelf gap, so the proof lift also
        # requires 8 mm of motion above its own reset position.
        rack_clearance = torch.minimum(grasp.proof_lift_m, shelf_gap)
        potentials["grasp"]["proof_lift"] = (rack_clearance / 0.008).clamp(0, 1)
        diagnostics["rack_shelf_gap_m"] = float(shelf_gap.item())
        diagnostics["rack_clearance_m"] = float(rack_clearance.item())
        return IsaacMetricSnapshot(
            target_logical_id=logical_id,
            target_asset_name=self.names[pool_id],
            target_box_type=box_type,
            target_region=region,
            raw_by_phase=raw,
            potentials_by_phase=potentials,
            diagnostics=diagnostics,
            contacts=contacts,
            hand_to_box_pose=hand_to_box_pose,
            rack_clearance_m=rack_clearance,
            box_footprint_corners_belt=corners.unsqueeze(0),
            box_bottom_height_m=bottom_height.reshape(1),
            box_tilt_rad=box_tilt,
            overlaps_other_belt_box=(clearance < 0),
            belt_body_force_n=belt_body_force_n,
            belt_sensor_available=belt_sensor_available,
            gripper_box_distance_m=gripper_box_distance_m,
        )
