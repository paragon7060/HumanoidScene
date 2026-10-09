"""Known static rack/body proxies for optional, TRAIN-only soft IK cost.

No simulator, contact labels, reward, Q or evaluation data are read. Approximate
geometry does not replace physical collision termination or prove safe motion.
"""
import hashlib
import json
from pathlib import Path

import torch

from ....core.paths import ASSET_DIR
from ....workcell.workcell_layout import scale as workcell_scale
from ..demo_replay import _rotation_matrix
from .tensor_arm_kinematics import rotation

GEOMETRY_PATH = Path(__file__).resolve().parents[1] / 'geometry/known_rack_entry_proxies.json'
GEOMETRY_SHA256 = '2361701aa4b2c8e502cf2f64a841a9c5d284a60d655eaaac9c9f45d2afae5cd2'
CLEARANCE_MARGIN_M = .010
CLEARANCE_ACTIVATION_M = .015
CLEARANCE_RECOVERY_CAP_M = .002
CLEARANCE_WEIGHT = 4.


def rack_clearance_contract():
    return dict(name='known_static_rack_whole_palm_forearm_soft_IK_cost_v1',
        authored_geometry_SHA256=GEOMETRY_SHA256, static_boxes=19,
        palm_convex_hull_vertices_per_hand=247, forearm_axis_samples_per_hand=33,
        forearm_swept_radius_m=.05, rack_scale=[1., 1., 1.],
        margin_m=CLEARANCE_MARGIN_M, activation_distance_m=CLEARANCE_ACTIVATION_M,
        outward_recovery_cap_m_per_tick=CLEARANCE_RECOVERY_CAP_M,
        least_squares_weight=CLEARANCE_WEIGHT,
        pending_arm_target_lead_in_geometry_reference=True,
        geometry_linearization_at_existing_projected_pending_arm_goal=True,
        pending_lead_projected_through_existing_80mrad_and_URDF_goal_bounds=True,
        local_goal_sensitivity_zero_for_saturated_pending_joint=True,
        measured_joint_rack_pose_and_known_asset_geometry_only=True,
        torso_finite_difference_same_fixed_pitch_IK=True,
        approximate_geometry_NOT_physical_collision_verification=True,
        rollers_boxes_obstacles_self_collision_NOT_in_this_soft_cost=True,
        original_force_termination_limits_close_gates_DR_and_success_preserved=True,
        scope='original_selected20percent_guided_TRAIN_rows_only', no_extra_RNG=True)


def box_signed_distance_gradient(points, centers, rotations, half_sizes):
    """Point/OBB signed distance and a deterministic subgradient in asset frame."""
    local = torch.einsum('npbj,bji->npbi', points[:, :, None] - centers, rotations)
    excess = local.abs() - half_sizes
    outside = excess.clamp_min(0)
    length = outside.norm(dim=-1)
    maximum, axis = excess.max(-1)
    distance = length + maximum.clamp_max(0)
    signs = torch.where(local >= 0, 1., -1.)
    gradient = signs * outside / length.clamp_min(1e-12)[..., None]
    inside_gradient = torch.nn.functional.one_hot(axis, 3).to(local) * signs
    gradient = torch.where((length > 1e-12)[..., None], gradient, inside_gradient)
    return distance, torch.einsum('npbi,bji->npbj', gradient, rotations)


def projected_pending_arm_lead(raw, kinematics):
    """Existing goal clipping at zero increment, and its local dq sensitivity."""
    q = raw[:, :20][:, kinematics.columns].reshape(len(raw), 14)
    offset = raw[:, 416:436][:, kinematics.columns].reshape(len(raw), 14)
    low, high = kinematics.lower.flatten() + .01, kinematics.upper.flatten() - .01
    lead_goal = q + offset.clamp(-.08, .08)
    target = lead_goal.maximum(low).minimum(high)
    sensitive = (offset.abs() < .08) & (lead_goal > low) & (lead_goal < high)
    return target - q, sensitive.to(raw)


def regularize_clearance(H, b, gradients, distances, pending_lead, *, pending_geometry_gradients=None):
    """Add a soft margin cost, including the already commanded arm target lead.

Farther-than-margin active points permit inward motion toward the margin. Only
outward recovery is capped; this does not turn all nearby points into repulsion.
The caller still applies its original joint/torso limits to the solved proposal.
"""
    if H.ndim != 3 or H.shape[-2:] != (b.shape[1], b.shape[1]) \
            or gradients.shape != (len(H), 4, b.shape[1]) \
            or distances.shape != (len(H), 4) or pending_lead.shape != b.shape \
            or any(not torch.isfinite(x).all() for x in (H, b, gradients, distances, pending_lead)):
        raise ValueError('Finite matched task and four body-proxy rows required')
    lead_gradients = gradients if pending_geometry_gradients is None else pending_geometry_gradients
    if lead_gradients.shape != gradients.shape or not torch.isfinite(lead_gradients).all():
        raise ValueError('Finite matched projected pending-goal geometry derivatives required')
    lead_distance = (lead_gradients * pending_lead[:, None]).sum(-1)
    active = (distances < CLEARANCE_ACTIVATION_M) | \
             (distances + lead_distance < CLEARANCE_ACTIVATION_M)
    desired = (CLEARANCE_MARGIN_M - distances).clamp_max(CLEARANCE_RECOVERY_CAP_M)
    residual = desired - lead_distance
    weighted = gradients * (active.to(H) * CLEARANCE_WEIGHT)[..., None]
    return (H + weighted.transpose(-1, -2) @ gradients,
            b + (weighted.transpose(-1, -2) @ residual[..., None]).squeeze(-1), active)


class RackEntryClearance:
    def __init__(self, kinematics):
        if hashlib.sha256(GEOMETRY_PATH.read_bytes()).hexdigest() != GEOMETRY_SHA256:
            raise ValueError('Known rack/body geometry fingerprint differs')
        config = json.loads(GEOMETRY_PATH.read_text())
        for relative, expected in config['source_assets_SHA256'].items():
            p = ASSET_DIR / relative
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected:
                raise ValueError('Known rack/body source asset differs: ' + relative)
        if list(workcell_scale('rack')) != config['rack_scale']:
            raise ValueError('This reviewed static rack proxy requires the original unit rack scale')
        self.kinematics = kinematics
        tensor = kinematics.tensor
        self.centers = tensor([v['center_m'] for v in config['static_boxes']])
        self.rotations = tensor([v['rotation_asset_from_box'] for v in config['static_boxes']])
        self.halves = tensor([v['half_size_m'] for v in config['static_boxes']])
        self.names = [v['name'] for v in config['static_boxes']]
        self.palms = [tensor(v['palm_vertices_from_closed_TCP_m']) for v in config['hands']]
        self.lines = [tensor(v['forearm_axis_points_from_link4_m']) for v in config['hands']]
        self.radii = [v['forearm_radius_m'] for v in config['hands']]
        self.guided_rows = 0
        self.active_body_rows = 0
        self.minimum_nominal_proxy_distance_m = None

    def forearm_fk(self, q):
        kin = self.kinematics
        # fk performs the same device/shape/finite checks before this method is
        # reached. Keep the default TCP fk arithmetic entirely unchanged.
        p = q.new_zeros(len(q), 3)
        R = torch.eye(3, device=q.device, dtype=q.dtype).expand(len(q), 3, 3).clone()
        for xyz, origin_R, kind, column, axis in kin.parent_chain:
            p = p + (R @ xyz[:, None]).squeeze(-1); R = R @ origin_R
            if kind == 'revolute': R = R @ rotation(axis, q[:, column])
            elif kind == 'prismatic': p = p + (R @ axis[:, None]).squeeze(-1) * q[:, column, None]
        positions, rotations, jacobians = [], [], []
        for columns, chain, arm in zip(kin.columns, kin.chains, kin.arms):
            tip, orientation = p.clone(), R.clone(); axes, origins = [], []
            found = False
            for joint, (xyz, origin_R, axis) in zip(arm.joints, chain):
                tip = tip + (orientation @ xyz[:, None]).squeeze(-1); orientation = orientation @ origin_R
                if axis is not None:
                    origins.append(tip); axes.append((orientation @ axis[:, None]).squeeze(-1))
                    orientation = orientation @ rotation(axis, q[:, columns[len(axes) - 1]])
                if joint.child == 'zarm_' + arm.side[0] + '4_link': found = True; break
            if not found or len(axes) != 4: raise ValueError('Reviewed four-joint forearm chain required')
            axes, origins = torch.stack(axes, -2), torch.stack(origins, -2)
            J = q.new_zeros(len(q), 6, 7)
            J[:, :3, :4] = torch.linalg.cross(axes, tip[:, None] - origins).transpose(-1, -2)
            J[:, 3:, :4] = axes.transpose(-1, -2)
            positions.append(tip); rotations.append(orientation); jacobians.append(J)
        return torch.stack(positions, 1), torch.stack(rotations, 1), torch.stack(jacobians, 1)

    def frames(self, q):
        tcp = self.kinematics.fk(q)
        forearm = self.forearm_fk(q)
        return [(tcp[0][:, h], tcp[1][:, h], tcp[2][:, h]) for h in range(2)] + \
               [(forearm[0][:, h], forearm[1][:, h], forearm[2][:, h]) for h in range(2)]

    @torch.no_grad()
    def measure(self, q, rack_pose):
        if rack_pose.shape != (len(q), 9) or not torch.isfinite(rack_pose).all():
            raise ValueError('Finite measured rack position and rotation required')
        rack_R = _rotation_matrix(rack_pose[:, 3:])
        frames = self.frames(q); distances, gradients, local_points, root_normals = [], [], [], []
        closest_boxes = []
        for body, (p, R, J) in enumerate(frames):
            h = body % 2; local = self.palms[h] if body < 2 else self.lines[h]
            offset = torch.einsum('nij,pj->npi', R, local)
            points = p[:, None] + offset
            asset_points = torch.einsum('nij,npj->npi', rack_R.transpose(-1, -2), points - rack_pose[:, None, :3])
            ds, normals = box_signed_distance_gradient(asset_points, self.centers, self.rotations, self.halves)
            if body >= 2: ds = ds - self.radii[h]
            minimum, index = ds.flatten(1).min(-1)
            sample, box = index // len(self.centers), index % len(self.centers)
            rows = torch.arange(len(q), device=q.device)
            normal = (rack_R @ normals[rows, sample, box, :, None]).squeeze(-1)
            chosen_offset = offset[rows, sample]
            Jpoint = J[:, :3] + torch.linalg.cross(J[:, 3:].transpose(-1, -2), chosen_offset[:, None]).transpose(-1, -2)
            g = q.new_zeros(len(q), 14)
            g[:, h * 7:(h + 1) * 7] = (normal[:, :, None] * Jpoint).sum(1)
            distances.append(minimum); gradients.append(g); local_points.append(local[sample])
            root_normals.append(normal); closest_boxes.append(box)
        return dict(distances=torch.stack(distances, 1), gradients=torch.stack(gradients, 1),
            local_points=torch.stack(local_points, 1), root_normals=torch.stack(root_normals, 1),
            closest_boxes=torch.stack(closest_boxes, 1), frames=frames)

    def selected_points(self, q, local_points):
        return torch.stack([p + (R @ local_points[:, i, :, None]).squeeze(-1)
                            for i, (p, R, _) in enumerate(self.frames(q))], 1)

    def record(self, measured, active):
        self.guided_rows += len(active); self.active_body_rows += int(active.sum())
        value = float(measured['distances'].min())
        self.minimum_nominal_proxy_distance_m = value if self.minimum_nominal_proxy_distance_m is None \
            else min(value, self.minimum_nominal_proxy_distance_m)

    def report(self):
        return dict(scope='current_collection_wave_only_NOT_resume_cumulative',
            guided_rows=self.guided_rows, active_body_rows=self.active_body_rows,
            minimum_nominal_proxy_distance_m=self.minimum_nominal_proxy_distance_m,
            distance_reference='existing_projected_pending_arm_goal_with_measured_torso',
            approximate_distances_NOT_contact_force_or_safe_motion_evidence=True)
