"""Stationary flap-top grasp: physical constraints and geometric contact tests."""

import torch


def upper_band_contacts(points, centers, halves, normal_axes, *, band, margin):
    """Per-finger contacts in flap-local metres; absent contacts (NaN) are false.

    points: (env, hand, jaw, xyz); centers/halves: (env, hand, xyz).
    Test both sides of the same flap, not just any two contacts near the box.
    """
    delta = points - centers[:, :, None]
    in_bounds = (delta.abs() <= halves[:, :, None] + margin).all(-1)
    near_top = delta[..., 2] >= halves[:, :, None, 2] - band
    valid = torch.isfinite(points).all(-1) & in_bounds & near_top
    signed = delta.gather(-1, normal_axes[:, :, None, None].expand(-1, -1, 2, 1)).squeeze(-1)
    opposed = signed[..., 0] * signed[..., 1] < 0
    return valid, opposed


def grasp_status(valid, opposed, contact_force, spec):
    """Require opposed jaws on each selected hand; the other hand may support."""
    fingers = valid & (contact_force.reshape(-1, 2, 2) > spec.grasp_force)
    hands = fingers.all(-1) & opposed
    grasped = hands[:, spec.grasp_hand_indices].all(-1)
    return fingers, hands, grasped


class FlapGrasp:
    """Measure designated flap links and filtered finger contact positions."""

    def __init__(self, task):
        self.task = task
        self.body_ids = []
        centers, halves = [], []
        for name, box in zip(task.spec.box_names, task.boxes):
            ids, _ = box.find_bodies(list(task.spec.grasp_flaps), preserve_order=True)
            if len(ids) != 2:
                raise ValueError(f"Missing target flap bodies: {name}")
            self.body_ids.append(ids)
            geometry = task.cfg.geometry[name].flaps
            centers.append([geometry[f].center for f in task.spec.grasp_flaps])
            halves.append([geometry[f].half_size for f in task.spec.grasp_flaps])
        self.centers = torch.tensor(centers, device=task.device)
        self.halves = torch.tensor(halves, device=task.device)
        if (self.halves[..., 2] * 2 <= task.spec.flap_top_band).any():
            raise ValueError("Flap top band must be smaller than the physical flap height.")
        self.normal_axes = self.halves.argmin(-1)
        if (self.normal_axes == 2).any():
            raise ValueError("Flap grasp expects an upright thin plate whose local +Z is the upper edge.")
        self.finger_ids, _ = task.robot.find_bodies(list(task.spec.finger_bodies), preserve_order=True)

    def measure(self):
        from .geometry import rotate, unrotate
        t = self.task
        positions = torch.stack([b.data.body_link_pos_w[:, ids]
                                 for b, ids in zip(t.boxes, self.body_ids)], dim=1)
        quats = torch.stack([b.data.body_link_quat_w[:, ids]
                            for b, ids in zip(t.boxes, self.body_ids)], dim=1)
        t.flap_pos = positions[t.ids, t.active_box]
        t.flap_quat = quats[t.ids, t.active_box]
        centers, halves = self.centers[t.active_box], self.halves[t.active_box]
        axes = self.normal_axes[t.active_box]
        targets = centers.clone()
        targets[..., 2] += halves[..., 2] - t.spec.flap_grasp_depth
        t.grips = t.flap_pos + rotate(t.flap_quat, targets)
        t.hand_target_distance = (t.tools - t.grips).norm(dim=-1)
        t.reach_distance = t.hand_target_distance[:, t.spec.grasp_hand_indices].mean(-1)
        normal = torch.nn.functional.one_hot(axes, 3).float()
        normal = rotate(t.flap_quat, normal)
        fingers = t.robot.data.body_link_pos_w[:, self.finger_ids].reshape(t.num_envs, 2, 2, 3)
        closing = fingers[:, :, 0] - fingers[:, :, 1]
        closing /= closing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        t.grasp_alignment = (closing * normal).sum(-1).abs().clamp(0, 1)
        points, residuals = [], []
        for i in range(4):
            sensor = t._env.scene[f"grasp_contact_{i}"].data
            if sensor.contact_pos_w is None:
                raise RuntimeError("flap_top needs track_contact_points=True on each filtered finger sensor.")
            points.append(sensor.contact_pos_w[t.ids, 0, t.active_box])
            assigned = sensor.force_matrix_w[t.ids, 0, t.active_box]
            residual = (sensor.net_forces_w[:, 0] - assigned).norm(dim=-1)
            # The support hand may touch any part of the box. Obstacle contacts
            # are measured independently on every robot body.
            residuals.append(residual if i // 2 in t.spec.grasp_hand_indices else torch.zeros_like(residual))
        points = torch.stack(points, dim=1).reshape(t.num_envs, 2, 2, 3)
        local = unrotate(t.flap_quat[:, :, None].expand(-1, -1, 2, -1),
                         points - t.flap_pos[:, :, None])
        valid, opposed = upper_band_contacts(local, centers, halves, axes,
            band=t.spec.flap_top_band, margin=t.spec.flap_contact_margin)
        t.finger_grasp_contacts, t.hand_grasp_flags, t.grasped = grasp_status(
            valid, opposed, t.contact_force, t.spec)
        t.unexpected_finger_force = torch.stack(residuals, dim=-1)
        t.released = (t.contact_force < t.spec.grasp_force).all(-1)
