"""Stationary flap grasp: shared candidate surfaces and physical contact tests."""

import torch
from .grasp_contact_latch import GraspContactLatch, opposed_jaws


def upper_band_contacts(points, centers, halves, normal_axes, *, band, margin):
    """Per-finger contacts in flap-local metres; absent contacts (NaN) are false.

    points: (env, hand, jaw, xyz); centers/halves: (env, hand, xyz).
    Test both sides of the same flap, not just any two contacts near the box.
    """
    delta = points - centers[:, :, None]
    in_bounds = (delta.abs() <= halves[:, :, None] + margin).all(-1)
    near_top = True if band is None else delta[..., 2] >= halves[:, :, None, 2] - band
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


def closest_flap_surface(points, centers, halves, normal_axes):
    """Closest point on either broad face, including its boundary; all in flap-local metres.

    Arbitrary leading dimensions are allowed (typically env, hand, candidate).
    Unlike a solid-AABB distance, a point inside the thin plate is not distance zero.
    """
    delta = points - centers
    nearest = delta.clamp(-halves, halves)
    normal = normal_axes[..., None]
    side = torch.where(delta.gather(-1, normal) >= 0, 1., -1.)
    nearest = nearest.scatter(-1, normal, side * halves.gather(-1, normal))
    return centers + nearest


def choose_contact_candidate(held, distances):
    """Retain a held flap's identity; otherwise use the closest surface for diagnostics."""
    return torch.where(held.any(-1), held.to(torch.int64).argmax(-1), distances.argmin(-1))


class FlapGrasp:
    """Each hand independently measures BOTH candidate flaps, without mixing jaw forces."""

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
        if task.spec.flap_contact_region == "top_band" and (self.halves[..., 2] * 2 <= task.spec.flap_top_band).any():
            raise ValueError("Flap top band must be smaller than the physical flap height.")
        self.normal_axes = self.halves.argmin(-1)
        if (self.normal_axes == 2).any():
            raise ValueError("Flap grasp expects an upright thin plate whose local +Z is the upper edge.")
        self.finger_ids, _ = task.robot.find_bodies(list(task.spec.finger_bodies), preserve_order=True)
        self.latch = GraspContactLatch(task.num_envs, task.device, task.spec, channels=4)

    def reset(self, env_ids):
        self.latch.reset(env_ids)

    def measure(self):
        from .geometry import rotate, unrotate
        t = self.task
        positions = torch.stack([b.data.body_link_pos_w[:, ids]
                                 for b, ids in zip(t.boxes, self.body_ids)], dim=1)
        quats = torch.stack([b.data.body_link_quat_w[:, ids]
                            for b, ids in zip(t.boxes, self.body_ids)], dim=1)
        pos = positions[t.ids, t.active_box]  # env, candidate, xyz
        quat = quats[t.ids, t.active_box]
        centers, halves = self.centers[t.active_box], self.halves[t.active_box]
        axes = self.normal_axes[t.active_box]
        # Geometry for every hand/candidate pair. Closest surface always determines
        # distance, even while a contact latch retains a different flap identity.
        n = t.num_envs
        pair_pos = pos[:, None].expand(-1, 2, -1, -1)
        pair_q = quat[:, None].expand(-1, 2, -1, -1)
        pair_c = centers[:, None].expand(-1, 2, -1, -1)
        pair_h = halves[:, None].expand(-1, 2, -1, -1)
        pair_a = axes[:, None].expand(-1, 2, -1)
        tool_local = unrotate(pair_q, t.tools[:, :, None] - pair_pos)
        nearest = closest_flap_surface(tool_local, pair_c, pair_h, pair_a)
        targets = pair_pos + rotate(pair_q, nearest)
        distances = (t.tools[:, :, None] - targets).norm(dim=-1)
        nearest_id = distances.argmin(-1)
        batch = t.ids[:, None]
        hands = torch.arange(2, device=t.device)[None]
        t.grips = targets[batch, hands, nearest_id]
        t.flap_pos = pair_pos[batch, hands, nearest_id]
        t.flap_quat = pair_q[batch, hands, nearest_id]
        t.hand_target_distance = distances.amin(-1)
        t.nearest_flap_index = nearest_id
        t.grasp_candidate_distance = distances
        t.reach_distance = t.hand_target_distance[:, t.spec.grasp_hand_indices].mean(-1)
        normal = rotate(pair_q, torch.nn.functional.one_hot(pair_a, 3).float())
        fingers = t.robot.data.body_link_pos_w[:, self.finger_ids].reshape(t.num_envs, 2, 2, 3)
        jaw_q = pair_q[:, :, :, None].expand(-1, -1, -1, 2, -1)
        jaw_local = unrotate(jaw_q, fingers[:, :, None] - pair_pos[:, :, :, None])
        closing = fingers[:, :, 0] - fingers[:, :, 1]
        closing /= closing.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        alignment = (closing[:, :, None] * normal).sum(-1).abs().clamp(0, 1)
        t.grasp_alignment = alignment[batch, hands, nearest_id]
        points, forces, residuals = [], [], []
        for i in range(4):
            sensor = t._env.scene[f"grasp_contact_{i}"].data
            if sensor.contact_pos_w is None:
                raise RuntimeError("flap_top needs track_contact_points=True on each filtered finger sensor.")
            # Filter order: box0/flap0, box0/flap1, box1/flap0, ...
            points.append(sensor.contact_pos_w[:, 0].reshape(n, t.n, 2, 3)[t.ids, t.active_box])
            assigned = sensor.force_matrix_w[:, 0].reshape(n, t.n, 2, 3)[t.ids, t.active_box]
            forces.append(assigned.norm(dim=-1))
            residual = (sensor.net_forces_w[:, 0] - assigned.sum(-2)).norm(dim=-1)
            # The support hand may touch any part of the box. Obstacle contacts
            # are measured independently on every robot body.
            residuals.append(residual if i // 2 in t.spec.grasp_hand_indices else torch.zeros_like(residual))
        points = torch.stack(points, 1).reshape(n, 2, 2, 2, 3).transpose(2, 3)
        force = torch.stack(forces, 1).reshape(n, 2, 2, 2).transpose(2, 3)  # env, hand, flap, jaw
        local = unrotate(jaw_q, points - pair_pos[:, :, :, None])
        flat_local = local.reshape(n, 4, 2, 3)
        flat_c, flat_h, flat_a = pair_c.reshape(n, 4, 3), pair_h.reshape(n, 4, 3), pair_a.reshape(n, 4)
        flat_jaw = jaw_local.reshape(n, 4, 2, 3)
        band = t.spec.flap_top_band if t.spec.flap_contact_region == "top_band" else None
        valid, opposed = upper_band_contacts(flat_local, flat_c, flat_h, flat_a,
            band=band, margin=t.spec.flap_contact_margin)
        # Contact positions are sensor representatives, not guaranteed to lie on
        # opposite signed faces of a thin collision plate. If both filtered
        # allowed-region contacts exist, actual jaw placement may establish opposition.
        opposed |= opposed_jaws(flat_jaw, flat_c, flat_a)
        valid, opposed = valid.reshape(n, 2, 2, 2), opposed.reshape(n, 2, 2)
        valid_force = valid & (force > t.spec.grasp_force)
        strict = valid_force.all(-1) & opposed
        # Optional top-band acquisition can slide slightly down while holding.
        # Surface mode never expands outside the actual plate bounds + margin.
        hold_valid, _ = upper_band_contacts(flat_local, flat_c, flat_h, flat_a,
            band=None if band is None else band + t.spec.grasp_hold_slip_m, margin=t.spec.flap_contact_margin)
        sustained = (hold_valid.reshape(n, 2, 2, 2) & (force > t.spec.grasp_force * .5)).all(-1) & opposed
        # One latch per hand/flap. Never borrow grace from another candidate.
        previous = self.latch.active.reshape(n, 2, 2)
        acquire_id = distances.masked_fill(~strict, torch.inf).argmin(-1)
        acquisition = torch.nn.functional.one_hot(acquire_id, 2).bool() & strict
        acquisition &= ~previous.any(-1, keepdim=True)
        strict_latch = (strict & previous) | acquisition
        held = self.latch.update(strict_latch.reshape(n, 4), sustained.reshape(n, 4), flat_jaw,
                                 t._env.common_step_counter, t._env.step_dt).reshape(n, 2, 2)
        choice = choose_contact_candidate(held, distances)
        t.contact_flap_index = choice
        t.grasp_candidate_force, t.grasp_candidate_region = force, valid
        t.grasp_candidate_opposed, t.grasp_candidate_raw, t.grasp_candidate_held = opposed, strict, held
        t.grasp_candidate_missing_s = self.latch.missing_s.reshape(n, 2, 2).clone()
        has_reference = self.latch.reference_gap > 0
        slip = (flat_jaw.mean(-2) - self.latch.reference_midpoint).norm(dim=-1)
        opening = (flat_jaw[:, :, 0] - flat_jaw[:, :, 1]).norm(dim=-1) - self.latch.reference_gap
        t.grasp_candidate_slip_m = torch.where(has_reference, slip, 0.).reshape(n, 2, 2)
        t.grasp_candidate_opening_m = torch.where(has_reference, opening, 0.).reshape(n, 2, 2)
        t.grasp_candidate_contact_local = local  # debugger: env, hand, flap, jaw, xyz (metres)
        t.contact_force = force[batch, hands, choice].reshape(n, 4)
        t.grasp_contact_in_band = valid[batch, hands, choice]
        t.grasp_jaws_opposed = opposed[batch, hands, choice]
        t.finger_grasp_contacts = valid_force[batch, hands, choice]
        t.raw_hand_grasp_flags = strict.any(-1)
        t.hand_grasp_flags = held.any(-1)
        t.grasped = t.hand_grasp_flags[:, t.spec.grasp_hand_indices].all(-1)
        t.grasp_contact_missing_s = t.grasp_candidate_missing_s[batch, hands, choice]
        t.unexpected_finger_force = torch.stack(residuals, dim=-1)
        t.released = (force < t.spec.grasp_force).all((1, 2, 3)) & ~t.grasped
