"""Display-only finger calibration and proposed paired grasp targets, never reward state."""

import math

import torch


def paired_face_targets(points, centers, halves, axes):
    """Local inputs: [..., jaw=2, xyz], [..., xyz], [...]. Returns matched opposite-face points.

    Use one tangential anchor (clamped jaw midpoint) for both faces. Choose the
    jaw/face assignment minimizing the worse finger distance, not independent flaps.
    """
    midpoint = points.mean(-2)
    anchor = (midpoint - centers).clamp(-halves, halves)
    axis = axes[..., None]
    normal = torch.nn.functional.one_hot(axes, 3).to(points.dtype)
    tangent = anchor.scatter(-1, axis, torch.zeros_like(axes[..., None], dtype=points.dtype))
    half_thickness = halves.gather(-1, axis)
    a = centers + tangent + half_thickness * normal
    b = centers + tangent - half_thickness * normal
    direct = torch.stack((a, b), -2)
    swapped = direct.flip(-2)
    cost = (points - direct).norm(dim=-1).amax(-1)
    reverse_cost = (points - swapped).norm(dim=-1).amax(-1)
    reverse = reverse_cost < cost
    return torch.where(reverse[..., None, None], swapped, direct), torch.minimum(cost, reverse_cost)


class GraspMarkers:
    """World-space RTX-visible primitives; no rigid bodies, collisions, or task writes."""

    def __init__(self, env, offsets, radius=.004, flap="auto", show_targets=True):
        import isaaclab.sim as sim
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        self.env = env
        from ...robots.end_effector import calibration_definition
        definition = calibration_definition()
        self.calibrated = definition is not None and not any(offsets)
        if self.calibrated:
            offsets = [v for name in ("r_f_finger", "r_b_finger") for v in definition["offsets"][name]]
        self.offsets = torch.tensor(offsets, device=env.device, dtype=torch.float32).reshape(2, 3)
        if not torch.isfinite(self.offsets).all() or self.offsets.abs().max() > .2:
            raise ValueError("Finger marker offsets must be finite, in metres, and within +/-0.2 m.")
        if not math.isfinite(radius) or not .001 <= radius <= .03:
            raise ValueError("Finger marker radius must be between 0.001 and 0.03 m.")
        self.flap = flap
        self.show_targets = show_targets
        self.visible = True
        self.legend = ("MARKERS: cyan r_f_finger / blue r_b_finger" +
                       ("; orange/pink goals" if show_targets else "; references ONLY"))
        self.offset_label = "\n".join(
            f"{name} local offset [m]: " + ", ".join(f"{v:.4f}" for v in values)
            for name, values in zip(("F", "B"), self.offsets.tolist()))
        colors = ((0., 1., 1.), (.15, .3, 1.), (1., .4, 0.), (1., .1, .6))
        markers = {}
        for index, color in enumerate(colors):
            material = sim.PreviewSurfaceCfg(diffuse_color=color, emissive_color=color)
            # Small goal cubes help distinguish points on a thin flap; do not offset
            # them away from their actual face just to make the view look separated.
            markers[f"point_{index}"] = (sim.SphereCfg(radius=radius, visual_material=material)
                                  if index < 2 else sim.CuboidCfg(size=(.003, .003, .003), visual_material=material))
        self.markers = VisualizationMarkers(VisualizationMarkersCfg(
            prim_path="/Visuals/RLGraspReferences", markers=markers))
        self.info = "DISPLAY ONLY: finger offsets not used by rewards"
        print(f"[GRASP MARKERS] offsets(F,B)={self.offsets.tolist()} m; flap={flap}. "
              f"Calibrated defaults={self.calibrated}. PC G toggles markers.", flush=True)
        self.update()

    def toggle(self):
        self.visible = not self.visible
        self.markers.set_visibility(self.visible)
        print(f"[GRASP MARKERS] {'ON' if self.visible else 'OFF'}", flush=True)

    def update(self):
        from ..mdp.geometry import rotate, unrotate
        if not self.visible:
            return
        t = self.env.command_manager.get_term("workcell")
        grasp = t.flap_grasp
        box_id = int(t.active_box[0].item())
        ids = grasp.finger_ids[2:4]  # right f finger, right b finger
        data = t.robot.data
        references = data.body_link_pos_w[0, ids] + rotate(data.body_link_quat_w[0, ids], self.offsets)
        if not self.show_targets:
            valid = bool(torch.isfinite(references).all())
            self.markers.set_visibility(valid)
            if valid:
                self.markers.visualize(translations=references, marker_indices=[0, 1])
                gap = (references[0] - references[1]).norm().item() * 1000
                self.info = f"DISPLAY ONLY: reference spacing={gap:.1f}mm\n" + self.offset_label
            else:
                self.info = "MARKERS: invalid finger pose; hidden"
            return
        box = t.boxes[box_id]
        flap_ids = grasp.body_ids[box_id]
        pos, quat = box.data.body_link_pos_w[0, flap_ids], box.data.body_link_quat_w[0, flap_ids]
        q = quat[:, None].expand(-1, 2, -1)
        local = unrotate(q, references[None].expand(2, -1, -1) - pos[:, None])
        goals, costs = paired_face_targets(local, grasp.centers[box_id], grasp.halves[box_id], grasp.normal_axes[box_id])
        if self.flap != "auto":
            if self.flap not in t.spec.grasp_flaps:
                raise ValueError(f"Marker flap {self.flap} is not in {t.spec.grasp_flaps}")
            candidate = t.spec.grasp_flaps.index(self.flap)
        elif bool(t.hand_grasp_flags[0, 1]):
            candidate = int(t.contact_flap_index[0, 1].item())
        else:
            candidate = int(costs.argmin().item())
        targets = pos[candidate] + rotate(quat[candidate].expand(2, -1), goals[candidate])
        positions = torch.cat((references, targets), 0)
        if not torch.isfinite(positions).all():
            self.markers.set_visibility(False)
            self.info = "MARKERS: invalid pose; hidden until valid"
            return
        self.markers.set_visibility(True)
        self.markers.visualize(translations=positions, marker_indices=[0, 1, 2, 3])
        distances = (references - targets).norm(dim=-1).mul(100).tolist()
        self.info = (f"PREVIEW {t.spec.grasp_flaps[candidate]} F/B={distances[0]:.1f}/{distances[1]:.1f}cm\n"
                     "DISPLAY ONLY: paired goals are not current reaching reward")

    def close(self):
        self.markers.set_visibility(False)
