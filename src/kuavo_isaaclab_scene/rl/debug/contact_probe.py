"""Per-link contact report and world markers for the Quest v2 demonstration run.

The training safety term aggregates every monitored link into one rack force
and one obstacle force, so an operator who sees "unsafe" cannot tell which link
touched what.  This probe reads the sensor buffers the termination manager
already updated in the same control step and reports the force per link.  It
never reconfigures sensors and never changes physics, rewards or termination.

Markers are drawn at the touching link's body origin, which is the exact link
the safety term charged but not the exact contact patch on its surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..multi_box.debug.contact_force import per_body_forces


RACK = "rack"
OBSTACLE = "obstacle"
MARKER_KINDS = (RACK, OBSTACLE)


@dataclass(frozen=True)
class LinkContact:
    """One robot link's contact with the rack or with any other obstacle."""

    body: str
    kind: str
    force_n: float
    position_w: tuple[float, float, float]


def link_contacts(bodies, rack_force_n, obstacle_force_n, positions_w, report_force_n):
    """Return the reportable link contacts of one environment, strongest first."""
    contacts = []
    for column, body in enumerate(bodies):
        position = tuple(round(float(value), 4) for value in positions_w[column])
        for kind, forces in ((RACK, rack_force_n), (OBSTACLE, obstacle_force_n)):
            force = float(forces[column])
            if force >= report_force_n:
                contacts.append(LinkContact(body, kind, force, position))
    return tuple(sorted(contacts, key=lambda contact: -contact.force_n))


def describe(contacts, limit: int = 3) -> str:
    """Summarise the strongest contacts for one console line."""
    if not contacts:
        return "no monitored link contact"
    shown = "; ".join(
        f"{contact.kind} {contact.body} {contact.force_n:.1f} N at "
        f"({contact.position_w[0]:.2f}, {contact.position_w[1]:.2f}, "
        f"{contact.position_w[2]:.2f}) m"
        for contact in contacts[:limit]
    )
    remaining = len(contacts) - limit
    return shown + (f"; +{remaining} more" if remaining > 0 else "")


class ContactProbe:
    """Read-only per-link view of the contacts behind the aggregate safety term."""

    def __init__(self, env, *, report_force_n=1.0, marker_radius=0.03, show_markers=True):
        # Import inside the constructor so the report helpers above stay usable
        # without Isaac Sim.
        from ..multi_box.debug.contact_sensors import (
            V2_COLLISION_BODY_NAMES,
            V2_OBSTACLE_SENSOR_NAME,
            V2_RACK_SENSOR_NAMES,
        )
        if float(report_force_n) <= 0.0:
            raise ValueError("Contact report threshold must be positive.")
        self.env = env
        self.report_force_n = float(report_force_n)
        self.obstacle_sensor = V2_OBSTACLE_SENSOR_NAME
        self.rack_sensors = V2_RACK_SENSOR_NAMES
        self.bodies = tuple(V2_COLLISION_BODY_NAMES)
        names = list(env.scene[V2_OBSTACLE_SENSOR_NAME].body_names)
        if set(names) != set(self.bodies):
            raise RuntimeError("V2 obstacle sensor covers different robot bodies.")
        self.sensor_indices = tuple(names.index(body) for body in self.bodies)
        robot = env.scene["robot"]
        self.robot_body_ids = [robot.find_bodies(body)[0][0] for body in self.bodies]
        self.latched = ()
        self.markers = None
        if show_markers:
            import isaaclab.sim as sim
            from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
            colors = {RACK: (1.0, 0.1, 0.1), OBSTACLE: (1.0, 0.55, 0.0)}
            self.markers = VisualizationMarkers(VisualizationMarkersCfg(
                prim_path="/Visuals/V2DemoContacts",
                markers={
                    kind: sim.SphereCfg(
                        radius=marker_radius,
                        visual_material=sim.PreviewSurfaceCfg(
                            diffuse_color=colors[kind], emissive_color=colors[kind]),
                    )
                    for kind in MARKER_KINDS
                },
            ))
            self.markers.set_visibility(False)

    def measure(self):
        """Report this control step's contacts from the already-updated buffers."""
        net_forces = self.env.scene[self.obstacle_sensor].data.net_forces_w
        matrices = tuple(
            self.env.scene[name].data.force_matrix_w for name in self.rack_sensors)
        if net_forces is None or any(matrix is None for matrix in matrices):
            return ()
        rack, obstacle = per_body_forces(net_forces, matrices, self.sensor_indices)
        positions = self.env.scene["robot"].data.body_pos_w[0, self.robot_body_ids]
        return link_contacts(
            self.bodies, rack[0], obstacle[0], positions, self.report_force_n)

    def show(self, contacts) -> None:
        """Track live contacts unless a terminal contact is currently latched."""
        if self.markers is None or self.latched:
            return
        self._draw(contacts)

    def latch(self, contacts) -> None:
        """Keep the terminal contact visible after the scene has been reset."""
        if self.markers is None:
            return
        self.latched = tuple(contacts)
        self._draw(self.latched)

    def clear(self) -> None:
        """Drop the latched contact when the operator starts the next attempt."""
        self.latched = ()
        if self.markers is not None:
            self.markers.set_visibility(False)

    def _draw(self, contacts) -> None:
        if not contacts:
            self.markers.set_visibility(False)
            return
        translations = torch.tensor(
            [contact.position_w for contact in contacts],
            device=self.env.device, dtype=torch.float32)
        self.markers.set_visibility(True)
        self.markers.visualize(
            translations=translations,
            marker_indices=[MARKER_KINDS.index(contact.kind) for contact in contacts],
        )

