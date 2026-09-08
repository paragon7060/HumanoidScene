"""URDF kinematic contract inspection, without importing Isaac or a GPU backend.

This is not a dynamics/collision model and cannot establish physical safety.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import xml.etree.ElementTree as ET

import numpy as np

from .geometry import axis_rotation, origin_matrix, vector


def joint_indices(requested, available) -> tuple[int, ...]:
    """Map exact names; never infer a simulator order from URDF order."""
    if (not requested or len(set(requested)) != len(requested)
            or len(set(available)) != len(available)):
        raise ValueError("joint names must be nonempty (requested) and unique")
    missing = set(requested) - set(available)
    if missing:
        raise ValueError(f"missing joints: {sorted(missing)}")
    index = {name: i for i, name in enumerate(available)}
    return tuple(index[name] for name in requested)


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    parent: str
    child: str
    xyz: tuple[float, ...]
    rpy: tuple[float, ...]
    axis: tuple[float, ...]
    lower: float | None
    upper: float | None
    velocity: float | None
    mimic: str | None


class UrdfModel:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        root = ET.parse(path).getroot()
        if root.tag != "robot":
            raise ValueError("expected a URDF robot element")
        links = root.findall("link")
        self.links = tuple(link.attrib["name"] for link in links)
        if len(set(self.links)) != len(self.links) or not self.links:
            raise ValueError("URDF links must be nonempty and unique")
        self.collision_links = tuple(link.attrib["name"] for link in links
                                     if link.find("collision") is not None)
        self.joints: dict[str, Joint] = {}
        self.by_child: dict[str, Joint] = {}
        for node in root.findall("joint"):
            name, kind = node.attrib["name"], node.attrib["type"]
            if kind not in {"fixed", "revolute", "continuous", "prismatic"}:
                raise ValueError(f"unsupported URDF joint type {kind}: {name}")
            parent, child = node.find("parent").attrib["link"], node.find("child").attrib["link"]
            if name in self.joints or child in self.by_child:
                raise ValueError(f"duplicate joint name or child: {name}/{child}")
            if parent not in self.links or child not in self.links:
                raise ValueError(f"missing parent/child link: {name}")
            origin, axis, limit = node.find("origin"), node.find("axis"), node.find("limit")
            def triplet(element, key, default):
                raw = default if element is None else element.get(key, default)
                return tuple(vector([float(v) for v in raw.split()], 3, key))
            xyz = triplet(origin, "xyz", "0 0 0")
            rpy = triplet(origin, "rpy", "0 0 0")
            direction = triplet(axis, "xyz", "1 0 0")
            if kind != "fixed" and np.linalg.norm(direction) < 1e-12:
                raise ValueError(f"zero joint axis: {name}")
            lower = upper = velocity = None
            if kind != "fixed":
                # Some packaged continuous wheel joints omit limits. Preserve
                # missingness during inspection; never invent motion limits.
                if limit is not None and "velocity" in limit.attrib:
                    velocity = float(limit.attrib["velocity"])
                    if not math.isfinite(velocity) or velocity <= 0:
                        raise ValueError(f"invalid velocity limit: {name}")
                if kind != "continuous" and limit is not None and "lower" in limit.attrib and "upper" in limit.attrib:
                    lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                    if not all(map(math.isfinite, (lower, upper))) or lower > upper:
                        raise ValueError(f"invalid position limits: {name}")
            mimic = node.find("mimic")
            joint = Joint(name, kind, parent, child, xyz, rpy, direction, lower, upper,
                          velocity, None if mimic is None else mimic.attrib["joint"])
            self.joints[name] = self.by_child[child] = joint
        roots = set(self.links) - set(self.by_child)
        if len(roots) != 1:
            raise ValueError(f"URDF must have one root link, got {sorted(roots)}")
        self.root_link = roots.pop()
        for link in self.links:
            self.chain(link)  # Reject disconnected cycles before use.

    def chain(self, link: str) -> tuple[Joint, ...]:
        if link not in self.links:
            raise ValueError(f"unknown link: {link}")
        result, visited = [], set()
        while link != self.root_link:
            if link in visited or link not in self.by_child:
                raise ValueError(f"cycle or disconnected link: {link}")
            visited.add(link)
            joint = self.by_child[link]
            result.append(joint)
            link = joint.parent
        return tuple(reversed(result))

    def fk(self, link: str, positions: dict[str, float]) -> np.ndarray:
        """Root-to-link transform; every movable ancestor needs an explicit value.

        No default-zero torso, no implicit mimic/closed-linkage approximation.
        This tree FK does not check contacts, self-collision or drive tracking.
        """
        result = np.eye(4)
        for joint in self.chain(link):
            result = result @ origin_matrix(joint.xyz, joint.rpy)
            if joint.kind == "fixed":
                continue
            if joint.mimic is not None:
                raise ValueError(f"mimic FK requires a linkage adapter: {joint.name}")
            if joint.name not in positions:
                raise ValueError(f"missing joint position: {joint.name}")
            if joint.kind != "continuous" and joint.lower is None:
                raise ValueError(f"missing position limits: {joint.name}")
            value = float(positions[joint.name])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite joint position: {joint.name}")
            if joint.lower is not None and not joint.lower <= value <= joint.upper:
                raise ValueError(f"joint position outside URDF limits: {joint.name}")
            motion = np.eye(4)
            if joint.kind == "prismatic":
                axis = np.asarray(joint.axis)
                motion[:3, 3] = axis / np.linalg.norm(axis) * value
            else:
                motion[:3, :3] = axis_rotation(joint.axis, value)
            result = result @ motion
        return result
