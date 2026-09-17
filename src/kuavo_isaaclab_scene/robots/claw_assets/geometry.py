"""Offline tip clearance from the same URDF/STL and closed linkage used by Isaac.

The gap is between the inner edges of each mesh's distal 1 mm band, projected
on the claw base X axis. It is a geometric reference, not a contact simulation.
"""

import hashlib
import math
from pathlib import Path
import re
import struct
import xml.etree.ElementTree as ET

import numpy as np

from .package import load_claw_asset
from .linkage import DRIVER_OPEN_MIN, passive_joint_angles


def _stl_vertices(path):
    payload = Path(path).read_bytes()
    count = struct.unpack_from("<I", payload, 80)[0] if len(payload) >= 84 else 0
    if len(payload) == 84 + count * 50:
        facets = np.frombuffer(payload, dtype=np.dtype([
            ("normal", "<f4", (3,)), ("vertices", "<f4", (3, 3)), ("attribute", "<u2")
        ]), offset=84)
        return facets["vertices"].reshape(-1, 3).astype(float)
    vertices = re.findall(r"vertex\s+(\S+)\s+(\S+)\s+(\S+)", payload.decode("ascii"))
    if not vertices:
        raise ValueError(f"No STL vertices: {path}")
    return np.asarray(vertices, dtype=float)


def _rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * math.cos(angle) + (1 - math.cos(angle)) * np.outer(axis, axis) + math.sin(angle) * cross


class TwoFingerGeometry:
    def __init__(self, side="left", tip_band_m=0.001):
        if not math.isfinite(tip_band_m) or tip_band_m <= 0:
            raise ValueError("Tip band must be positive")
        self.asset = load_claw_asset(side)
        root = ET.parse(self.asset.urdf_path).getroot()
        self.joints = {j.find("child").get("link"): j for j in root.findall("joint")}
        self.tip_points = {}
        self.tip_band_m = tip_band_m
        digest = hashlib.sha256(self.asset.urdf_path.read_bytes())
        for jaw in "fb":
            link = f"{self.asset.prefix}_{jaw}_finger"
            visual = root.find(f"./link[@name='{link}']/visual")
            mesh = visual.find("geometry/mesh")
            path = (self.asset.urdf_path.parent / mesh.get("filename")).resolve()
            digest.update(path.read_bytes())
            vertices = _stl_vertices(path)
            vertices *= np.asarray(mesh.get("scale", "1 1 1").split(), dtype=float)
            tip = vertices[vertices[:, 2] <= vertices[:, 2].min() + tip_band_m]
            origin = visual.find("origin")
            transform = self._origin(origin)
            self.tip_points[jaw] = tip @ transform[:3, :3].T + transform[:3, 3]
        digest.update(str(tip_band_m).encode())
        self.sha256 = digest.hexdigest()

    @staticmethod
    def _origin(origin):
        transform = np.eye(4)
        if origin is None:
            return transform
        transform[:3, 3] = np.asarray(origin.get("xyz", "0 0 0").split(), dtype=float)
        roll, pitch, yaw = map(float, origin.get("rpy", "0 0 0").split())
        transform[:3, :3] = _rotation((0, 0, 1), yaw) @ _rotation((0, 1, 0), pitch) @ _rotation((1, 0, 0), roll)
        return transform

    def finger_transform(self, driver, jaw):
        sign = 1 if jaw == "f" else -1
        q = sign * driver
        q3, _ = passive_joint_angles(q, jaw)
        angles = {f"{self.asset.prefix}_{jaw}_bar_1_joint": q,
                  f"{self.asset.prefix}_{jaw}_bar_3_joint": q3}
        chain = []
        link = f"{self.asset.prefix}_{jaw}_finger"
        while link != self.asset.root_link:
            joint = self.joints[link]
            chain.append(joint)
            link = joint.find("parent").get("link")
        transform = np.eye(4)
        for joint in reversed(chain):
            transform = transform @ self._origin(joint.find("origin"))
            if joint.get("type") != "fixed":
                rotation = np.eye(4)
                rotation[:3, :3] = _rotation(
                    list(map(float, joint.find("axis").get("xyz").split())),
                    angles.get(joint.get("name"), 0.0))
                transform = transform @ rotation
        return transform

    def gap_m(self, driver):
        points = {}
        for jaw in "fb":
            transform = self.finger_transform(driver, jaw)
            points[jaw] = self.tip_points[jaw] @ transform[:3, :3].T + transform[:3, 3]
        return float(points["f"][:, 0].min() - points["b"][:, 0].max())

    def driver_for_gap_m(self, width):
        if not math.isfinite(width) or width < 0:
            raise ValueError("Gap must be a finite non-negative distance")
        minimum, maximum = self.gap_m(0), self.gap_m(DRIVER_OPEN_MIN)
        if width > maximum + 1e-9:
            raise ValueError(f"Requested {width} m exceeds validated maximum {maximum} m")
        if width <= minimum:
            return 0.0  # Rounded CAD retains about 25 micrometres at q=0.
        lower, upper = DRIVER_OPEN_MIN, 0.0
        for _ in range(48):
            middle = (lower + upper) / 2
            if self.gap_m(middle) > width:
                lower = middle
            else:
                upper = middle
        return (lower + upper) / 2
