"""Whole-URDF self-collision geometry and jointly filtered position commands.

All modeled non-allowed pairs are checked, not only wrists or the nearest arm.
Meshes use conservative convex hulls; missing collision geometry uses visuals.
This guards simulated commands, not physical safety or environment collisions.
"""
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import minimize
import trimesh

from .urdf_arm_ik import axis_rotation, skew


def load_fcl():
    try:
        return importlib.import_module("fcl")
    except ModuleNotFoundError:
        local = Path(__file__).resolve().parents[3] / ".external/self-collision"
        if local.is_dir():
            sys.path.append(str(local))
        try:
            return importlib.import_module("fcl")
        except ModuleNotFoundError as error:
            raise RuntimeError("Self-collision requires python-fcl. Run ./setup_self_collision.sh; "
                               "collision checking will NOT be silently disabled.") from error


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        r, p, y = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        t[:3, :3] = axis_rotation([0, 0, 1], y) @ axis_rotation([0, 1, 0], p) @ axis_rotation([1, 0, 0], r)
    return t


@dataclass
class Edge:
    name: str
    parent: str
    child: str
    transform: np.ndarray
    kind: str
    axis: np.ndarray
    index: int | None
    multiplier: float = 1.
    offset: float = 0.


@dataclass
class Shape:
    link: str
    transform: np.ndarray
    geometry: object
    obj: object
    radius: float
    source: str


class CollisionStop(RuntimeError):
    pass


class RobotCollisionModel:
    def __init__(self, urdf, allowed_pairs=None):
        self.fcl = load_fcl()
        self.path = Path(urdf).resolve(strict=True)
        tree = ET.parse(self.path).getroot()
        raw = tree.findall("joint")
        links = [l.get("name") for l in tree.findall("link")]
        roots = set(links) - {j.find("child").get("link") for j in raw}
        if len(roots) != 1:
            raise ValueError("Self-collision needs one URDF tree root")
        self.root = roots.pop()
        self.names = [j.get("name") for j in raw if j.get("type") != "fixed" and j.find("mimic") is None]
        self.lower, self.upper, self.velocity = [], [], []
        for name in self.names:
            j = next(j for j in raw if j.get("name") == name)
            limit = j.find("limit")
            if limit is None:
                if j.get("type") != "continuous":
                    raise ValueError(f"Missing joint limits: {name}")
                limit = ET.Element("limit", {"velocity": "30"})
            self.lower.append(float(limit.get("lower", "-1e9")))
            self.upper.append(float(limit.get("upper", "1e9")))
            self.velocity.append(float(limit.get("velocity", "1")))
        self.lower, self.upper, self.velocity = map(np.asarray, (self.lower, self.upper, self.velocity))
        self.edges, self.paths = [], {self.root: []}
        while raw:
            progress = False
            for j in raw[:]:
                parent, child = j.find("parent").get("link"), j.find("child").get("link")
                if parent not in self.paths:
                    continue
                kind = j.get("type")
                if kind not in ("fixed", "revolute", "continuous", "prismatic"):
                    raise ValueError(f"Unsupported collision joint {kind}")
                axis = np.fromstring(j.find("axis").get("xyz", "1 0 0") if j.find("axis") is not None else "1 0 0", sep=" ")
                axis /= np.linalg.norm(axis)
                mimic = j.find("mimic")
                name = j.get("name")
                index = None if kind == "fixed" else self.names.index(name if mimic is None else mimic.get("joint"))
                edge = Edge(name, parent, child, origin(j.find("origin")), kind, axis, index,
                            float(mimic.get("multiplier", "1")) if mimic is not None else 1.,
                            float(mimic.get("offset", "0")) if mimic is not None else 0.)
                self.paths[child] = self.paths[parent] + [edge]
                self.edges.append(edge); raw.remove(j); progress = True
            if not progress:
                raise ValueError("Unresolved URDF tree/mimic dependency")
        self.shapes, self.empty_links = [], []
        for link in tree.findall("link"):
            geometries = link.findall("collision") or link.findall("visual")
            source = "collision" if link.findall("collision") else "visual fallback"
            if not geometries:
                self.empty_links.append(link.get("name"))
            for item in geometries:
                geom = item.find("geometry")
                shape, radius = self._geometry(list(geom)[0])
                self.shapes.append(Shape(link.get("name"), origin(item.find("origin")), shape,
                                         self.fcl.CollisionObject(shape), radius, source))
        allowed = {}
        if allowed_pairs:
            data = json.loads(Path(allowed_pairs).read_text())
            for entry in data["allowed_pairs"]:
                a, b = entry["links"]
                if a not in links or b not in links or not entry.get("reason"):
                    raise ValueError(f"Invalid explicit collision exclusion: {entry}")
                allowed[frozenset((a, b))] = entry["reason"]
        self.pairs, self.exclusions, motion_bounds = [], [], []
        for i, a in enumerate(self.shapes):
            for k in range(i + 1, len(self.shapes)):
                b = self.shapes[k]
                pa, pb = self.paths[a.link], self.paths[b.link]
                common = 0
                while common < min(len(pa), len(pb)) and pa[common].name == pb[common].name:
                    common += 1
                moving = [e for e in pa[common:] + pb[common:] if e.index is not None]
                reason = allowed.get(frozenset((a.link, b.link)))
                if not moving:
                    reason = "same rigid assembly"
                elif len(moving) == 1:
                    reason = "adjacent articulated links / joint interface"
                if reason:
                    self.exclusions.append((a.link, b.link, reason)); continue
                self.pairs.append((i, k))
                bound = np.zeros(len(self.names))
                for shape, path in ((a, pa[common:]), (b, pb[common:])):
                    for n, edge in enumerate(path):
                        if edge.index is not None:
                            radius = shape.radius + np.linalg.norm(shape.transform[:3, 3])
                            radius += sum(np.linalg.norm(e.transform[:3, 3]) for e in path[n+1:])
                            # Prismatic travel after an ancestor enlarges its rotational radius.
                            radius += sum(max(abs(self.lower[e.index]), abs(self.upper[e.index])) * abs(e.multiplier)
                                          for e in path[n+1:] if e.kind == "prismatic")
                            bound[edge.index] += abs(edge.multiplier) * (1. if edge.kind == "prismatic" else radius)
                motion_bounds.append(bound)
        self.motion_bounds = np.asarray(motion_bounds)
        self.pair_indices = np.asarray(self.pairs, dtype=int)
        self.radii = np.array([s.radius for s in self.shapes])
        self._request = self.fcl.DistanceRequest(enable_nearest_points=True)

    def _geometry(self, g):
        fcl = self.fcl
        if g.tag == "sphere":
            r = float(g.get("radius")); return fcl.Sphere(r), r
        if g.tag in ("cylinder", "capsule"):
            r, length = float(g.get("radius")), float(g.get("length"))
            cls = fcl.Cylinder if g.tag == "cylinder" else fcl.Capsule
            radius = np.hypot(r, length / 2) if g.tag == "cylinder" else r + length / 2
            return cls(r, length), radius
        if g.tag == "box":
            size = np.fromstring(g.get("size"), sep=" ")
            return fcl.Box(*size), np.linalg.norm(size) / 2
        if g.tag != "mesh":
            raise ValueError(f"Unsupported collision geometry {g.tag}")
        filename = g.get("filename")
        if "://" in filename:
            raise ValueError(f"Resolve mesh URI to a local URDF-relative path first: {filename}")
        path = (self.path.parent / filename).resolve(strict=True)
        mesh = trimesh.load(path, force="mesh", process=False)
        mesh.vertices *= np.fromstring(g.get("scale", "1 1 1"), sep=" ")
        hull = mesh.convex_hull
        vertices = np.asarray(hull.vertices, dtype=np.float64)
        faces = np.column_stack((np.full(len(hull.faces), 3), hull.faces)).astype(np.int32).ravel()
        return fcl.Convex(vertices, len(hull.faces), faces), np.linalg.norm(vertices, axis=1).max()

    def forward(self, q):
        q = np.asarray(q)
        if q.shape != (len(self.names),) or not np.isfinite(q).all():
            raise CollisionStop("Non-finite or incomplete self-collision joint state")
        frames, joint_frames = {self.root: np.eye(4)}, {}
        twists = {self.root: (np.zeros((3, len(self.names))), np.zeros((3, len(self.names))))}
        for edge in self.edges:
            t = frames[edge.parent] @ edge.transform
            joint_frames[edge.name] = t.copy()
            motion = np.eye(4)
            if edge.index is not None:
                value = q[edge.index] * edge.multiplier + edge.offset
                if edge.kind == "prismatic":
                    motion[:3, 3] = edge.axis * value
                else:
                    motion[:3, :3] = axis_rotation(edge.axis, value)
            frames[edge.child] = t @ motion
            linear, angular = twists[edge.parent]
            if edge.index is not None:
                linear, angular = linear.copy(), angular.copy()
                axis = t[:3, :3] @ edge.axis * edge.multiplier
                if edge.kind == "prismatic":
                    linear[:, edge.index] += axis
                else:
                    linear[:, edge.index] += skew(t[:3, 3]) @ axis
                    angular[:, edge.index] += axis
            twists[edge.child] = (linear, angular)
        centers = []
        for shape in self.shapes:
            t = frames[shape.link] @ shape.transform
            shape.obj.setTransform(self.fcl.Transform(t[:3, :3], t[:3, 3]))
            centers.append(t[:3, 3])
        self.frames, self.joint_frames, self.twists = frames, joint_frames, twists
        return np.asarray(centers)

    def point_jacobian(self, link, point):
        linear, angular = self.twists[link]
        return linear - skew(point) @ angular

    def distances(self, q, threshold=.05, gradients=False):
        centers = self.forward(q)
        a, b = self.pair_indices.T
        # Bounding spheres give conservative lower distances for broadphase.
        distances = np.linalg.norm(centers[b] - centers[a], axis=1) - self.radii[a] - self.radii[b]
        rows = []
        for index in np.flatnonzero(distances < threshold):
            sa, sb = self.shapes[a[index]], self.shapes[b[index]]
            result = self.fcl.DistanceResult()
            d = float(self.fcl.distance(sa.obj, sb.obj, self._request, result))
            if not np.isfinite(d):
                raise CollisionStop(f"Invalid collision query: {sa.link}/{sb.link}")
            distances[index] = d
            if gradients and 0 < d < threshold:
                p1, p2 = map(np.asarray, result.nearest_points)
                normal = (p2 - p1) / max(np.linalg.norm(p2 - p1), 1e-12)
                row = normal @ (self.point_jacobian(sb.link, p2) - self.point_jacobian(sa.link, p1))
                rows.append((index, row))
        return distances, rows

    def pair_name(self, index):
        a, b = self.pairs[index]
        return f"{self.shapes[a].link} / {self.shapes[b].link}"


class SelfCollisionFilter:
    def __init__(self, model, clearance=.003, influence=.04):
        if not 0 < clearance < influence:
            raise ValueError("Expected 0 < self-collision clearance < influence")
        self.model, self.clearance, self.influence = model, clearance, influence
        self.status = {}

    def filter_light(self, q, desired, dt):
        """Bounded-cost endpoint guard; no gradients, optimizer or swept path.

        Called only once per control tick. Broadphase considers all modeled
        pairs; FCL only checks nearby candidates. This is NOT continuous
        collision avoidance and does not monitor intervening physics steps.
        """
        q, desired = np.asarray(q), np.asarray(desired)
        if dt <= 0 or not np.isfinite(q).all() or not np.isfinite(desired).all():
            raise CollisionStop("Invalid self-collision command")
        d0, _ = self.model.distances(q, self.clearance)
        nearest = int(np.argmin(d0))
        if d0[nearest] < self.clearance:
            raise CollisionStop(f"Current pose violates clearance: {self.model.pair_name(nearest)} "
                                f"distance={d0[nearest]:.5f}m; reset/reposition while stopped")
        step = np.minimum(self.model.velocity, 2.5) * dt
        lower = np.maximum(-step, self.model.lower - q)
        upper = np.minimum(step, self.model.upper - q)
        if np.any(lower > upper):
            raise CollisionStop("Joint state outside collision-model limits")
        candidate = q + np.clip(desired-q, lower, upper)
        if np.array_equal(candidate, q):
            blocked = False
        else:
            d1, _ = self.model.distances(candidate, self.clearance)
            blocked = bool(d1.min() < self.clearance)
        if blocked:
            candidate = q.copy()
        self.status = {"minimum_distance_m": float(d0[nearest]), "pair": self.model.pair_name(nearest),
                       "scale": 0. if blocked else 1.,
                       "modified": bool(np.max(np.abs(candidate-desired)) > 1e-6)}
        return candidate

    def check_state(self, q, dt, measured_velocity):
        """Every-physics-step monitor, including actual velocity/coasting."""
        d0, _ = self.model.distances(q, self.influence)
        nearest = int(np.argmin(d0))
        if d0[nearest] < self.clearance:
            raise CollisionStop(f"Current pose violates clearance: {self.model.pair_name(nearest)} "
                                f"distance={d0[nearest]:.5f}m; reset/reposition while stopped")
        if np.any(np.asarray(measured_velocity) != 0):
            coast = q + np.asarray(measured_velocity) * dt
            dc, _ = self.model.distances(coast, self.influence)
            if not self._swept_safe(q, coast, d0, dc):
                raise CollisionStop("Measured joint velocity predicts self-collision in next physics step")
        return d0

    def cached_path_safe(self, q, target, d0):
        # Actual dynamics may deviate from the previously certified path.
        # Re-certify from the measured pose every physics step, not just once
        # when a new teleop command arrives.
        dt, _ = self.model.distances(target, self.influence)
        return self._swept_safe(q, target, d0, dt)

    def _swept_safe(self, q0, q1, d0, d1, depth=0):
        if min(d0.min(), d1.min()) < self.clearance:
            return False
        # A Lipschitz bound on the entire joint-interpolated motion certifies
        # separation BETWEEN endpoints; this is not endpoint-only sampling.
        movement = self.model.motion_bounds @ np.abs(q1 - q0)
        if np.all(np.minimum(d0, d1) - movement / 2 >= self.clearance):
            return True
        if depth >= 7:
            return False  # Uncertified path is rejected, never assumed safe.
        mid = (q0 + q1) / 2
        dm, _ = self.model.distances(mid, self.influence)
        return self._swept_safe(q0, mid, d0, dm, depth+1) and self._swept_safe(mid, q1, dm, d1, depth+1)

    def filter(self, q, desired, dt, measured_velocity=None):
        q, desired = np.asarray(q), np.asarray(desired)
        if dt <= 0 or not np.isfinite(desired).all():
            raise CollisionStop("Invalid self-collision command")
        d0, rows = self.model.distances(q, self.influence, gradients=True)
        nearest = int(np.argmin(d0))
        if d0[nearest] < self.clearance:
            raise CollisionStop(f"Current pose violates clearance: {self.model.pair_name(nearest)} "
                                f"distance={d0[nearest]:.5f}m; reset/reposition while stopped")
        if measured_velocity is not None and np.any(np.asarray(measured_velocity) != 0):
            coast = q + np.asarray(measured_velocity) * dt
            dc, _ = self.model.distances(coast, self.influence)
            if not self._swept_safe(q, coast, d0, dc):
                raise CollisionStop("Measured joint velocity predicts self-collision in next physics step")
        step = np.minimum(self.model.velocity, 2.5) * dt
        lower = np.maximum(-step, self.model.lower - q)
        upper = np.minimum(step, self.model.upper - q)
        if np.any(lower > upper):
            raise CollisionStop("Joint state outside collision-model limits")
        delta = np.clip(desired - q, lower, upper)
        if rows:
            ids = [i for i, _ in rows]
            a = np.stack([row for _, row in rows])
            rhs = -4. * dt * (d0[ids] - self.clearance)
            if np.any(a @ delta < rhs):
                nominal = delta.copy()
                result = minimize(lambda x: .5 * np.sum((x-nominal)**2), np.zeros_like(delta),
                                  jac=lambda x: x-nominal, method="SLSQP", bounds=list(zip(lower, upper)),
                                  constraints={"type": "ineq", "fun": lambda x: a@x-rhs, "jac": lambda x: a},
                                  options={"maxiter": 40, "ftol": 1e-10})
                delta = result.x if result.success and np.all(a@result.x >= rhs-1e-8) else np.zeros_like(delta)
        scale = 1.
        for _ in range(9):
            candidate = q + scale * delta
            d1, _ = self.model.distances(candidate, self.influence)
            if self._swept_safe(q, candidate, d0, d1):
                break
            scale *= .5
        else:
            scale = 0.; candidate = q.copy()
        self.status = {"minimum_distance_m": float(d0[nearest]), "pair": self.model.pair_name(nearest),
                       "scale": scale, "modified": bool(np.max(np.abs(candidate-desired)) > 1e-6)}
        return candidate
