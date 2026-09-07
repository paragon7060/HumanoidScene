"""URDF arm kinematics and bounded local IK. NumPy only; no simulator imports.

This is NOT a collision-free motion planner. Bounds enforce joint travel and
rate limits; a shoulder sphere only rejects grossly out-of-reach targets.
"""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def axis_rotation(axis, angle):
    k = skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1. - np.cos(angle)) * (k @ k)


def quat_matrix(q):
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    w, v = q[0], q[1:]
    return np.eye(3) + 2 * w * skew(v) + 2 * skew(v) @ skew(v)


def rotation_error(target, current):
    r = target @ current.T
    angle = np.arccos(np.clip((np.trace(r) - 1) / 2, -1., 1.))
    vee = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    if angle < 1e-6:
        return .5 * vee
    if np.pi - angle < 1e-5:
        values, vectors = np.linalg.eigh((r + r.T) / 2)
        axis = vectors[:, np.argmax(values)]
        if axis @ vee < 0:
            axis = -axis
        return axis * angle
    return vee * angle / (2 * np.sin(angle))


def box_qp(h, b, lower, upper):
    """Minimize .5*x'Hx-b'x within box bounds using a small active set.

    Unlike clipping an unconstrained IK result, remaining joints are re-solved
    whenever a joint saturates. H must be positive definite.
    """
    if np.any(lower > upper) or not all(np.isfinite(a).all() for a in (h, b, lower, upper)):
        raise ValueError("Invalid bounded IK problem")
    x = np.clip(np.zeros_like(b), lower, upper)
    fixed = np.where(x <= lower + 1e-10, -1, np.where(x >= upper - 1e-10, 1, 0))
    locked = upper - lower < 1e-10
    for _ in range(128):
        free = fixed == 0
        direction = np.zeros_like(x)
        if np.any(free):
            direction[free] = np.linalg.solve(h[np.ix_(free, free)], (b - h @ x)[free])
        alpha, hit = 1., -1
        for i in np.flatnonzero(free):
            if abs(direction[i]) < 1e-12:
                continue
            bound = upper[i] if direction[i] > 0 else lower[i]
            step = (bound - x[i]) / direction[i]
            if step < alpha:
                alpha, hit = max(0., step), i
        x = np.clip(x + alpha * direction, lower, upper)
        if hit >= 0:
            fixed[hit] = 1 if direction[hit] > 0 else -1
            continue
        gradient = h @ x - b
        violation = np.where(fixed == -1, -gradient, np.where(fixed == 1, gradient, 0.))
        violation[locked] = 0.
        if violation.max(initial=0.) < 1e-8:
            return x
        fixed[int(np.argmax(violation))] = 0
    raise RuntimeError("Bounded IK active set did not converge")


@dataclass
class Joint:
    name: str
    child: str
    xyz: np.ndarray
    rotation: np.ndarray
    axis: np.ndarray | None


class UrdfArm:
    def __init__(self, path, side):
        if side not in ("left", "right"):
            raise ValueError(side)
        self.path = str(Path(path).resolve(strict=True))
        self.side = side
        letter = side[0]
        self.names = [f"zarm_{letter}{i}_joint" for i in range(1, 8)]
        self.tip = f"zarm_{letter}7_end_effector"
        root = ET.parse(self.path).getroot()
        by_name = {j.attrib["name"]: j for j in root.findall("joint")}
        by_child = {j.find("child").attrib["link"]: j for j in root.findall("joint")}
        self.parent = by_name[self.names[0]].find("parent").attrib["link"]
        chain, link = [], self.tip
        while link != self.parent:
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").attrib["link"]
        self.joints, lower, upper, speed = [], [], [], []
        movable = []
        for j in reversed(chain):
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            rotation = (axis_rotation([0, 0, 1], rpy[2]) @ axis_rotation([0, 1, 0], rpy[1])
                        @ axis_rotation([1, 0, 0], rpy[0]))
            axis = None
            if j.get("type") != "fixed":
                if j.get("type") != "revolute" or j.find("mimic") is not None:
                    raise ValueError(f"Unsupported arm joint: {j.attrib}")
                movable.append(j.get("name"))
                axis = np.fromstring(j.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                limit = j.find("limit")
                lower.append(float(limit.get("lower")))
                upper.append(float(limit.get("upper")))
                speed.append(float(limit.get("velocity")))
            self.joints.append(Joint(j.get("name"), j.find("child").get("link"), xyz, rotation, axis))
        if movable != self.names:
            raise ValueError(f"Unexpected arm chain: {movable}")
        self.lower, self.upper, self.speed = map(np.asarray, (lower, upper, speed))
        self.shoulder = self.joints[0].xyz.copy()
        self.reach = sum(np.linalg.norm(j.xyz) for j in self.joints[1:])

    def fk(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Expected seven finite arm joint angles")
        p, r, axes, origins, points = np.zeros(3), np.eye(3), [], [], {}
        for joint in self.joints:
            p = p + r @ joint.xyz
            r = r @ joint.rotation
            if joint.axis is not None:
                origins.append(p.copy())
                axes.append(r @ joint.axis)
                r = r @ axis_rotation(joint.axis, q[len(axes) - 1])
            points[joint.child] = p.copy()
        jac = np.zeros((6, 7))
        for i, (axis, origin) in enumerate(zip(axes, origins)):
            jac[:3, i] = np.cross(axis, p - origin)
            jac[3:, i] = axis
        return p, r, jac, points

    def project_target(self, target):
        target = np.asarray(target, dtype=float)
        delta = target - self.shoulder
        # Conservative gross reach bound, not an exact orientation-aware workspace.
        radius = .95 * self.reach
        return self.shoulder + delta * min(1., radius / max(np.linalg.norm(delta), 1e-9))

    def validate_live(self, q, position, rotation, jacobian, limits):
        p, r, jac, _ = self.fk(q)
        if not all(np.isfinite(a).all() for a in (position, rotation, jacobian, limits)):
            raise ValueError("Non-finite live USD model state")
        errors = (np.linalg.norm(p - position), np.linalg.norm(rotation_error(r, rotation)),
                  np.max(np.abs(jac - jacobian)))
        if errors[0] > .01 or errors[1] > .03 or errors[2] > .05:
            raise ValueError(f"URDF/USD mismatch ({self.side}): position={errors[0]:.4f}m, "
                             f"rotation={errors[1]:.4f}rad, Jacobian max={errors[2]:.4f}; "
                             "check robot/tool frames; URDF IK was NOT enabled")
        if np.max(np.abs(limits - np.column_stack((self.lower, self.upper)))) > .01:
            raise ValueError(f"URDF/USD joint limits differ ({self.side}); check model conversion")
        return errors

    def ready_pose(self):
        # Mirrored elbow-bent seed. Optimize position using this model's fixed
        # wrist/tool transform (S63 differs from S56/S200062).
        sign = 1 if self.side == "left" else -1
        q = np.array([-.35, .20 * sign, 0., -1.35, 0., 0., .35])
        q = np.clip(q, self.lower + .08, self.upper - .08)
        rest = q.copy()
        target = self.shoulder + np.array([.32, .04 * sign, -.32])
        for _ in range(100):
            p, _, jac, _ = self.fk(q)
            j = jac[:3]
            h = j.T @ j + .002 * np.eye(7)
            b = j.T @ (target - p) + .0005 * (rest - q)
            dq = box_qp(h, b, np.maximum(-.08, self.lower + .08 - q),
                        np.minimum(.08, self.upper - .08 - q))
            q += dq
            if np.linalg.norm(target - p) < .003:
                break
        if np.linalg.norm(self.fk(q)[0] - target) > .025:
            raise ValueError(f"Cannot generate a ready pose for {self.path} {self.side}")
        return q

    def step(self, q, target, target_rotation, rest, previous_velocity, dt, response,
             orientation_weight=.5, lower=None, upper=None):
        q = np.asarray(q)
        low = self.lower if lower is None else np.maximum(lower, self.lower)
        high = self.upper if upper is None else np.minimum(upper, self.upper)
        if np.any(low >= high) or np.any(q < low - .02) or np.any(q > high + .02):
            raise ValueError("Live arm angles/limits do not match the URDF")
        p, r, jac, _ = self.fk(q)
        effective = self.project_target(target)
        ep = effective - p
        # Position takes priority when far away. Restore wrist tracking as
        # position converges rather than twisting to satisfy both at once.
        weight = orientation_weight * min(1., .06 / max(np.linalg.norm(ep), .06))
        j = jac.copy()
        j[:3] *= 3.
        j[3:] *= weight
        error = np.r_[3. * ep, rotation_error(target_rotation, r) * weight]
        singular = np.linalg.svd(j, compute_uv=False)[-1]
        damping = max(response.damping, .04 + .12 * max(0., 1. - singular / .08))
        h = j.T @ j + (damping ** 2 + .008) * np.eye(7)
        # Continuous posture preference penalizes winding away from the
        # validated initial/reference pose. Never restart IK from zeros.
        b = j.T @ (response.pose_gain * error) + .008 * 1.5 * (rest - q)
        vmax = np.minimum(self.speed, response.max_velocity)
        margin = np.minimum(.08, (high - low) * .1)
        # Velocity damper slows BEFORE a joint stop. Outside the margin it
        # allows only inward motion, without requiring an instantaneous jump.
        vlo = np.maximum(-vmax, np.minimum(0., 4. * (low + margin - q)))
        vhi = np.minimum(vmax, np.maximum(0., 4. * (high - margin - q)))
        hard_lo = np.maximum(vlo, (low - q) / dt)
        hard_hi = np.minimum(vhi, (high - q) / dt)
        acceleration = response.max_acceleration * dt
        lo = np.maximum(hard_lo, previous_velocity - acceleration)
        hi = np.minimum(hard_hi, previous_velocity + acceleration)
        # Position/rate safety outranks acceleration when recovering near a
        # stop or changing response profiles. Brake rather than cross a limit.
        conflict = lo > hi
        stop = np.clip(np.zeros(7), hard_lo, hard_hi)
        lo[conflict] = hi[conflict] = stop[conflict]
        velocity = box_qp(h, b, lo, hi)
        return velocity, effective, {
            "projection_m": float(np.linalg.norm(target - effective)),
            "position_error_m": float(np.linalg.norm(ep)),
            "limit_margin_rad": float(np.minimum(q - low, high - q).min()),
            "orientation_weight": float(weight),
            "requested_parent_m": np.asarray(target).tolist(),
            "actual_parent_m": p.tolist(),
        }
