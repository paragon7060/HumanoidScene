"""CPU geometry for collision inspection; no simulator imports."""

import numpy as np


def polygon_geometry(points):
    points = np.asarray(points, dtype=float)
    triangles = np.cross(points[1:-1] - points[0], points[2:] - points[0])
    weights = np.linalg.norm(triangles, axis=-1) / 2
    area = weights.sum()
    if area < 1e-12:
        return points.mean(0), np.zeros(3), 0.
    center = (((points[0] + points[1:-1] + points[2:]) / 3) * weights[:, None]).sum(0) / area
    normal = triangles.sum(0)
    normal /= max(np.linalg.norm(normal), 1e-12)
    return center, normal, float(area)


def inward_face(polygons, direction):
    """Largest strongly facing convex polygon; semantic pad identity is NOT inferred."""
    direction = np.asarray(direction, dtype=float)
    direction /= max(np.linalg.norm(direction), 1e-12)
    best, score = None, 0.
    for vertices, outward in polygons:
        center, _, area = polygon_geometry(vertices)
        facing = float(np.dot(outward, direction))
        value = area * max(facing, 0.)**4 if facing > .5 else 0.
        if value > score:
            score, best = value, (vertices, center, outward)
    return best


def hull_polygons(vertices, faces):
    """Orient convex face normals outward, including reflected USD transforms."""
    vertices = np.asarray(vertices, dtype=float)
    center = vertices.mean(0)
    result = []
    for face in faces:
        polygon = vertices[np.asarray(face, dtype=int)]
        p, normal, area = polygon_geometry(polygon)
        if area <= 1e-12:
            continue
        if np.dot(normal, p - center) < 0:
            normal = -normal
        result.append((polygon, normal))
    return result


def polygon_edges(polygons):
    return np.asarray([[p[i], p[(i+1) % len(p)]] for p, _ in polygons for i in range(len(p))])


def force_arrow(point, force, metres_per_newton=.01, max_length=.12):
    force = np.asarray(force, dtype=float)
    magnitude = np.linalg.norm(force)
    if not np.isfinite(force).all() or not np.isfinite(point).all() or magnitude < 1e-6:
        return np.empty((0, 2, 3))
    direction = force / magnitude
    length = min(magnitude * metres_per_newton, max_length)
    end = point + length * direction
    axis = np.eye(3)[np.argmin(np.abs(direction))]
    side = np.cross(direction, axis)
    side /= np.linalg.norm(side)
    head = min(.012, length * .3)
    return np.array([[point, end], [end, end-head*direction+head*.5*side],
                     [end, end-head*direction-head*.5*side]])
