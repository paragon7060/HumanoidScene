"""Measure opposing contact patches on the convex meshes used by PhysX."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FingerPad:
    body: str
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    area: float
    meshes: tuple[str, ...]


def convex_contact_patch(points, inward):
    """Area centroid of hull facets most parallel to the inward jaw direction.

    This is a geometric reference patch, not a replacement for measured contacts.
    Using the convex hull matches the hand collider approximation in contact_physics.
    """
    import numpy as np
    from scipy.spatial import ConvexHull
    points = np.asarray(points, dtype=float)
    inward = np.asarray(inward, dtype=float)
    inward /= np.linalg.norm(inward)
    hull = ConvexHull(points)
    triangles = points[hull.simplices]
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=-1) / 2
    alignment = hull.equations[:, :3] @ inward
    if alignment.max() < .95:
        raise ValueError("No inward-facing contact patch; inspect this gripper geometry")
    selected = alignment >= alignment.max() - .001
    area = areas[selected].sum()
    center = np.average(triangles[selected].mean(1), axis=0, weights=areas[selected])
    normal = np.average(hull.equations[selected, :3], axis=0, weights=areas[selected])
    normal /= np.linalg.norm(normal)
    return center, normal, float(area)


def finger_pads(stage, robot_root, names):
    """Read enabled collision meshes in each current, scale-free rigid link frame."""
    import numpy as np
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    cache = UsdGeom.XformCache()
    bodies = [stage.GetPrimAtPath(f"{robot_root}/{name}") for name in names]
    if len(bodies) != 4 or not all(b and b.HasAPI(UsdPhysics.RigidBodyAPI) for b in bodies):
        raise ValueError("Expected four physical finger bodies in left/right jaw order")
    transforms = [cache.GetLocalToWorldTransform(b) for b in bodies]
    pads = []
    for index, body in enumerate(bodies):
        rigid = Gf.Matrix4d().SetRotate(Gf.Transform(transforms[index]).GetRotation())
        rigid.SetTranslateOnly(transforms[index].ExtractTranslation())
        other = transforms[index ^ 1].ExtractTranslation()
        inward = np.array(rigid.GetInverse().Transform(other))
        patches, paths = [], []
        for prim in Usd.PrimRange(body, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
                continue
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() != "convexHull":
                raise ValueError(f"Unsupported finger collision approximation: {prim.GetPath()}")
            transform = cache.GetLocalToWorldTransform(prim) * rigid.GetInverse()
            points = [tuple(transform.Transform(v)) for v in UsdGeom.Mesh(prim).GetPointsAttr().Get()]
            patches.append(convex_contact_patch(points, inward))
            paths.append(str(prim.GetPath().MakeRelativePath(body.GetPath())))
        # Multiple separated pads need an explicit choice; never bridge empty space.
        if len(patches) != 1:
            raise ValueError(f"Expected one jaw collision mesh on {body.GetPath()}, got {len(patches)}")
        center, normal, area = patches[0]
        pads.append(FingerPad(names[index], tuple(map(float, center)), tuple(map(float, normal)), area, tuple(paths)))
    return pads
