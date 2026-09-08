"""Snapshot each enabled USD physics collider, not aggregate visual/rack bounds.

Import after AppLauncher. Boxes bound individual colliders conservatively;
triangle/convex shape fidelity is not claimed. Live poses require USD updates
(the validation runner deliberately uses use_fabric=False).
"""

from __future__ import annotations

from itertools import product
import math
import numpy as np

from .geometry import inverse_transform, matrix_pose, pose_matrix


def bounded_cuboid(local_low, local_high, local_to_world) -> tuple[list, list]:
    """Include all inherited scale exactly once; reject sheared colliders."""
    low, high = np.asarray(local_low, float), np.asarray(local_high, float)
    transform = np.asarray(local_to_world, float)
    if (low.shape != (3,) or high.shape != (3,) or transform.shape != (4, 4)
            or not np.isfinite(transform).all() or not np.isfinite(low).all()
            or not np.isfinite(high).all() or np.any(high <= low)):
        raise ValueError("collider requires finite nonempty bounds and a 4x4 transform")
    scale = np.linalg.norm(transform[:3, :3], axis=0)
    if np.any(scale <= 0):
        raise ValueError("zero collider scale")
    rigid = np.eye(4)
    rigid[:3, :3] = transform[:3, :3] / scale
    if np.linalg.det(rigid[:3, :3]) < 0:
        rigid[:3, 2] *= -1  # A centred symmetric box tolerates reflected axes.
    rigid[:3, 3] = (transform @ np.r_[.5 * (low + high), 1])[:3]
    return matrix_pose(rigid), ((high - low) * scale).tolist()


def snapshot_colliders(stage, robot_root: str) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics
    # Collision geometry is often invisible/guide-purpose, unlike its visuals.
    # USD's constructor takes ``useExtentsHint`` before ``ignoreVisibility``;
    # passing the latter by keyword alone is ambiguous on USD 24.x.
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                               ["default", "render", "proxy", "guide"],
                               True, True)
    xforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    records, disabled, unsupported = [], [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if not prim.IsActive() or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            disabled.append(path)
            continue
        if not prim.IsA(UsdGeom.Boundable):
            unsupported.append({"path": path, "type": prim.GetTypeName()})
            continue
        local = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
        if local.IsEmpty():
            unsupported.append({"path": path, "reason": "empty bounds"})
            continue
        owner = prim
        while owner and not owner.HasAPI(UsdPhysics.RigidBodyAPI):
            owner = owner.GetParent()
        owner_path = str(owner.GetPath()) if owner else None
        pose, dims = bounded_cuboid(local.GetMin(), local.GetMax(),
                                    np.asarray(xforms.GetLocalToWorldTransform(prim)).T)
        is_robot = path.startswith(robot_root + "/")
        records.append({"path": path, "owner": owner_path, "robot": is_robot,
                        "shape": prim.GetTypeName(), "pose_w": pose, "dims": dims,
                        "approximation": "per-collider oriented bounding box"})
    if unsupported:
        raise ValueError(f"unrepresented enabled colliders: {unsupported}")
    if not records or not any(r["robot"] for r in records):
        raise ValueError("empty collision scene or no robot colliders")
    return {"colliders": records, "disabled_colliders": disabled,
            "excluded_enabled_colliders": [], "continuous_collision_guarantee": False}


def world_config(snapshot: dict, robot_base_pose_w) -> dict:
    base_inverse = inverse_transform(pose_matrix(robot_base_pose_w))
    return {"cuboid": {f"obstacle_{i}": {
        "pose": matrix_pose(base_inverse @ pose_matrix(item["pose_w"])), "dims": item["dims"],
    } for i, item in enumerate(snapshot["colliders"]) if not item["robot"]}}


def cover_cuboid(pose, dims, max_cell_m: float) -> list[dict]:
    """Cover the entire box with circumscribed spheres, including its corners.

    This intentionally over-approximates; it can cause false collisions. No
    surface-only fit is presented as a volumetric coverage guarantee.
    """
    if not math.isfinite(max_cell_m) or max_cell_m <= 0:
        raise ValueError("max_cell_m must be positive")
    dims = np.asarray(dims, float)
    if dims.shape != (3,) or not np.isfinite(dims).all() or np.any(dims <= 0):
        raise ValueError("sphere cover requires three positive dimensions")
    counts = np.maximum(1, np.ceil(dims / max_cell_m)).astype(int)
    if np.prod(counts) > 10000:
        raise ValueError("collider sphere cover exceeds 10000 cells; review geometry")
    step = dims / counts
    radius = float(np.linalg.norm(step) / 2)
    transform = pose_matrix(pose)
    return [{"center": (transform @ np.r_[(np.array(index)+.5)*step-dims/2, 1])[:3].tolist(),
             "radius": radius} for index in product(*(range(n) for n in counts))]
