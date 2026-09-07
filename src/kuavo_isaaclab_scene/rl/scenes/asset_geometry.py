"""Measure composed USD geometry in the scale-free frame used by rigid body poses."""

from dataclasses import dataclass, field
from itertools import product
from pxr import Gf, Usd, UsdGeom, UsdPhysics


@dataclass(frozen=True)
class RigidGeometry:
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    body_path: str


@dataclass(frozen=True)
class BoxGeometry(RigidGeometry):
    flaps: dict[str, RigidGeometry] = field(default_factory=dict)


def rigid_geometry(stage, prim, spawn_scale):
    bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    local = bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
    transform = transform * Gf.Matrix4d().SetScale(Gf.Vec3d(*spawn_scale))
    rigid = Gf.Matrix4d().SetRotate(Gf.Transform(transform).GetRotation())
    rigid.SetTranslateOnly(transform.ExtractTranslation())
    # PhysX root/body poses have rotation/translation, not the nested USD scale.
    # Include all inherited physical-wrapper scales exactly once in geometry.
    scaled_local = transform * rigid.GetInverse()
    points = [scaled_local.Transform(Gf.Vec3d(*p)) for p in product(
        *[(local.GetMin()[i], local.GetMax()[i]) for i in range(3)])]
    low = tuple(min(p[i] for p in points) for i in range(3))
    high = tuple(max(p[i] for p in points) for i in range(3))
    half = tuple((high[i] - low[i]) / 2 for i in range(3))
    if min(half) <= 0:
        raise ValueError(f"Invalid rigid bounds: {prim.GetPath()}")
    return RigidGeometry(tuple((high[i] + low[i]) / 2 for i in range(3)), half,
        str(prim.GetPath().MakeRelativePath(stage.GetDefaultPrim().GetPath())))


def box_geometry(asset_cfg, flap_names=()):
    stage = Usd.Stage.Open(asset_cfg.spawn.usd_path)
    if stage is None:
        raise ValueError(f"Unable to open box USD: {asset_cfg.spawn.usd_path}")
    found = {}
    for name in ("Body", *flap_names):
        bodies = [p for p in stage.Traverse() if p.GetName() == name and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(bodies) != 1:
            raise ValueError(f"Box USD must contain exactly one rigid {name}, found {len(bodies)}.")
        found[name] = rigid_geometry(stage, bodies[0], asset_cfg.spawn.scale or (1., 1., 1.))
    body = found.pop("Body")
    return BoxGeometry(body.center, body.half_size, body.body_path, found)
