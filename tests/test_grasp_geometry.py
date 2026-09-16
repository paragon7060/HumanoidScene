"""Collision-derived references must ignore decorative meshes and follow rigid poses."""

from itertools import product

import numpy as np
import pytest
import torch

from kuavo_isaaclab_scene.robots.grasp_geometry import convex_contact_patch, finger_pads
from kuavo_isaaclab_scene.rl.mdp.flap_grasp import closing_geometry


def test_contact_patch_is_on_inner_surface_not_link_origin_or_mesh_center():
    vertices = np.array(list(product((-.005, .005), (-.01, .01), (-.07, -.05))))
    center, normal, area = convex_contact_patch(vertices, (-1, 0, 0))
    np.testing.assert_allclose(center, (-.005, 0, -.06), atol=1e-8)
    np.testing.assert_allclose(normal, (-1, 0, 0), atol=1e-8)
    assert area == pytest.approx(.0004)


def test_live_collision_geometry_ignores_visuals_and_uses_scaled_body_frame():
    Usd = pytest.importorskip("pxr.Usd")
    UsdGeom = pytest.importorskip("pxr.UsdGeom")
    UsdPhysics = pytest.importorskip("pxr.UsdPhysics")
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Robot")
    root.AddTranslateOp().Set((10, 20, 30))
    root.AddScaleOp().Set((2, 2, 2))
    names = ("lf", "lb", "rf", "rb")
    for index, name in enumerate(names):
        body = UsdGeom.Xform.Define(stage, "/Robot/" + name)
        body.AddTranslateOp().Set((.03 if index % 2 == 0 else -.03, index // 2, 0))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        for kind in ("collision", "decorative"):
            mesh = UsdGeom.Mesh.Define(stage, f"/Robot/{name}/{kind}")
            vertices = list(product((-.005, .005), (-.01, .01), (-.07, -.05)))
            mesh.CreatePointsAttr(vertices if kind == "collision" else [(x+100,y,z) for x,y,z in vertices])
            if kind == "collision":
                UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
    pads = finger_pads(stage, "/Robot", names)
    np.testing.assert_allclose(pads[0].center, (-.01, 0, -.12), atol=1e-6)
    np.testing.assert_allclose(pads[1].center, (.01, 0, -.12), atol=1e-6)
    assert all(p.meshes == ("collision",) for p in pads)


def test_closing_requires_flap_between_aligned_jaws_near_the_target():
    centers = torch.zeros(1, 2, 3)
    halves = torch.tensor([[[.002, .1, .1], [.002, .1, .1]]])
    pads = torch.tensor([[[[-.02, 0, .085], [.02, 0, .085]],
                         [[.01, 0, .085], [.03, 0, .085]]]])
    axes = torch.zeros(1, 2, dtype=torch.long)
    gap, error, capture = closing_geometry(pads, centers, halves, axes,
        torch.tensor([[0., .02]]), torch.ones(1, 2), .015)
    assert capture.tolist() == [[True, False]]
    torch.testing.assert_close(error, gap - .004)
    assert not closing_geometry(pads, centers, halves, axes,
        torch.tensor([[.1, .1]]), torch.ones(1, 2), .015)[2].any()
    assert not closing_geometry(pads, centers, halves, axes,
        torch.zeros(1, 2), torch.full((1, 2), .5), .015)[2].any()
