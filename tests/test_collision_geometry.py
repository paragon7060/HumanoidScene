"""Pure CPU checks, no USD stage, physics, or rendering."""

import numpy as np
from kuavo_isaaclab_scene.rl.debug.collision_geometry import (
    polygon_geometry, hull_polygons, inward_face, force_arrow)
from kuavo_isaaclab_scene.rl.debug.collision_overlay import rotate_points


def cube():
    vertices = np.array([(x,y,z) for x in (-1.,1.) for y in (-1.,1.) for z in (-1.,1.)])
    faces = [[0,1,3,2], [4,6,7,5], [0,4,5,1], [2,3,7,6], [0,2,6,4], [1,5,7,3]]
    return vertices, faces


def test_polygon_center_area_and_normal():
    center, normal, area = polygon_geometry([[0,0,0],[2,0,0],[2,2,0],[0,2,0]])
    np.testing.assert_allclose(center, [1,1,0])
    np.testing.assert_allclose(normal, [0,0,1])
    assert area == 4


def test_inward_face_uses_opposing_direction_and_reflected_shapes():
    vertices, faces = cube()
    vertices[:, 0] *= -1
    polys = hull_polygons(vertices, faces)
    _, center, normal = inward_face(polys, [1.,0.,0.])
    np.testing.assert_allclose(center, [1,0,0])
    np.testing.assert_allclose(normal, [1,0,0])
    _, center, _ = inward_face(polys, [-1.,0.,0.])
    np.testing.assert_allclose(center, [-1,0,0])
    assert inward_face(polys, [0.,0.,0.]) is None


def test_pair_force_arrow_uses_direction_scale_and_cap():
    arrow = force_arrow(np.zeros(3), [0.,2.,0.])
    np.testing.assert_allclose(arrow[0, 1], [0,.02,0])
    arrow = force_arrow(np.zeros(3), [100.,0.,0.])
    np.testing.assert_allclose(arrow[0, 1], [.12,0,0])
    assert not len(force_arrow(np.full(3, np.nan), [1,0,0]))
    assert not len(force_arrow(np.zeros(3), [0,0,0]))


def test_live_rigid_rotation_translation():
    result = rotate_points([[1,0,0]], [8,2,3,2**-.5,0,0,2**-.5])
    np.testing.assert_allclose(result, [[8,3,3]], atol=1e-12)
