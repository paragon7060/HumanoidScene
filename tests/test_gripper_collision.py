import pytest

from kuavo_isaaclab_scene.planning.gripper_collision import (
    GRIPPER_COLLISION_FRAMES,
    SPHERE_COORDINATE_FRAME,
    SUPPORTED_MAX_OVERSHOOT_M,
    load_gripper_collision_spheres,
    load_gripper_mesh_bounds,
)


def test_generated_spheres_use_urdf_link_coordinates():
    assert SPHERE_COORDINATE_FRAME == "urdf_link_frame"
    assert len(GRIPPER_COLLISION_FRAMES) == 28
    assert "l_f_bar_4" in GRIPPER_COLLISION_FRAMES
    assert "r_b_bar_4" in GRIPPER_COLLISION_FRAMES
    assert "l_d405_camera" in GRIPPER_COLLISION_FRAMES
    assert "r_d405_camera" in GRIPPER_COLLISION_FRAMES


@pytest.mark.parametrize("max_overshoot_m", SUPPORTED_MAX_OVERSHOOT_M)
def test_generated_gripper_collision_sphere_presets_are_complete(max_overshoot_m):
    frames = load_gripper_collision_spheres(max_overshoot_m)

    assert tuple(frames) == GRIPPER_COLLISION_FRAMES
    assert all(frames.values())
    assert all(
        sphere["radius"] > 0
        for spheres in frames.values()
        for sphere in spheres
    )


def test_generated_gripper_collision_spheres_reject_unknown_preset():
    with pytest.raises(ValueError, match="max overshoot"):
        load_gripper_collision_spheres(0.003)


def test_generated_gripper_mesh_bounds_are_complete():
    bounds = load_gripper_mesh_bounds()

    assert tuple(bounds) == GRIPPER_COLLISION_FRAMES
    assert all(
        all(high[index] > low[index] for index in range(3))
        for low, high in bounds.values()
    )
