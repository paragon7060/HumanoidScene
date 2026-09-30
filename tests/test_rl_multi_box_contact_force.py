"""V2 rack and surrounding-obstacle force separation."""

from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import (
    maximum_filtered_force,
    maximum_non_rack_force,
    eligible_obstacle_targets,
    per_body_filtered_forces,
)
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


def test_eligible_obstacles_exclude_task_boxes_and_floor_by_construction():
    scene = SimpleNamespace(**{
        name: SimpleNamespace(prim_path="/World/" + name)
        for name in ("ground", "rack", "s2_small_0", "conveyor_surface",
                     "conveyor_rail_left", "conveyor_rail_right", "conveyor_leg_0")})
    paths = eligible_obstacle_targets(scene)
    assert set(paths) == {"/World/" + name for name in (
        "conveyor_surface", "conveyor_rail_left", "conveyor_rail_right", "conveyor_leg_0")}


def test_individual_obstacle_pair_max_does_not_sum_or_cancel_contacts():
    force = torch.zeros(1, 1, 2, 3)
    force[0, 0, :, 0] = torch.tensor([6.0, -6.0])
    # A net resultant would vanish; safety must see each actual 6 N contact.
    per_body = per_body_filtered_forces((force,))
    torch.testing.assert_close(per_body, torch.tensor([[6.0]]))
    assert per_body.amax().item() > 5.0


def test_rack_force_is_removed_without_hiding_simultaneous_obstacle_contact():
    rack_force = torch.tensor([5.0, 12.0, 5.0])
    other_force = torch.tensor([0.0, 0.0, 6.0])
    net = torch.zeros(3, 2, 3)
    net[:, 1, 0] = rack_force + other_force
    rack_matrix = torch.zeros(3, 1, 1, 3)
    rack_matrix[:, 0, 0, 0] = rack_force
    empty_matrix = torch.zeros_like(rack_matrix)
    env = SimpleNamespace(
        num_envs=3,
        scene={
            "base": SimpleNamespace(data=SimpleNamespace(force_matrix_w=rack_matrix)),
            "arm": SimpleNamespace(data=SimpleNamespace(force_matrix_w=empty_matrix)),
        },
    )

    measured_rack = maximum_filtered_force(env, ("base", "arm"))
    measured_other = maximum_non_rack_force(net, (rack_matrix, empty_matrix), (1, 0))
    torch.testing.assert_close(measured_rack, rack_force)
    torch.testing.assert_close(measured_other, other_force, atol=1e-6, rtol=0)
    assert ((measured_rack > 10.0) | (measured_other > 5.0)).tolist() == [
        False, True, True,
    ]
    spec = MultiBoxSpec()
    spec.validate()
    assert spec.rack_contact_force == 10.0
