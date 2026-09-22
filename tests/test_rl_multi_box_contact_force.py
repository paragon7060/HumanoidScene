"""V2 rack and surrounding-obstacle force separation."""

from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import (
    maximum_filtered_force,
    maximum_non_rack_force,
)
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec


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
