"""Paired marker geometry only: no simulator or renderer."""

import torch
from kuavo_isaaclab_scene.rl.debug.grasp_markers import paired_face_targets


def test_pair_is_on_opposite_faces_with_common_in_bounds_anchor():
    points = torch.tensor([[[.04, .03, .2], [-.02, .05, .16]]])
    goals, cost = paired_face_targets(points, torch.zeros(1, 3),
        torch.tensor([[.002, .1, .055]]), torch.tensor([0]))
    torch.testing.assert_close(goals, torch.tensor([[[.002, .04, .055], [-.002, .04, .055]]]))
    torch.testing.assert_close(cost, (points-goals).norm(dim=-1).amax(-1))


def test_jaw_order_can_reverse_without_choosing_the_same_face():
    points = torch.tensor([[[-.03, 0., 0.], [.04, 0., 0.]]])
    goals, _ = paired_face_targets(points, torch.zeros(1, 3),
        torch.tensor([[.002, .1, .055]]), torch.tensor([0]))
    assert goals[0, 0, 0] < 0 < goals[0, 1, 0]


def test_candidates_have_independent_axis_center_and_bounds():
    centers = torch.tensor([[0., 0., 0.], [.5, .6, .7]])
    halves = torch.tensor([[.002, .1, .055], [.1, .003, .06]])
    points = centers[:, None] + torch.tensor([[[.04, .01, 0.], [-.04, .01, 0.]],
                                             [[.03, -.02, 0.], [.03, .02, 0.]]])
    goals, _ = paired_face_targets(points, centers, halves, torch.tensor([0, 1]))
    relative = goals - centers[:, None]
    torch.testing.assert_close(relative[1, :, 1], torch.tensor([-.003, .003]))
    assert (relative.abs() <= halves[:, None] + 1e-6).all()
