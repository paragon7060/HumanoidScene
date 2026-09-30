"""The two-axis torso command must never request forward pitch."""

import torch

from kuavo_isaaclab_scene.rl.multi_box.geometry.upright_torso import (
    planar_position, torso_links_from_urdf, upright_joint_step,
)
from kuavo_isaaclab_scene.robots.robot_model import resolve_robot_model


def test_batched_torso_xz_motion_preserves_pitch_and_joint_limits():
    links = torch.tensor(torso_links_from_urdf(
        resolve_robot_model("s63", "leju-twofinger").urdf_path))
    initial = torch.tensor([[0.231, -0.473, 0.263], [0.350, -0.650, 0.300]])
    reference = initial.sum(-1)
    limits = torch.tensor([[[0.0, 1.448], [-2.809, 0.0], [-0.157, 2.984]]]).expand(2, -1, -1)
    current = initial.clone()
    start = planar_position(current[:, :2], links)
    for _ in range(25):
        target = planar_position(current[:, :2], links) + torch.tensor([[0.003, 0.002], [-0.002, 0.003]])
        current, _ = upright_joint_step(current, target, reference, links, limits)
        torch.testing.assert_close(current.sum(-1), reference, atol=1e-6, rtol=0)
        assert bool((current >= limits[..., 0]).all())
        assert bool((current <= limits[..., 1]).all())
    finish = planar_position(current[:, :2], links)
    assert finish[0, 0] > start[0, 0]
    assert finish[0, 1] > start[0, 1]
    assert finish[1, 0] < start[1, 0]
    assert finish[1, 1] > start[1, 1]


def test_upright_torso_zero_command_holds_initial_pose():
    links = torch.tensor(torso_links_from_urdf(
        resolve_robot_model("s63", "leju-twofinger").urdf_path))
    current = torch.tensor([[0.231, -0.473, 0.263]])
    limits = torch.tensor([[[0.0, 1.448], [-2.809, 0.0], [-0.157, 2.984]]])
    held, achieved = upright_joint_step(
        current, planar_position(current[:, :2], links), current.sum(-1), links, limits)
    torch.testing.assert_close(held, current, atol=1e-6, rtol=0)
    torch.testing.assert_close(achieved, planar_position(current[:, :2], links))
