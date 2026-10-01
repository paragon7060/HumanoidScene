import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.joint_offset import JointOffsetController


def test_joint_goal_labels_decode_to_identical_physical_delta_commands():
    mapper = JointOffsetController(.2)
    obs = torch.zeros(8, 464); obs[:, 439] = 1
    obs[:, 416:436] = torch.randn(8, 20) * .01
    command = torch.rand(8, 24) * 2 - 1
    goals = mapper.label_coordinates(obs, command)
    actual = mapper.physical_commands(obs, goals)
    assert torch.allclose(actual, command, atol=1e-6)
    assert torch.equal(goals[:, :3], command[:, :3])
    assert torch.equal(goals[:, 18:22], command[:, 18:22])


def test_analytic_pending_target_feedback_is_bounded_and_does_not_mutate_critic():
    mapper = JointOffsetController(.2)
    obs = torch.zeros(1, 464); obs[:, 439] = 1; obs[:, 419] = .01
    saved = obs.clone()
    assert mapper.physical_commands(obs, torch.zeros(1, 24))[0, 3] == -1
    assert mapper.actor_input(obs)[0, 419] == 0
    assert torch.equal(saved, obs)
    command = torch.ones(1, 24)
    actual = mapper.physical_commands(obs, command)
    assert actual.abs().max() <= 1
    obs[:, 439] = 0
    with pytest.raises(ValueError, match='telemetry'):
        mapper.physical_commands(obs, command)
