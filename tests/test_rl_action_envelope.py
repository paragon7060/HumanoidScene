"""Ensure the replay-only envelope preserves arms and follows named terms."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.action_envelope import diagnostic_body_limits, mix_teacher_actor

TERMS = [('base', 3), ('upper_body', 15), ('height', 2),
         ('left_gripper', 1), ('right_gripper', 1), ('head', 2)]


def test_correction_probe_preserves_teacher_jaws_and_limits_actor_influence():
    teacher = torch.full((3, 24), .2)
    teacher[:, 20:22] = torch.tensor([-1., 1.])
    actor = -torch.ones_like(teacher)
    executed = mix_teacher_actor(teacher, actor, .05, gripper_columns=(20, 21))
    assert torch.allclose(executed[:, :20], torch.full((3, 20), .14))
    assert torch.equal(executed[:, 20:22], teacher[:, 20:22])
    assert torch.equal(mix_teacher_actor(teacher, actor, 0, gripper_columns=(20, 21)), teacher)
    assert torch.equal(teacher[:, :20], torch.full((3, 20), .2))
    with pytest.raises(ValueError, match='mix'):
        mix_teacher_actor(teacher, actor, .3, gripper_columns=(20, 21))


def test_body_command_limits_retain_full_arms_and_jaws():
    limits = diagnostic_body_limits(TERMS)
    assert limits[:4].tolist() == pytest.approx([.2, .2, .4, .35])
    assert torch.equal(limits[4:18], torch.ones(14))
    assert limits[18:20].tolist() == pytest.approx([.2, .2])
    assert torch.equal(limits[20:], torch.ones(4))
    actual = torch.tensor([[.1959, -.1421, .3672, -.3236]+[1.]*14+[.1764, -.1586, 1., -1., 0., 0.]])
    assert torch.equal(actual.clamp(-limits, limits), actual)


def test_envelope_mapping_uses_term_names_and_rejects_older_torso():
    limits = diagnostic_body_limits(list(reversed(TERMS)))
    assert limits[:2].tolist() == [1., 1.]
    assert limits[-3:].tolist() == pytest.approx([.2, .2, .4])
    with pytest.raises(ValueError, match='24-D'):
        diagnostic_body_limits([(name, 3 if name == 'height' else width) for name, width in TERMS])
    with pytest.raises(ValueError, match='24-D'):
        diagnostic_body_limits(TERMS+[('base', 3)])
