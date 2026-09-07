"""Flap-top contact predicates; no application/physics startup required."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.flap_grasp import upper_band_contacts
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


def sample():
    points = torch.tensor([[[[-.002, 0., .04], [.002, 0., .04]],
                            [[-.002, 0., .04], [.002, 0., .04]]]])
    centers = torch.zeros(1, 2, 3)
    half = torch.tensor([[[.002, .1, .055], [.002, .1, .055]]])
    axes = torch.zeros(1, 2, dtype=torch.long)
    return points, centers, half, axes


def check(points, centers, half, axes):
    return upper_band_contacts(points, centers, half, axes, band=.03, margin=.004)


def test_upper_edges_need_two_opposed_jaws():
    p, c, h, a = sample()
    valid, opposite = check(p, c, h, a)
    assert valid.all() and opposite.all()
    p[0, 0, 0, 0] = .002
    valid, opposite = check(p, c, h, a)
    assert valid.all() and not opposite[0, 0] and opposite[0, 1]


def test_bottom_contact_and_absent_contact_do_not_count():
    p, c, h, a = sample()
    p[0, 0, :, 2] = 0.0
    p[0, 1, 0] = float("nan")
    valid, _ = check(p, c, h, a)
    assert not valid[0, 0].any() and not valid[0, 1, 0] and valid[0, 1, 1]


def test_contact_outside_plate_width_is_invalid():
    p, c, h, a = sample()
    p[0, 0, 0, 1] = .2
    valid, _ = check(p, c, h, a)
    assert not valid[0, 0, 0]


def test_other_flap_normal_and_local_offset():
    p, c, h, a = sample()
    p = p[..., [1, 0, 2]]
    h = h[..., [1, 0, 2]]
    c += torch.tensor([.02, .01, .03])
    p += c[:, :, None]
    valid, opposite = check(p, c, h, a + 1)
    assert valid.all() and opposite.all()


def test_flap_task_rejects_wrong_stage_and_single_hand():
    spec = dict(control_mode="arms-only", grasp_mode="flap_top", required_grasp_hands=2)
    task_spec("pick", **spec).validate()
    with pytest.raises(ValueError, match="two hands"):
        task_spec("pick", **{**spec, "required_grasp_hands": 1}).validate()
    with pytest.raises(ValueError, match="two distinct"):
        task_spec("pick", **spec, grasp_flaps=("flap_front", "flap_front")).validate()
    with pytest.raises(ValueError, match="top band"):
        task_spec("pick", **spec, flap_grasp_depth=.04).validate()
