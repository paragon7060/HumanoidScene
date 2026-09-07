"""Flap-top contact predicates; no application/physics startup required."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.mdp.flap_grasp import upper_band_contacts, grasp_status
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


def test_flap_task_accepts_one_or_two_hands_and_rejects_bad_settings():
    spec = dict(control_mode="arms-only", grasp_mode="flap_top", required_grasp_hands=2)
    task_spec("pick", **spec).validate()
    task_spec("pick", **{**spec, "required_grasp_hands": 1}).validate()
    with pytest.raises(ValueError, match="stationary"):
        task_spec("place", **spec, reset_bank="bank").validate()
    with pytest.raises(ValueError, match="grasp_hand"):
        task_spec("pick", **spec, grasp_hand="either").validate()
    with pytest.raises(ValueError, match="two distinct"):
        task_spec("pick", **spec, grasp_flaps=("flap_front", "flap_front")).validate()
    with pytest.raises(ValueError, match="top band"):
        task_spec("pick", **spec, flap_grasp_depth=.04).validate()


@pytest.mark.parametrize("side,index", [("left", 0), ("right", 1)])
def test_single_hand_requires_selected_opposed_jaws_and_allows_other_hand_support(side, index):
    spec = task_spec("pick", control_mode="arms-only", grasp_mode="flap_top", grasp_hand=side)
    p, c, h, a = sample()
    p[:, 1-index] = float("nan")
    valid, opposed = check(p, c, h, a)
    force = torch.zeros(1, 2, 2)
    force[:, index] = 1.0
    fingers, hands, grasped = grasp_status(valid, opposed, force, spec)
    assert hands[0, index] and not hands[0, 1-index] and grasped.item()
    # Contacts from the supporting hand neither replace nor invalidate the pinch.
    force[:, 1-index] = 10.
    assert grasp_status(valid, opposed, force, spec)[2].item()
    force[:, index, 0] = .1
    assert not grasp_status(valid, opposed, force, spec)[2].item()
    force[:, index, 0] = 1.
    opposed[:, index] = False
    assert not grasp_status(valid, opposed, force, spec)[2].item()


def test_two_hand_grasp_still_requires_both_and_batch_entries_are_independent():
    spec = task_spec("pick", control_mode="arms-only", grasp_mode="flap_top", required_grasp_hands=2)
    valid = torch.ones(3, 2, 2, dtype=torch.bool)
    valid[1, 0, 0] = False
    valid[2, 1, 1] = False
    opposed = torch.ones(3, 2, dtype=torch.bool)
    result = grasp_status(valid, opposed, torch.ones(3, 4), spec)
    assert result[2].tolist() == [True, False, False]
