"""Contact acquisition/retention must stay bounded and isolated across resets."""

import torch
from kuavo_isaaclab_scene.rl.mdp.grasp_contact_latch import GraspContactLatch, opposed_jaws
from kuavo_isaaclab_scene.rl.tasks.specs import task_spec


def setup():
    latch = GraspContactLatch(2, "cpu", task_spec("pick", grasp_contact_grace_s=.1))
    jaw = torch.tensor([[[[-.01, 0., 0.], [.01, 0., 0.]],
                         [[-.01, 0., 0.], [.01, 0., 0.]]]]).repeat(2, 1, 1, 1)
    yes = torch.ones(2, 2, dtype=torch.bool)
    return latch, jaw, yes


def test_filtered_force_alone_cannot_acquire_without_upper_band_grasp():
    latch, jaw, yes = setup()
    assert not latch.update(~yes, yes, jaw, 0, 1/30).any()


def test_short_contact_dropout_is_bounded_and_not_counted_twice():
    latch, jaw, yes = setup()
    assert latch.update(yes, yes, jaw, 0, 1/30).all()
    assert latch.update(~yes, ~yes, jaw, 1, 1/30).all()
    gap = latch.missing_s.clone()
    latch.update(~yes, ~yes, jaw, 1, 1/30)
    torch.testing.assert_close(latch.missing_s, gap)
    assert latch.update(~yes, ~yes, jaw, 2, 1/30).all()
    assert not latch.update(~yes, ~yes, jaw, 3, 1/30).any()


def test_live_contact_retains_grasp_outside_strict_band():
    latch, jaw, yes = setup()
    latch.update(yes, yes, jaw, 0, 1/30)
    jaw[..., 2] += .01
    assert latch.update(~yes, yes, jaw, 1, 1/30).all()


def test_opening_or_relative_slip_releases_immediately():
    latch, jaw, yes = setup()
    latch.update(yes, yes, jaw, 0, 1/30)
    jaw[0, :, 0, 0] -= .02
    jaw[1, ..., 2] += .03
    assert not latch.update(~yes, ~yes, jaw, 1, 1/30).any()


def test_verified_contact_overrides_origin_motion_and_updates_dropout_reference():
    latch, jaw, yes = setup()
    latch.update(yes, yes, jaw, 0, 1/30)
    jaw[..., 2] += .08  # Greater than the old 2 cm lifetime limit.
    jaw[:, :, 0, 0] -= .02  # Opening proxy also exceeds the old 8 mm limit.
    assert latch.update(~yes, yes, jaw, 1, 1/30).all()
    torch.testing.assert_close(latch.reference_midpoint, jaw.mean(-2))
    assert not latch.missing_s.any()
    # Brief missing contacts use the latest verified geometry, not initial geometry.
    assert latch.update(~yes, ~yes, jaw, 2, 1/30).all()
    jaw[..., 2] += .03
    assert not latch.update(~yes, ~yes, jaw, 3, 1/30).any()


def test_partial_reset_does_not_reset_other_environment():
    latch, jaw, yes = setup()
    latch.update(yes, yes, jaw, 0, 1/30)
    latch.reset(torch.tensor([0]))
    flags = latch.update(~yes, yes, jaw, 0, 1/30)
    assert not flags[0].any() and flags[1].all()


def test_actual_jaws_can_straddle_thin_plate_contact_midplane():
    _, jaw, _ = setup()
    centers = torch.zeros(2, 2, 3)
    axes = torch.zeros(2, 2, dtype=torch.long)
    assert opposed_jaws(jaw, centers, axes).all()
    jaw[..., 0] += .1
    assert not opposed_jaws(jaw, centers, axes).any()
