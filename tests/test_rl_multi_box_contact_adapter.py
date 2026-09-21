"""Exercise the v2 contact-to-grasp bridge without starting Isaac Sim."""

import math
from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.success import (
    FingerFlapContacts,
    RelativePoseStabilityTracker,
    classify_pinches,
)
from kuavo_isaaclab_scene.rl.multi_box.geometry.rack import box_shelf_clearance_m
from kuavo_isaaclab_scene.rl.multi_box.debug.grasp_probe import QuestGraspProbe
from kuavo_isaaclab_scene.rl.multi_box.debug.contact_force import maximum_filtered_force
from kuavo_isaaclab_scene.rl.multi_box.scene.spawn import logical_cells
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec
from kuavo_isaaclab_scene.workcell.rack_box_layout import BOX_DIMENSIONS_M
from kuavo_isaaclab_scene.workcell.workcell_layout import scale


def _contacts():
    force = torch.zeros(1, 2, 2, 2)
    region = torch.zeros_like(force, dtype=torch.bool)
    opposed = torch.zeros(1, 2, 2, dtype=torch.bool)
    for hand, flap in ((0, 0), (1, 1)):
        force[0, hand, flap] = 10.0
        region[0, hand, flap] = True
        opposed[0, hand, flap] = True
    return FingerFlapContacts(force, region, opposed, torch.ones(1, dtype=torch.bool))


def test_filtered_rack_force_takes_the_maximum_over_links_and_contacts():
    scene = {
        "rack_0": SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.tensor([
                [[[3.0, 4.0, 0.0]]],
                [[[0.0, 0.0, 0.0]]],
            ]),
        )),
        "rack_1": SimpleNamespace(data=SimpleNamespace(
            force_matrix_w=torch.tensor([
                [[[0.0, 0.0, 12.0]]],
                [[[0.0, 8.0, 0.0]]],
            ]),
        )),
    }
    env = SimpleNamespace(scene=scene, num_envs=2)
    torch.testing.assert_close(
        maximum_filtered_force(env, ("rack_0", "rack_1")),
        torch.tensor([12.0, 8.0]),
    )


def test_each_hand_must_have_two_valid_jaws_on_one_flap():
    contacts = _contacts()
    pinch = classify_pinches(contacts, min_jaw_force_n=5.0)
    assert pinch.hand_pinching.tolist() == [[True, True]]
    assert pinch.hand_flap_index.tolist() == [[0, 1]]

    contacts.force_n[0, 0, 0, 1] = 4.9
    assert classify_pinches(contacts, min_jaw_force_n=5.0).hand_pinching.tolist() == [[False, True]]
    contacts.force_n[0, 0, 0, 1] = 10.0
    contacts.in_region[0, 1, 1, 0] = False
    assert classify_pinches(contacts, min_jaw_force_n=5.0).hand_pinching.tolist() == [[True, False]]


def test_ambiguous_or_unavailable_contact_never_establishes_pinch():
    contacts = _contacts()
    contacts.force_n[0, 0, 1] = 10.0
    contacts.in_region[0, 0, 1] = True
    contacts.opposed[0, 0, 1] = True
    pinch = classify_pinches(contacts, min_jaw_force_n=5.0)
    assert pinch.ambiguous_hands.tolist() == [[True, False]]
    assert pinch.hand_flap_index.tolist() == [[-1, 1]]
    contacts.available[0] = False
    assert not classify_pinches(contacts, min_jaw_force_n=5.0).hand_pinching.any()


def test_relative_pose_drift_is_measured_from_initial_pinch_and_resets_on_release():
    tracker = RelativePoseStabilityTracker(1, "cpu")
    pose = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0.]]])
    pinch = torch.tensor([[True, True]])
    assert tracker.update(pose, pinch).all()
    pose[0, 0, 0] = 0.011
    assert tracker.update(pose, pinch).tolist() == [[False, True]]
    pose[0, 0, 0] = 0.0
    angle = math.radians(11.0) / 2
    pose[0, 1, 3:] = torch.tensor([math.cos(angle), 0., 0., math.sin(angle)])
    assert tracker.update(pose, pinch).tolist() == [[True, False]]
    tracker.update(pose, torch.zeros_like(pinch))
    assert tracker.update(pose, pinch).all()


def test_sloped_shelf_clearance_matches_spawn_gap_and_lift():
    cell = logical_cells(MultiBoxSpec())[0]
    position, rotation = cell.local_pose("small", scale("rack"))
    box_pose = torch.tensor([(*position, *rotation)])
    rack_pose = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
    dimensions = BOX_DIMENSIONS_M["small"]
    initial = box_shelf_clearance_m(
        box_pose, rack_pose, dimensions, shelf=2, rack_scale=scale("rack"))
    assert torch.allclose(initial, torch.tensor([0.008]), atol=1e-4)
    box_pose[:, 2] += 0.008
    lifted = box_shelf_clearance_m(
        box_pose, rack_pose, dimensions, shelf=2, rack_scale=scale("rack"))
    assert torch.allclose(lifted - initial, torch.tensor([0.008]), atol=1e-5)


def test_quest_probe_requires_hold_and_resets_when_target_changes():
    probe = QuestGraspProbe("cpu")
    pose = torch.tensor([[[0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 1., 0., 0., 0.]]])
    snapshot = SimpleNamespace(
        target_logical_id=2, contacts=_contacts(), hand_to_box_pose=pose,
        rack_clearance_m=torch.tensor([0.009]),
    )
    for _ in range(7):
        result = probe.update(snapshot, 1 / 30)
        assert not result.success.success.item()
    result = probe.update(snapshot, 1 / 30)
    assert result.success.success.item()
    assert "success=1" in result.report()
    snapshot.target_logical_id = 3
    assert not probe.update(snapshot, 1 / 30).success.success.item()
