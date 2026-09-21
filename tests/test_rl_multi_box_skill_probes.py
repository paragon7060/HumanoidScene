"""Read-only Quest skill probes require an ordered physical grasp and placement."""

from types import SimpleNamespace

import torch

from kuavo_isaaclab_scene.rl.multi_box.debug.carry_probe import QuestCarryProbe
from kuavo_isaaclab_scene.rl.multi_box.debug.grasp_probe import QuestGraspProbe
from kuavo_isaaclab_scene.rl.multi_box.debug.place_probe import QuestPlaceProbe
from kuavo_isaaclab_scene.rl.multi_box.geometry.pad_distance import pad_to_boxes_clearance_m
from kuavo_isaaclab_scene.rl.multi_box.success import FingerFlapContacts


def _contacts(force=10.0):
    values = torch.zeros(1, 2, 2, 2)
    region = torch.zeros_like(values, dtype=torch.bool)
    opposed = torch.zeros(1, 2, 2, dtype=torch.bool)
    for hand, flap in ((0, 0), (1, 1)):
        values[0, hand, flap] = force
        region[0, hand, flap] = force > 0
        opposed[0, hand, flap] = True
    return FingerFlapContacts(values, region, opposed, torch.ones(1, dtype=torch.bool))


def _snapshot():
    return SimpleNamespace(
        target_logical_id=1,
        contacts=_contacts(),
        hand_to_box_pose=torch.tensor([[[0., 0., 0., 1., 0., 0., 0.],
                                        [0., 0., 0., 1., 0., 0., 0.]]]),
        rack_clearance_m=torch.tensor([0.009]),
        box_footprint_corners_belt=torch.tensor([
            [[-0.20, -0.10], [-0.20, 0.10], [0.20, -0.10], [0.20, 0.10]],
        ]),
        box_bottom_height_m=torch.tensor([0.10]),
        box_tilt_rad=torch.tensor([0.0]),
        overlaps_other_belt_box=torch.tensor([False]),
        belt_body_force_n=torch.tensor([0.0]),
        belt_sensor_available=torch.tensor([True]),
        gripper_box_distance_m=torch.tensor([[0.005, 0.005]]),
        raw_by_phase={"place": SimpleNamespace(
            long_axis_error_rad=torch.tensor([0.0]),
            linear_speed_mps=torch.tensor([0.0]),
            angular_speed_radps=torch.tensor([0.0]),
        )},
    )


def test_carry_and_place_probe_require_grasp_then_release_and_support():
    grasp, carry, place = QuestGraspProbe("cpu"), QuestCarryProbe("cpu"), QuestPlaceProbe("cpu")
    snapshot = _snapshot()
    for _ in range(8):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
        place_result = place.update(snapshot, grasp_result, carry_result, 1 / 30)
    assert grasp_result.success.success.item()
    assert carry_result.result.success.item()
    assert not place_result.result.success.item()

    snapshot.contacts = _contacts(0.0)
    snapshot.box_bottom_height_m = torch.tensor([0.0])
    snapshot.belt_body_force_n = torch.tensor([0.3])
    snapshot.gripper_box_distance_m = torch.tensor([[0.03, 0.03]])
    for _ in range(15):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
        place_result = place.update(snapshot, grasp_result, carry_result, 1 / 30)
    assert place_result.result.success.item()
    assert "success=1" in place_result.report()

    snapshot.target_logical_id = 2
    grasp_result = grasp.update(snapshot, 1 / 30)
    carry_result = carry.update(snapshot, grasp_result)
    assert not place.update(snapshot, grasp_result, carry_result, 1 / 30).result.success.item()


def test_place_release_does_not_require_pad_retreat():
    grasp, carry, place = QuestGraspProbe("cpu"), QuestCarryProbe("cpu"), QuestPlaceProbe("cpu")
    snapshot = _snapshot()
    for _ in range(8):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
        place.update(snapshot, grasp_result, carry_result, 1 / 30)
    assert carry_result.result.success.item()

    snapshot.contacts = _contacts(0.0)
    snapshot.box_bottom_height_m = torch.tensor([0.0])
    snapshot.belt_body_force_n = torch.tensor([0.3])
    # Opened hands may remain beside the box after physical release.
    snapshot.gripper_box_distance_m = torch.tensor([[0.005, 0.005]])
    for _ in range(15):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
        place_result = place.update(snapshot, grasp_result, carry_result, 1 / 30)
    assert place_result.result.released_and_clear.item()
    assert place_result.result.success.item()


def test_grasp_stability_reference_starts_after_rack_clearance():
    probe = QuestGraspProbe("cpu")
    snapshot = _snapshot()
    snapshot.rack_clearance_m = torch.tensor([0.0])

    # Normal shelf extraction can move the box relative to the first-contact
    # pose by more than the final 10 mm hold tolerance.
    for _ in range(3):
        result = probe.update(snapshot, 1 / 30)
    snapshot.hand_to_box_pose[:, :, 0] = 0.03
    for _ in range(3):
        result = probe.update(snapshot, 1 / 30)
    assert not result.stable_hands.any()

    # Clearing the rack establishes a fresh reference.  A stationary valid
    # opposing-flap grasp must then pass the unchanged 0.25 s proof hold.
    snapshot.rack_clearance_m = torch.tensor([0.009])
    for _ in range(8):
        result = probe.update(snapshot, 1 / 30)
    assert result.stable_hands.all()
    assert result.success.success.item()


def test_carry_stability_remains_active_below_rack_after_proof_lift():
    grasp, carry = QuestGraspProbe("cpu"), QuestCarryProbe("cpu")
    snapshot = _snapshot()

    # Complete the rack-relative proof lift and latch grasp completion.
    for _ in range(8):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
    assert grasp_result.success.success.item()
    assert carry_result.grasp_previously_completed

    # The conveyor is below the rack.  Lowering a still-stable, physically
    # valid pinch must not disable the hand-box stability tracker.
    snapshot.rack_clearance_m = torch.tensor([-0.20])
    grasp_result = grasp.update(snapshot, 1 / 30)
    carry_result = carry.update(snapshot, grasp_result)
    assert grasp_result.stable_hands.all()
    assert carry_result.result.grasp_maintained.item()
    assert carry_result.result.success.item()


def test_carry_uses_physical_pinch_after_grasp_proof_is_complete():
    grasp, carry = QuestGraspProbe("cpu"), QuestCarryProbe("cpu")
    snapshot = _snapshot()

    for _ in range(8):
        grasp_result = grasp.update(snapshot, 1 / 30)
        carry_result = carry.update(snapshot, grasp_result)
    assert grasp_result.success.success.item()

    # Normal compliant settling can exceed the proof-lift pose tolerance.
    # The opposing physical pinch remains valid and therefore so does carry.
    snapshot.hand_to_box_pose[:, :, 0] = 0.03
    grasp_result = grasp.update(snapshot, 1 / 30)
    carry_result = carry.update(snapshot, grasp_result)
    assert not grasp_result.success.stable.item()
    assert carry_result.result.grasp_maintained.item()
    assert carry_result.result.success.item()


def test_conservative_pad_distance_does_not_overstate_2cm_clearance():
    pad = torch.tensor([[0., 0., 0., 1., 0., 0., 0.]])
    box = torch.tensor([[0.04, 0., 0., 1., 0., 0., 0.]])
    center = torch.zeros(1, 3)
    half = torch.full((1, 3), 0.005)
    far = pad_to_boxes_clearance_m(pad, (0.002, 0.018, 0.020), box, center, half)
    assert far.item() >= 0.02
    box[0, 0] = 0.025
    close = pad_to_boxes_clearance_m(pad, (0.002, 0.018, 0.020), box, center, half)
    assert close.item() < 0.02
