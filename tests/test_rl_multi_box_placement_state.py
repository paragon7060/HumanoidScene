"""CPU checks for live placement and full-task success state."""

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import HighLevelSelection
from kuavo_isaaclab_scene.rl.multi_box.state import PlacementSetTracker


def test_all_active_boxes_must_be_live_placed_for_full_success():
    tracker = PlacementSetTracker(2, "cpu")
    active = torch.zeros(2, 12, dtype=torch.bool)
    active[0, :2] = True
    active[1, :3] = True
    valid = active.clone()
    valid[1, 2] = False
    state = tracker.update(active, valid, 0.5)
    assert state.full_success.tolist() == [True, False]
    assert state.placed[0, :2].all()
    assert state.selectable[1, 2]
    assert not state.selectable[0, :2].any()


def test_displaced_box_immediately_loses_placed_and_becomes_selectable():
    tracker = PlacementSetTracker(1, "cpu")
    active = torch.zeros(1, 12, dtype=torch.bool)
    active[0, 4] = True
    state = tracker.update(active, active.clone(), 0.5)
    assert state.placed[0, 4] and state.full_success[0]
    valid = active.clone()
    valid[0, 4] = False
    state = tracker.update(active, valid, 0.01)
    assert not state.placed[0, 4]
    assert state.selectable[0, 4]
    assert not state.full_success[0]
    assert state.hold_time_s[0, 4] == 0


def test_empty_scene_is_not_a_vacuous_success_and_reset_is_per_environment():
    tracker = PlacementSetTracker(2, "cpu")
    empty = torch.zeros(2, 12, dtype=torch.bool)
    assert not tracker.update(empty, empty, 1.0).full_success.any()
    active = empty.clone(); active[:, 0] = True
    tracker.update(active, active, 0.5)
    tracker.reset(torch.tensor([0]))
    assert not tracker.placed[0, 0] and tracker.placed[1, 0]


def test_high_level_cannot_select_placed_box():
    active = torch.zeros(1, 12, dtype=torch.bool); active[0, 3] = True
    selectable = active.clone(); selectable[0, 3] = False
    action = HighLevelSelection(torch.tensor([3]))
    with pytest.raises(ValueError, match="selectable"):
        action.validate(selectable)
