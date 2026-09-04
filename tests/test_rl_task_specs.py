"""Pure configuration checks; no Isaac Lab runtime import."""

import pytest
from kuavo_isaaclab_scene.rl.tasks.specs import TASKS, PREDECESSOR, task_spec


@pytest.mark.parametrize("name", TASKS)
def test_skill_preset(name):
    spec = task_spec(name, reset_bank="/example/bank" if name in ("carry", "place", "press_button") else None)
    spec.validate()
    assert spec.name == name
    assert spec.episode_length_s > spec.hold_seconds


@pytest.mark.parametrize("name", ("carry", "place", "press_button"))
def test_no_floating_box_reset(name):
    with pytest.raises(ValueError, match=PREDECESSOR[name]):
        task_spec(name).validate()


def test_pick_may_use_approach_snapshot():
    task_spec("pick", reset_bank="/example/approach").validate()


def test_no_empty_task_or_duplicate_boxes():
    for boxes in ((), ("small_box_0", "small_box_0")):
        with pytest.raises(ValueError, match="unique"):
            task_spec("full", box_names=boxes).validate()


def test_overfilled_stopped_conveyor_rejected():
    with pytest.raises(ValueError, match="space"):
        task_spec("full", box_names=("a", "b", "c", "d"), prefill_count=1).validate()


def test_full_cannot_start_from_a_grasp_bank():
    with pytest.raises(ValueError, match="fresh rack"):
        task_spec("full", reset_bank="/example/bank").validate()
