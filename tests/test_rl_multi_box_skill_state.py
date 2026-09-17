"""CPU checks for locked-target deployable skill orchestration."""

import torch

from kuavo_isaaclab_scene.rl.multi_box.hierarchy import (
    DeployableTransitionEvidence,
    HighLevelSelection,
    SKILL_IDS,
    SkillStateMachine,
)


def evidence(n, *, grasp=(), carry=(), place=()):
    values = [torch.zeros(n, dtype=torch.bool) for _ in range(3)]
    for tensor, indices in zip(values, (grasp, carry, place)):
        tensor[list(indices)] = True
    return DeployableTransitionEvidence(*values)


def test_high_level_selects_only_box_and_target_stays_locked_through_skills():
    machine = SkillStateMachine(2, "cpu")
    selectable = torch.zeros(2, 12, dtype=torch.bool)
    selectable[0, 3] = selectable[1, 8] = True
    machine.assign(HighLevelSelection(torch.tensor([3, 8])), selectable)
    assert machine.state().target_box.tolist() == [3, 8]
    assert machine.state().current_skill.tolist() == [SKILL_IDS["grasp"]] * 2

    state = machine.advance(evidence(2, grasp=(0, 1), carry=(0, 1), place=(0, 1)))
    assert state.target_box.tolist() == [3, 8]
    assert state.current_skill.tolist() == [SKILL_IDS["carry"]] * 2
    state = machine.advance(evidence(2, carry=(0, 1)))
    assert state.current_skill.tolist() == [SKILL_IDS["place"]] * 2
    state = machine.advance(evidence(2, place=(0, 1)))
    assert state.completed_box.tolist() == [3, 8]
    assert state.needs_target.all()


def test_transition_advances_at_most_one_skill_per_control_step():
    machine = SkillStateMachine(1, "cpu")
    selectable = torch.zeros(1, 12, dtype=torch.bool)
    selectable[0, 5] = True
    machine.assign(HighLevelSelection(torch.tensor([5])), selectable)
    all_true = evidence(1, grasp=(0,), carry=(0,), place=(0,))
    assert machine.advance(all_true).current_skill.item() == SKILL_IDS["carry"]
    assert machine.advance(all_true).current_skill.item() == SKILL_IDS["place"]
    assert machine.advance(all_true).completed_box.item() == 5


def test_current_skill_is_explicit_as_id_and_one_hot():
    machine = SkillStateMachine(1, "cpu")
    state = machine.state()
    assert state.current_skill.item() == SKILL_IDS["grasp"]
    assert state.current_skill_one_hot.tolist() == [[1.0, 0.0, 0.0]]
