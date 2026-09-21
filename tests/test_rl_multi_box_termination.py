"""Low-level success and failure are mutually exclusive terminal outcomes."""

from dataclasses import replace

import torch

from kuavo_isaaclab_scene.rl.multi_box.success import (
    SkillTerminationInput,
    low_level_termination,
)


def sample():
    yes = torch.tensor([True])
    no = torch.tensor([False])
    return SkillTerminationInput(
        success=no,
        unsafe=no,
        phase_armed=yes,
        grasp_maintained=yes,
        box_tilt_ok=yes,
        released=no,
        placement_region=no,
    )


def test_carry_grasp_loss_and_tilt_violation_are_terminal_failures():
    value = sample()
    lost = low_level_termination(
        "carry", replace(value, grasp_maintained=torch.tensor([False])))
    assert lost.grasp_loss.item() and lost.failure.item() and lost.terminated.item()

    tilted = low_level_termination(
        "carry", replace(value, box_tilt_ok=torch.tensor([False])))
    assert tilted.tilt_violation.item() and tilted.failure.item()


def test_place_release_outside_region_is_terminal_failure():
    value = replace(
        sample(), released=torch.tensor([True]), placement_region=torch.tensor([False]))
    result = low_level_termination("place", value)
    assert result.premature_release.item() and result.terminated.item()


def test_failure_wins_if_success_arrives_on_same_step():
    value = replace(
        sample(), success=torch.tensor([True]), unsafe=torch.tensor([True]))
    result = low_level_termination("grasp", value)
    assert result.failure.item()
    assert not result.success.item()
    assert result.terminated.item()


def test_unarmed_carry_does_not_call_missing_grasp_a_loss():
    value = replace(
        sample(), phase_armed=torch.tensor([False]),
        grasp_maintained=torch.tensor([False]))
    result = low_level_termination("carry", value)
    assert not result.failure.item()
    assert not result.terminated.item()
