"""Task semantics regression tests, CPU only; never launch Isaac Sim."""
from dataclasses import replace
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.spec import (
    DEFAULT_RACK_REGIONS,
    MAX_BOXES,
    PREDECESSOR,
    SCHEMA_VERSION,
    SKILLS,
    MultiBoxSpec,
    RackRegionSpec,
)
from kuavo_isaaclab_scene.rl.multi_box.kernels import placement_mask, advance_placement, potential_delta


def sample():
    local = torch.tensor([[[-.9, 0., .115], [-.3, 0., .115], [.3, 0., .115], [.9, 0., .115]]]).repeat(2, 1, 1)
    return local, torch.full_like(local, .1)


def placed(local, half, **overrides):
    args = dict(contact=torch.ones(2, 4), released=torch.ones(2, 4, dtype=torch.bool),
        speed=torch.zeros(2, 4), angular=torch.zeros(2, 4), upright=torch.ones(2, 4))
    args.update(overrides)
    return placement_mask(local, half, torch.tensor([1.275, .34]), **args,
        clearance=.015, tolerance=.035, min_force=.2, max_speed=.08, max_angular=.35, min_up=.76)


def test_all_four_must_be_supported_released_and_nonoverlapping():
    local, half = sample()
    assert placed(local, half).all()
    contacts = torch.ones(2, 4); contacts[0, 1] = 0
    assert not placed(local, half, contact=contacts)[0, 1]
    released = torch.ones(2, 4, dtype=torch.bool); released[1, 3] = False
    assert not placed(local, half, released=released)[1, 3]
    local[0, 1] = local[0, 0]
    mask = placed(local, half)
    assert not mask[0, :2].any() and mask[1].all()


def test_floating_overhanging_and_fast_boxes_are_not_placed():
    local, half = sample()
    local[0, 0, 2] += .2
    local[0, 1, 1] = .3
    speed = torch.zeros(2, 4); speed[0, 2] = 1
    assert not placed(local, half, speed=speed)[0, :3].any()


def test_placement_credit_cannot_be_farmed_and_final_success_is_live():
    timer = torch.zeros(2, 4); paid = torch.zeros(2, 4, dtype=torch.bool)
    valid = torch.ones(2, 4, dtype=torch.bool)
    for _ in range(5):
        timer, paid, credit, complete = advance_placement(valid, timer, paid, .1, .5)
    assert credit.all() and complete.all()
    valid[0, 2] = False
    timer, paid, credit, complete = advance_placement(valid, timer, paid, .1, .5)
    assert not complete[0].all() and complete[1].all() and not credit.any()
    valid[0, 2] = True
    for _ in range(5):
        timer, paid, credit, complete = advance_placement(valid, timer, paid, .1, .5)
        assert not credit.any()
    assert complete.all()


def test_potential_closed_loops_and_terminal_cancellation():
    values = [0., 1., 3., 0., 1., 3., 0.]
    total = 0.
    for i, (a, b) in enumerate(zip(values, values[1:])):
        total += .99**i * potential_delta(torch.tensor(a), torch.tensor(b), torch.tensor(False), .99)
    assert abs(total) < 1e-6
    assert potential_delta(torch.tensor(3.), torch.tensor(9.), torch.tensor(True), .99) == -3


def test_v2_contract_has_twelve_boxes_and_only_requested_rack_regions():
    spec = MultiBoxSpec()
    spec.validate()
    assert spec.schema_version == SCHEMA_VERSION == 2
    assert spec.max_boxes == MAX_BOXES == 12
    assert spec.region_names == (
        "shelf_2_right", "shelf_2_left", "shelf_3_right", "shelf_3_left")
    assert spec.spawn_shelves == (2, 3)
    assert spec.allowed_box_types("shelf_2_right") == ("small", "medium")
    assert spec.allowed_box_types("shelf_2_left") == ("small", "medium")
    assert spec.allowed_box_types("shelf_3_right") == ("small",)
    assert spec.allowed_box_types("shelf_3_left") == ("small",)
    assert all(region.box_type_sampling == "uniform" for region in spec.rack_regions)
    assert spec.region_sampling == "uniform"
    assert spec.spawn_count_range == (1, 12)
    assert spec.episode_seconds is None
    assert spec.workspace_radius == 1.5
    assert spec.collision_constraints_enabled


def test_each_low_level_skill_spawns_exactly_one_random_box():
    for skill in SKILLS:
        spec = replace(MultiBoxSpec(), strategy="staged", skill=skill,
                       reset_bank="predecessor-bank" if skill in PREDECESSOR else None)
        spec.validate()
        assert spec.spawn_count_range == (1, 1)
    with pytest.raises(ValueError, match="exactly one"):
        replace(MultiBoxSpec(), low_level_spawn_count=2).validate()
    with pytest.raises(ValueError, match="inclusive range"):
        replace(MultiBoxSpec(), full_spawn_count_range=(0, 13)).validate()


def test_v2_skills_exclude_approach_pick_and_extract():
    spec = MultiBoxSpec()
    assert SKILLS == ("grasp", "carry", "place")
    assert PREDECESSOR == {"carry": "grasp", "place": "carry"}
    for skill in ("carry", "place"):
        with pytest.raises(ValueError, match="reset bank"):
            replace(spec, strategy="staged", skill=skill).validate()
    for removed in ("approach", "pick", "extract"):
        with pytest.raises(ValueError, match="Unknown skill"):
            replace(spec, strategy="staged", skill=removed).validate()


def test_v2_contract_rejects_other_shelves_box_types_and_one_arm_control():
    invalid_regions = DEFAULT_RACK_REGIONS[:-1] + (
        RackRegionSpec("shelf_3_left", 3, "left", ("small", "medium")),)
    with pytest.raises(ValueError, match="Rack regions must be"):
        replace(MultiBoxSpec(), rack_regions=invalid_regions).validate()
    with pytest.raises(ValueError, match="shelves 2 and 3"):
        RackRegionSpec("shelf_1_left", 1, "left", ("small",)).validate()
    with pytest.raises(ValueError, match="all-joints"):
        replace(MultiBoxSpec(), action_space="right-arm").validate()
    with pytest.raises(ValueError, match="jitter"):
        replace(MultiBoxSpec(), rack_xy_jitter=(0.1, -0.1)).validate()


def test_v2_structure_imports_without_simulator_side_effects():
    from kuavo_isaaclab_scene.rl.multi_box import geometry, hierarchy, reset_bank, scene, skills, state

    assert skills.SKILLS == SKILLS
    assert all(module.__name__.startswith("kuavo_isaaclab_scene.rl.multi_box.")
               for module in (geometry, hierarchy, reset_bank, scene, skills, state))
