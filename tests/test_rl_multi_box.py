"""Task semantics regression tests, CPU only; never launch Isaac Sim."""
from dataclasses import replace
from types import SimpleNamespace
import pytest
import torch
from kuavo_isaaclab_scene.rl.multi_box.spec import MultiBoxSpec, validate_shelves
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


def test_specs_and_exact_shelf_assignments():
    spec = MultiBoxSpec(); spec.validate()
    for skill in ("extract", "carry", "place"):
        with pytest.raises(ValueError, match="reset bank"):
            replace(spec, strategy="staged", skill=skill).validate()
    plans = {n: SimpleNamespace(scene_key=n, shelf=s, on_rack=True) for n, s in zip(spec.box_names, spec.shelves)}
    validate_shelves(plans, spec)
    plans[spec.box_names[0]].shelf = 3
    with pytest.raises(ValueError, match="shelf 1"):
        validate_shelves(plans, spec)
