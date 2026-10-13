"""Geometry derivatives and real decoder clipping; not physical rollout tests."""
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_rack_clearance import (
    decoded_next_arm_targets, soft_rack_clearance_rows,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.rack_entry_clearance import RackEntryClearance
from kuavo_isaaclab_scene.rl.multi_box.experiments.tensor_arm_kinematics import TensorArmKinematics
from test_rl_cartesian_flap_probe import rotation6
from test_rl_servo_critic import decoder_fixture


def fixture():
    kin = TensorArmKinematics(dtype=torch.float64)
    geometry = RackEntryClearance(kin)
    q = torch.zeros(2, 20, dtype=torch.float64)
    q[:, :3] = torch.tensor([-.2, .3, -.1])
    for h, columns in enumerate(kin.columns):
        q[:, columns] = (kin.lower[h] + kin.upper[h]) / 2
    q[1, kin.columns[0]] += .013
    rack = q.new_zeros(2, 9)
    rack[:, :3] = q.new_tensor([.17, -.43, .11])
    rack[:, 3:] = rotation6(torch.eye(3, dtype=q.dtype).expand(2, 3, 3))
    return kin, geometry, q, rack


def test_geometry_forward_matches_existing_proxies_and_gradients_without_mutation():
    kin, geometry, q, rack = fixture()
    rng = torch.get_rng_state().clone()
    measured = geometry.measure(q, rack)
    differentiable = q.clone().requires_grad_(True)
    distances = geometry.differentiable_distances(differentiable, rack)
    assert torch.equal(distances.detach(), measured['distances'])
    for body in range(4):
        gradient = torch.autograd.grad(distances[:, body].sum(), differentiable,
                                       retain_graph=True)[0]
        assert torch.allclose(gradient[:, sum(kin.columns, [])],
                              measured['gradients'][:, body], atol=1e-10, rtol=1e-8)
        for col in (0, 1, 2, kin.columns[body % 2][0]):
            lo, hi = q.clone(), q.clone()
            lo[:, col] -= 1e-6
            hi[:, col] += 1e-6
            finite_difference = (geometry.differentiable_distances(hi, rack)[:, body] -
                                 geometry.differentiable_distances(lo, rack)[:, body]) / 2e-6
            assert torch.allclose(gradient[:, col], finite_difference, atol=3e-7, rtol=1e-5)
    assert differentiable.grad is None and torch.equal(rng, torch.get_rng_state())
    assert geometry.guided_rows == 0 and geometry.active_body_rows == 0


def test_cost_is_soft_per_row_worst_body_and_does_not_require_ten_mm():
    distances = torch.tensor([[.006789, .03, .02, .02], [.001, .004, .005, .006],
                              [-.002, .004, .005, .006]], dtype=torch.float64, requires_grad=True)
    cost = soft_rack_clearance_rows(distances, activation_m=.003, scale_m=.01)
    assert torch.allclose(cost, torch.tensor([0., .04, .25], dtype=cost.dtype))
    gradient = torch.autograd.grad(cost.sum(), distances)[0]
    assert gradient[0].eq(0).all() and gradient[:, 1:].eq(0).all()
    assert gradient[1, 0] < 0 and gradient[2, 0] < gradient[1, 0]
    moved = distances.detach() - 1e-6 * gradient
    assert (soft_rack_clearance_rows(moved, activation_m=.003, scale_m=.01) <= cost.detach()).all()


def test_saturated_actual_servo_command_keeps_zero_goal_gradient():
    raw, encoder = decoder_fixture(1)
    # The geometric primitive cannot make an actually clipped goal sensitive.
    for requested, expect_zero in ((.5, True), (.05, False)):
        goals = torch.zeros(1, 21)
        goals[:, 19:] = -1
        goals[:, 1] = requested
        goals.requires_grad_(True)
        command = encoder(raw, goals)
        synthetic_distances = torch.cat((.001 - .001 * command[:, 1:2],
                                         command.new_full((1, 3), .1)), -1)
        cost = soft_rack_clearance_rows(synthetic_distances, activation_m=.003, scale_m=.01)
        gradient = torch.autograd.grad(cost.sum(), goals)[0]
        assert bool(gradient.eq(0).all()) == expect_zero
        if not expect_zero:
            assert gradient[0, 1] > 0 and torch.count_nonzero(gradient) == 1


def test_actual_pending_arm_targets_include_delta_and_real_bounds_without_guided_lead_cap():
    raw, encoder = decoder_fixture(1)
    kin = TensorArmKinematics()
    columns = sum(kin.columns, [])
    for h, hand_columns in enumerate(kin.columns):
        raw[:, hand_columns] = (kin.lower[h] + kin.upper[h]) / 2
    # This is the compact actor's literal pending-error block, recovered by
    # servo_raw_actor, not raw464's field416:436.
    raw[:, 150:170] = .12
    encoder.center[:17] = raw[0, encoder.coordinates.joints.joint_columns] + .12
    goals = raw.new_zeros(1, 21)
    goals[:, 19:] = -1
    q = decoded_next_arm_targets(encoder, kin, raw, goals)
    assert torch.allclose(q[:, columns] - raw[:, columns], torch.full((1, 14), .12), atol=1e-7)
    assert torch.equal(q[:, [0, 1, 2, 3, 18, 19]], raw[:, [0, 1, 2, 3, 18, 19]])
    raw[:, columns] = kin.upper.flatten() - .001
    q = decoded_next_arm_targets(encoder, kin, raw, goals)
    assert (q[:, columns] <= kin.upper.flatten()).all()


def test_real_decoded_geometry_gradient_matches_finite_difference_and_saturation():
    raw, encoder = decoder_fixture(1)
    kin = TensorArmKinematics()
    geometry = RackEntryClearance(kin)
    for h, columns in enumerate(kin.columns):
        raw[:, columns] = (kin.lower[h] + kin.upper[h]) / 2
    raw[:, 150:170] = 0
    encoder.center[:17] = raw[0, encoder.coordinates.joints.joint_columns]
    raw[:, 68:71] = raw.new_tensor([.17, -.43, .11])
    raw[:, 71:77] = rotation6(torch.eye(3).unsqueeze(0))
    goals = raw.new_full((1, 21), .01)
    goals[:, 19:] = -1
    goals.requires_grad_(True)
    distances = geometry.differentiable_distances(decoded_next_arm_targets(encoder, kin, raw, goals), raw[:, 68:77])
    gradient = torch.autograd.grad(distances.sum(), goals)[0]
    assert torch.isfinite(gradient).all() and gradient[:, 1:15].abs().max() > 0
    for col in range(1, 15):
        lo, hi = goals.detach().clone(), goals.detach().clone()
        lo[:, col] -= .001
        hi[:, col] += .001
        def value(g):
            return geometry.differentiable_distances(decoded_next_arm_targets(encoder, kin, raw, g), raw[:, 68:77]).sum()
        finite_difference = (value(hi) - value(lo)) / .002
        assert torch.allclose(gradient[0, col], finite_difference, atol=3e-4, rtol=1e-2)
    assert gradient[:, [0, 15, 16, 17, 18, 19, 20]].eq(0).all()
    saturated = goals.detach().clone()
    saturated[:, 1:15] = .99
    saturated.requires_grad_(True)
    value = geometry.differentiable_distances(decoded_next_arm_targets(encoder, kin, raw, saturated), raw[:, 68:77])
    assert torch.autograd.grad(value.sum(), saturated)[0].eq(0).all()


@pytest.mark.parametrize('fault', ['nonfinite', 'width', 'activation', 'scale'])
def test_rejects_unreviewable_cost_inputs(fault):
    distances = torch.full((2, 4), .02)
    activation, scale = .003, .01
    if fault == 'nonfinite':
        distances[0, 0] = float('nan')
    elif fault == 'width':
        distances = distances[:, :3]
    elif fault == 'activation':
        activation = -.001
    else:
        scale = 0.
    with pytest.raises(ValueError):
        soft_rack_clearance_rows(distances, activation_m=activation, scale_m=scale)
