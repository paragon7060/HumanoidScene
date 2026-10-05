"""A non-vanishing TRAIN recovery signal must respect physical eligibility."""
import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.jaw_saturation import (
    jaw_saturation_config, jaw_saturation_penalty,
)


def test_recovery_gradient_when_the_binary_policy_gradient_is_saturated():
    logits = torch.tensor([[-34., 34.], [-3., 3.], [-34., 34.]], requires_grad=True)
    near = torch.tensor([[True, True], [True, True], [False, False]])
    # Even a predicted closing advantage gives virtually no local signal.
    q_gradient = torch.autograd.grad(-logits[0, 0].sigmoid(), logits, retain_graph=True)[0]
    assert q_gradient[0, 0].abs() < 1e-12
    loss, metrics = jaw_saturation_penalty(logits, near, jaw_saturation_config('logit4-soft'))
    gradient = torch.autograd.grad(loss, logits)[0]
    assert gradient[0, 0] < -1e-5 and gradient[0, 1] > 1e-5
    assert gradient[1:].eq(0).all()
    assert metrics['jaw_saturation_active_hands'] == 4
    assert metrics['jaw_saturation_saturated_hands'] == 2
    assert metrics['jaw_active_abs_logit_max'] == 34.
    # The helper never clips logits or prescribes their action sign.
    assert torch.equal(logits.detach(), torch.tensor([[-34., 34.], [-3., 3.], [-34., 34.]]))


def test_far_hands_cannot_create_a_penalty_or_gradient():
    logits = torch.tensor([[-50., 50.]], requires_grad=True)
    loss, metrics = jaw_saturation_penalty(logits, torch.zeros_like(logits, dtype=torch.bool),
        jaw_saturation_config('logit4-soft'))
    assert loss == 0 and torch.autograd.grad(loss, logits)[0].eq(0).all()
    assert metrics['jaw_saturation_active_hands'] == metrics['jaw_active_abs_logit_max'] == 0


def test_profile_and_corrupt_logits_do_not_silently_change_the_actor_objective():
    assert jaw_saturation_config(None) is jaw_saturation_config('off') is None
    with pytest.raises(ValueError, match='Unknown'):
        jaw_saturation_config('unknown')
    config = jaw_saturation_config('logit4-soft')
    with pytest.raises(ValueError, match='configuration'):
        jaw_saturation_penalty(torch.zeros(1, 2), torch.ones(1, 2, dtype=torch.bool), config|{'weight': 1.})
    with pytest.raises(ValueError, match='finite'):
        jaw_saturation_penalty(torch.tensor([[float('nan'), 0.]]),
            torch.zeros(1, 2, dtype=torch.bool), config)
