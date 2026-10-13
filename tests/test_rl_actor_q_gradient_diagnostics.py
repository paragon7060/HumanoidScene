"""Real servo clipping and optimizer/RNG invariance; no physics success proof."""
from copy import deepcopy

import pytest
import torch

from test_rl_servo_critic import decoder_fixture, make_agent
from test_rl_region_size_actor import setup
from kuavo_isaaclab_scene.rl.algorithms.actor_q_gradient_diagnostics import ActorBodyQGradientDiagnostics
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_size_actor import RegionSizeRoutedNetwork, SCOPES
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256


def fixture():
    norm, source, tokens = setup()
    raw, encoder = decoder_fixture(len(tokens))
    raw[:, 89:98] = tokens[:, 89:98]
    agent = make_agent(encoder)
    agent.actor_normalizer = norm
    agent.actor.network = RegionSizeRoutedNetwork(source, norm)
    agent.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=agent.config.lr)
    return agent, raw, encoder


def model_and_optimizer_digest(agent):
    state = agent.checkpoint()
    return structure_sha256(dict(model=state['model'], optimizers=state['optimizers']))


def test_real_decoder_separates_exact_clipped_zero_from_nonzero_Q_signal_without_mutation():
    agent, raw, encoder = fixture()
    with torch.no_grad():
        for i, head in enumerate(agent.actor.network.heads):
            head[-1].weight.zero_()
            head[-1].bias.zero_()
            head[-1].bias[1] = torch.atanh(torch.tensor(.5 if i == 0 else .05))
    before, rng, original = model_and_optimizer_digest(agent), torch.get_rng_state().clone(), raw.clone()
    body = agent.actor.network(agent.actor_normalizer(raw))[:, :19].tanh()
    goals = torch.cat((body, torch.ones(len(raw), 2)), -1)
    q = encoder(raw, goals)[:, 1, None].expand(-1, 4)
    report = ActorBodyQGradientDiagnostics(64).capture(agent, raw, body, q, torch.full_like(q, .25), 1.)
    clipped = report['scopes']['/'.join(SCOPES[0])]
    active = report['scopes']['/'.join(SCOPES[1])]
    assert clipped['rows'] == 2 and clipped['left_arm']['exact_zero_rows'] == 2
    assert clipped['mean_body_Q_objective_parameter_gradient_L2'] == 0
    assert active['left_arm']['exact_zero_rows'] == 0
    assert active['left_arm']['mean_row_gradient_L2'] == pytest.approx(5.)
    assert active['right_arm']['exact_zero_rows'] == 2
    assert active['mean_body_Q_objective_parameter_gradient_L2'] > 0
    assert all(p.grad is None for p in agent.actor.parameters())
    assert before == model_and_optimizer_digest(agent)
    assert torch.equal(rng, torch.get_rng_state()) and torch.equal(raw, original)
    q.mean().backward()  # The original optimization graph remains usable.
    assert any(p.grad is not None for p in agent.actor.parameters())


def test_actual_hybrid_update_with_diagnostics_keeps_models_all_Adam_entropy_and_RNG_exact():
    torch.manual_seed(910)
    plain, raw, _ = fixture()
    observed = deepcopy(plain)
    observed.actor_body_Q_gradient_diagnostics = ActorBodyQGradientDiagnostics(3)
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(len(raw), 4), action=plain.act(raw, True),
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(len(raw), 4),
        reward=torch.linspace(-1, 1, len(raw)), terminated=torch.ones(len(raw), dtype=torch.bool))
    before = structure_sha256(batch)
    for draw in range(2):
        rng = torch.get_rng_state().clone()
        off = plain.update(batch)
        after = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        on = observed.update(batch)
        assert torch.equal(after, torch.get_rng_state())
        assert model_and_optimizer_digest(plain) == model_and_optimizer_digest(observed)
        assert vars(plain.config) == vars(observed.config)
        measurement = on.pop('actor_body_Q_gradient_diagnostics', None)
        assert measurement['sampled_on_this_proposal'] == (draw == 0)
        assert measurement['measurements'] == 1
        assert measurement['latest_measurement']['proposal_attempt'] == 1
        assert on == off and structure_sha256(batch) == before
    off = plain.update(batch, update_actor=False)
    on = observed.update(batch, update_actor=False)
    status = on.pop('actor_body_Q_gradient_diagnostics')
    assert not status['actor_proposal_in_this_update'] and not status['sampled_on_this_proposal']
    assert status['proposal_attempts'] == 2 and status['measurements'] == 1
    assert on == off and model_and_optimizer_digest(plain) == model_and_optimizer_digest(observed)


def test_body_goal_signal_does_not_imply_actor_parameter_signal_through_a_fixed_mapping():
    agent, raw, encoder = fixture()
    body = agent.actor.network(agent.actor_normalizer(raw))[:, :19].tanh() * 0
    q = encoder(raw, torch.cat((body, torch.ones(len(raw), 2)), -1))[:, 1, None].expand(-1, 4)
    report = ActorBodyQGradientDiagnostics(1).capture(agent, raw, body, q, torch.full_like(q, .25), 1.)
    for fields in report['scopes'].values():
        assert fields['left_arm']['mean_row_gradient_L2'] == pytest.approx(5.)
        assert fields['mean_body_Q_objective_parameter_gradient_L2'] == 0


def test_opposing_retention_gradient_is_distinguished_without_mutating_original_backward():
    agent, raw, encoder = fixture()
    body = agent.actor.network(agent.actor_normalizer(raw))[:, :19].tanh() * .01
    q = encoder(raw, torch.cat((body, torch.ones(len(raw), 2)), -1))[:, 1, None].expand(-1, 4)
    probabilities = torch.full_like(q, .25)
    objective = -(probabilities*q).sum(-1).mean()
    retention = -3*objective
    total = objective+retention
    before, rng = model_and_optimizer_digest(agent), torch.get_rng_state().clone()
    diagnostic = ActorBodyQGradientDiagnostics(64)
    diagnostic.capture(agent, raw, body, q, probabilities, 1.)
    measured = diagnostic.complete_actor_objective(agent, total, retention, successful_train_rows=12)
    assert measured['successful_TRAIN_actor_rows'] == 12
    for fields in measured['scopes'].values():
        assert fields['nonfinite_gradient_elements'] == 0
        if fields['body_Q_parameter_gradient_L2']:
            assert fields['cosine_body_Q_and_total_actor_gradients'] == pytest.approx(-1.)
            assert fields['cosine_body_Q_and_success_retention_gradients'] == pytest.approx(-1.)
        else:
            assert fields['cosine_body_Q_and_total_actor_gradients'] is None
    assert diagnostic._pending is None
    assert all(p.grad is None for p in agent.actor.parameters())
    assert before == model_and_optimizer_digest(agent) and torch.equal(rng, torch.get_rng_state())
    total.backward()
    assert any(p.grad is not None for p in agent.actor.parameters())


def test_actual_success_retention_update_is_exact_with_sampled_objective_alignment():
    torch.manual_seed(957)
    plain, raw, _ = fixture()
    observed = deepcopy(plain)
    observed.actor_body_Q_gradient_diagnostics = ActorBodyQGradientDiagnostics(2)
    actions = plain.act(raw, True)
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(len(raw), 4), action=actions,
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(len(raw), 4),
        reward=torch.linspace(-1, 1, len(raw)), terminated=torch.ones(len(raw), dtype=torch.bool))
    success = dict(actor_obs=raw.clone(), action=actions.clone())
    inputs = structure_sha256((batch, success))
    for draw in range(2):
        rng = torch.get_rng_state().clone()
        off = plain.update(batch, successful_train=success, success_goal_weight=.3, success_jaw_weight=.2)
        end = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        on = observed.update(batch, successful_train=success, success_goal_weight=.3, success_jaw_weight=.2)
        assert torch.equal(end, torch.get_rng_state())
        diagnostic = on.pop('actor_body_Q_gradient_diagnostics')
        assert on == off and model_and_optimizer_digest(plain) == model_and_optimizer_digest(observed)
        assert structure_sha256((batch, success)) == inputs
        measurement = diagnostic['latest_measurement']['actor_objective_gradient_alignment']
        assert measurement['proposal_attempt'] == 1 and measurement['successful_TRAIN_actor_rows'] == len(raw)
        assert diagnostic['sampled_on_this_proposal'] == (draw == 0)
        assert observed.actor_body_Q_gradient_diagnostics._pending is None


def test_same_sample_regularization_direction_is_separate_from_Q_and_retention():
    agent, raw, encoder = fixture()
    body = agent.actor.network(agent.actor_normalizer(raw))[:, :19].tanh() * .01
    q = encoder(raw, torch.cat((body, torch.ones(len(raw), 2)), -1))[:, 1, None].expand(-1, 4)
    probability = torch.full_like(q, .25)
    q_loss = -(probability*q).sum(-1).mean()
    retention, cost = 2*q_loss, -3*q_loss
    total = q_loss+retention+cost
    original, rng = model_and_optimizer_digest(agent), torch.get_rng_state().clone()
    diagnostic = ActorBodyQGradientDiagnostics(1)
    diagnostic.capture(agent, raw, body, q, probability, 1.)
    diagnostic.complete_actor_objective(agent, total, retention,
        successful_train_rows=12, sampled_regularization_loss=cost)
    measured = diagnostic.latest['sampled_actor_regularization_gradient_alignment']
    for fields in measured['scopes'].values():
        assert fields['nonfinite_gradient_elements'] == 0
        if fields['body_Q_parameter_gradient_L2']:
            assert fields['weighted_sampled_regularization_parameter_gradient_L2'] == pytest.approx(
                3*fields['body_Q_parameter_gradient_L2'])
            assert fields['cosine_body_Q_and_sampled_regularization_gradients'] == pytest.approx(-1.)
            assert fields['cosine_success_retention_and_sampled_regularization_gradients'] == pytest.approx(-1.)
        else:
            assert fields['cosine_body_Q_and_sampled_regularization_gradients'] is None
    assert original == model_and_optimizer_digest(agent) and torch.equal(rng, torch.get_rng_state())
    assert all(p.grad is None for p in agent.actor.parameters())
    total.backward()  # Additional inspection has not consumed the actor graph.


def test_actual_clearance_regularization_diagnostics_preserve_six_head_updates_and_sampling():
    from test_rl_actor_rack_clearance_objective import SyntheticGeometry, configuration
    from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_rack_clearance import ActorRackClearanceObjective
    plain, raw, encoder = fixture()
    with torch.no_grad():
        for head in plain.actor.network.heads:
            head[-1].weight.zero_()
            head[-1].bias[:21].zero_()
    objective = ActorRackClearanceObjective(encoder, configuration(), raw)
    objective.geometry = SyntheticGeometry(objective.kinematics.columns[0][0])
    plain.actor_rack_clearance_config = configuration()
    plain._actor_rack_clearance_objective = objective
    observed = deepcopy(plain)
    observed.actor_body_Q_gradient_diagnostics = ActorBodyQGradientDiagnostics(2)
    actions = plain.act(raw, True)
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(len(raw), 4), action=actions,
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(len(raw), 4),
        reward=torch.linspace(-1, 1, len(raw)), terminated=torch.ones(len(raw), dtype=torch.bool))
    success = dict(actor_obs=raw.clone(), action=actions.clone())
    inputs = structure_sha256((batch, success))
    for draw in range(2):
        rng = torch.get_rng_state().clone()
        off = plain.update(batch, successful_train=success, success_goal_weight=.3, success_jaw_weight=.2)
        end = torch.get_rng_state().clone()
        torch.set_rng_state(rng)
        on = observed.update(batch, successful_train=success, success_goal_weight=.3, success_jaw_weight=.2)
        diagnostic = on.pop('actor_body_Q_gradient_diagnostics')
        assert on == off and model_and_optimizer_digest(plain) == model_and_optimizer_digest(observed)
        assert inputs == structure_sha256((batch, success)) and torch.equal(end, torch.get_rng_state())
        assert diagnostic['sampled_on_this_proposal'] == (draw == 0)
        assert on['actor_rack_clearance']['positive_cost_rows'] > 0
        measurement = diagnostic['latest_measurement']['sampled_actor_regularization_gradient_alignment']
        assert set(measurement['scopes']) == {'/'.join(scope) for scope in SCOPES}
        assert all(fields['nonfinite_gradient_elements'] == 0 for fields in measurement['scopes'].values())
        assert any(fields['weighted_sampled_regularization_parameter_gradient_L2'] > 0
                   for fields in measurement['scopes'].values())
        assert observed.actor_body_Q_gradient_diagnostics._pending is None


@pytest.mark.parametrize('interval', [0, -1, True, 1.5])
def test_invalid_sampling_intervals_are_rejected(interval):
    with pytest.raises(ValueError):
        ActorBodyQGradientDiagnostics(interval)
