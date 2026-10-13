"""Same-sample learning/checkpoint boundaries; no physical success evidence."""
from copy import deepcopy

import pytest
import torch

from test_rl_servo_critic import decoder_fixture, make_agent
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_rack_clearance import (
    ActorRackClearanceObjective, actor_clearance_config, explicit_actor_clearance_config,
    validate_actor_clearance_config,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256


def configuration():
    # Test-only values, not a selected production margin/weight.
    return actor_clearance_config(weight=.2, activation_m=.003, scale_m=.01)


class SyntheticGeometry:
    """Controlled geometry derivative passed through the real clipped servo."""
    def __init__(self, column, *, inactive=False):
        self.column = column
        self.inactive = inactive
        self.calls = 0

    def differentiable_distances(self, q, rack):
        self.calls += 1
        nearest = (.02 if self.inactive else .001) - .005*q[:, self.column]
        return torch.cat((nearest[:, None], q.new_full((len(q), 3), .1)), -1)


def setup(n=4, *, inactive=False):
    raw, encoder = decoder_fixture(n)
    agent = make_agent(encoder)
    with torch.no_grad():
        agent.actor.network[-1].weight.zero_()
        agent.actor.network[-1].bias[:21].zero_()
    config = configuration()
    objective = ActorRackClearanceObjective(encoder, config, raw)
    objective.geometry = SyntheticGeometry(objective.kinematics.columns[0][0], inactive=inactive)
    agent.actor_rack_clearance_config = config
    agent._actor_rack_clearance_objective = objective
    return agent, raw, objective


def batch_for(agent, raw):
    return dict(actor_obs=raw, critic_obs=raw.new_zeros(len(raw), 4),
        action=agent.act(raw, True), reward=raw.new_ones(len(raw)),
        next_actor_obs=raw.clone(), next_critic_obs=raw.new_zeros(len(raw), 4),
        terminated=torch.ones(len(raw), dtype=torch.bool))


def test_off_and_zero_exposure_same_sample_updates_preserve_all_optimizers_and_RNG():
    enabled, raw, geometry = setup(inactive=True)
    disabled = deepcopy(enabled)
    disabled.actor_rack_clearance_config = None
    disabled._actor_rack_clearance_objective = None
    batch = batch_for(disabled, raw)
    before = structure_sha256(batch)
    rng = torch.get_rng_state().clone()
    off = disabled.update(batch)
    after = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    on = enabled.update(batch)
    assert torch.equal(after, torch.get_rng_state()) and structure_sha256(batch) == before
    stat = on.pop('actor_rack_clearance')
    assert stat['rows'] == len(raw) and stat['positive_cost_rows'] == 0
    assert stat['weighted_loss'] == 0 and stat['extra_policy_samples'] == 0
    assert on == off
    for key in ('model', 'optimizers'):
        assert structure_sha256(enabled.checkpoint()[key]) == structure_sha256(disabled.checkpoint()[key])
    assert geometry.geometry.calls == 1


def test_nonzero_cost_uses_existing_sample_and_real_decoder_gradient_without_input_mutation():
    agent, raw, objective = setup()
    goals = torch.zeros(len(raw), 4, 21)
    goals[:, :, 19:] = -1
    goals.requires_grad_(True)
    original, rng = raw.clone(), torch.get_rng_state().clone()
    loss, stats = objective(raw, goals)
    assert loss > 0 and stats['actor_rack_clearance']['positive_cost_rows'] == len(raw)
    gradient = torch.autograd.grad(loss, goals)[0]
    assert gradient[:, 0, 1].abs().gt(0).all()
    assert gradient[:, 1:].eq(0).all() and gradient[:, :, [0, *range(2, 21)]].eq(0).all()
    # Servo coordinates have signed joint scales; verify the decoded physical
    # direction rather than assuming increasing a normalized goal raises q.
    lo, hi = goals.detach().clone(), goals.detach().clone()
    lo[:, :, 1] -= 1e-4; hi[:, :, 1] += 1e-4
    finite_difference = (objective(raw, hi)[0]-objective(raw, lo)[0]) / 2e-4
    assert gradient[:, 0, 1].sum() == pytest.approx(float(finite_difference), rel=.002)
    moved = goals.detach().clone()
    moved[:, :, 1] -= gradient[:, :1, 1]
    assert objective(raw, moved)[0] < loss
    assert torch.equal(raw, original) and torch.equal(rng, torch.get_rng_state())
    # A saturated command does not gain an invented gradient from the cost.
    goals = goals.detach(); goals[:, :, 1] = .9; goals.requires_grad_(True)
    assert torch.autograd.grad(objective(raw, goals)[0], goals)[0].eq(0).all()


def test_positive_actor_cost_changes_only_actor_proposal_not_critic_targets_or_sampling():
    enabled, raw, _ = setup()
    disabled = deepcopy(enabled)
    disabled.actor_rack_clearance_config = None
    batch = batch_for(disabled, raw)
    inputs = structure_sha256(batch)
    rng = torch.get_rng_state().clone()
    off = disabled.update(batch)
    end = torch.get_rng_state().clone()
    torch.set_rng_state(rng)
    on = enabled.update(batch)
    assert torch.equal(end, torch.get_rng_state()) and structure_sha256(batch) == inputs
    assert on['actor_rack_clearance']['weighted_loss'] > 0
    assert on['actor_loss'] > off['actor_loss'] and on['target_value_mean'] == off['target_value_mean']
    for prefix in ('q1.', 'q2.', 'target1.', 'target2.'):
        a = {k:v for k,v in enabled.state_dict().items() if k.startswith(prefix)}
        b = {k:v for k,v in disabled.state_dict().items() if k.startswith(prefix)}
        assert structure_sha256(a) == structure_sha256(b)
    assert structure_sha256(enabled.optimizers[1].state_dict()) == structure_sha256(disabled.optimizers[1].state_dict())
    assert not torch.equal(enabled.actor.network[-1].bias, disabled.actor.network[-1].bias)


def test_critic_only_and_frozen_policy_never_evaluate_geometry_and_restore_config_without_graph():
    agent, raw, objective = setup()
    report = agent.update(batch_for(agent, raw), update_actor=False)
    assert objective.geometry.calls == 0 and 'actor_rack_clearance' not in report
    saved = deepcopy(agent.checkpoint())
    digest = structure_sha256({k:v for k,v in saved.items() if k != 'config'})
    original_config = deepcopy(saved['config'])
    restored = make_agent(agent.goal_servo_critic_encoder)
    restored.restore(saved, training=False)
    assert restored.actor_rack_clearance_config == configuration()
    assert restored._actor_rack_clearance_objective is None
    assert structure_sha256({k:v for k,v in saved.items() if k != 'config'}) == digest
    assert saved['config'] == original_config
    assert torch.equal(agent.act(raw, True), restored.act(raw, True))
    assert all(not opt.state for opt in restored.optimizers)
    # Plain checkpoints remove a previously restored auxiliary configuration.
    plain = deepcopy(saved); plain.pop('actor_rack_clearance_config')
    restored.restore(plain, training=False)
    assert restored.actor_rack_clearance_config is None
    assert 'actor_rack_clearance_config' not in restored.checkpoint()


def test_chunking_keeps_row_mean_and_gradients_and_rejects_jaw_dependent_body():
    agent, raw, objective = setup(65)
    goals = raw.new_zeros(len(raw), 4, 21); goals[:, :, 19:] = -1; goals.requires_grad_(True)
    loss, stats = objective(raw, goals)
    assert objective.geometry.calls == 3 and stats['actor_rack_clearance']['rows'] == 65
    gradient = torch.autograd.grad(loss, goals)[0]
    assert gradient[:, 0, 1].abs().gt(0).all()
    different = goals.detach().clone(); different[:, 2, 4] = .1
    with pytest.raises(ValueError, match='share'):
        objective(raw, different)


@pytest.mark.parametrize('kwargs', [dict(weight=0), dict(weight=float('nan')), dict(weight=True),
    dict(activation_m=-.01), dict(scale_m=0)])
def test_invalid_configuration_is_rejected(kwargs):
    values = dict(weight=.2, activation_m=.003, scale_m=.01) | kwargs
    with pytest.raises(ValueError):actor_clearance_config(**values)


def test_cli_and_checkpoint_version_boundaries():
    assert explicit_actor_clearance_config() is None
    with pytest.raises(ValueError):explicit_actor_clearance_config(.2, .003, .01, training=False, has_train=True)
    with pytest.raises(ValueError):explicit_actor_clearance_config(.2, .003, None, training=True, has_train=True)
    with pytest.raises(ValueError):explicit_actor_clearance_config(.2, .003, .01, training=True, has_train=False)
    assert explicit_actor_clearance_config(.2, .003, .01, training=True, has_train=True) == configuration()
    for corrupted in (configuration() | dict(name='unknown'), configuration() | dict(chunk_rows=64),
                      configuration() | dict(geometry_SHA256='0'*64)):
        with pytest.raises(ValueError):validate_actor_clearance_config(corrupted)
    raw, encoder = decoder_fixture(1)
    with pytest.raises(ValueError):ActorRackClearanceObjective(None, configuration(), raw)
