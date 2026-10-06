from copy import deepcopy

import pytest
import torch

from test_rl_actual_flap_residual_sac import pilots
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import ActualFlapResidualSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_reanchored_sac import (
    ReanchoredActualFlapSACPilot, actual_actor_anchor_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def learned_source(tmp_path):
    _, source, warm, physical, stage, _ = pilots(tmp_path)
    with torch.no_grad():
        # Exercise nonzero learned goals and the actual 38 perception inputs,
        # rather than testing another copy of the zero-residual nominal actor.
        source.agent.actor.network[0].weight[:, 474:512].fill_(.01)
        source.agent.actor.network[-1].bias[:19].fill_(.8)
        source.agent.actor.network[-1].bias[19:21].add_(.3)
    source.directory.mkdir()
    source.save(final=True)
    checkpoint = next(source.directory.glob('checkpoint_*.pt'))
    state = torch.load(checkpoint, weights_only=True)
    new = ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'reanchored', stage,
        body_anchor_state=actual_actor_anchor_state(state), replay_capacity=1024,
        actor_min_replay_rows=64, exploration_correlation=.99,
        train_success_retention=True, measured_train_credit='measured-nstep16',
        jaw_behavior='joint-epsilon30', body_behavior='ramped-arm-bias20',
        body_saturation='mean3-soft')
    return source, new, warm, physical, stage, checkpoint


def test_reanchored_body_keeps_learned_greedy_goals_and_jaws_with_fresh_learning_state(tmp_path):
    source, new, *_ = learned_source(tmp_path)
    raw = torch.randn(32, 518)*.02
    raw[:, 144:146] = torch.tensor([1., 0.])
    raw[:, -1] = .15
    changed = raw.clone(); changed[:, -1] = .30
    expected = source.agent.act(raw, True)
    actual = new.agent.act(changed, True)
    torch.testing.assert_close(actual[:, :19], expected[:, :19], atol=0, rtol=0)
    assert torch.equal(actual[:, 19:], expected[:, 19:])
    torch.testing.assert_close(new.agent.parameters_at(new.agent.actor_normalizer(changed))[2],
        source.agent.parameters_at(source.agent.actor_normalizer(raw))[2], atol=2e-5, rtol=1e-6)
    assert new.replay.size == new.success_bank.size == new.measured_credit_bank.size == 0
    assert new.actor_updates == new.critic_updates == new.online_rows == 0
    assert new.agent.critic_normalizer.count == 0
    assert not any(opt.state for opt in new.agent.optimizers)
    assert not new.agent.parameters_at(new.agent.actor_normalizer(changed))[0].any()
    assert all(not p.requires_grad for p in new.actual_body_anchor.parameters())
    assert not any(k.startswith(('q', 'target', 'alpha')) for k in new.body_anchor_state['model'])
    assert new.contract != source.contract and source.correction_radius == .15
    assert staged_policy_class(new.artifact_type) is ReanchoredActualFlapSACPilot


def test_reanchored_sampling_moves_around_learned_goals_and_handles_inactive_bounds(tmp_path):
    source, new, *_ = learned_source(tmp_path)
    raw = torch.zeros(8, 518); raw[:, 144] = 1.; raw[:, -1] = .30
    nominal = source.executed_body_anchor(raw)
    anchor, scale = new.agent.anchor_and_scale(raw)
    assert (anchor - nominal).abs().max() > .01
    assert torch.all(scale > .15) and torch.all(scale <= .30)
    with torch.no_grad(): new.agent.actor.network[-1].bias[:19].fill_(.4)
    body, logp, logits = new.agent.continuous_sample(
        new.agent.actor_normalizer(raw), deterministic=True, raw=raw)
    torch.testing.assert_close(body, anchor + scale*torch.tanh(torch.full_like(anchor, .4)))
    branches, *_ = new.agent.enumerate_jaws(raw, body, logits)
    torch.testing.assert_close(branches[:, :, :19], body[:, None].expand(-1, 4, -1))
    labels = new.agent.act(raw, True)
    assert new.agent.success_body_loss(raw, torch.tanh(torch.full_like(anchor, .4)), labels) == 0
    assert torch.isfinite(logp).all() and (body.abs() <= 1).all()
    # No clipped point mass and no entropy/gradient for physical fixed goals.
    new.agent.executed_body_anchor = lambda obs: obs.new_ones(len(obs), 19)
    fixed, logp, _ = new.agent.continuous_sample(new.agent.actor_normalizer(raw), raw=raw)
    assert fixed.eq(1).all() and logp.eq(0).all()
    assert new.agent.body_entropy_target(None, torch.ones(8,19), raw).eq(0).all()


def test_reanchored_resume_restores_new_replay_models_and_all_four_optimizers(tmp_path):
    _, new, warm, physical, stage, _ = learned_source(tmp_path)
    raw = torch.zeros(64, 464); raw[:, 144] = 1.
    critic = torch.zeros(64, 530); extra = torch.zeros(64, 38)
    previous = new.act(raw, critic, 0, supplemental=extra)[1]
    new.observe(previous, raw, critic, torch.ones(64), torch.ones(64, dtype=torch.bool),
        0, supplemental=extra)
    batch = new.replay.sample(64, 'cpu')
    result = new.agent.update(batch)
    assert result['actor_updated'] and all(opt.state for opt in new.agent.optimizers)
    new.directory.mkdir(); new.save(final=True)
    checkpoint = next(new.directory.glob('checkpoint_*.pt'))
    restored = ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'restore', stage, checkpoint=checkpoint)
    assert restored.replay.size == 64 and restored.contract == new.contract
    for k, v in new.agent.state_dict().items(): assert torch.equal(v, restored.agent.state_dict()[k])
    for old, current in zip(new.agent.optimizers, restored.agent.optimizers):
        a, b = old.state_dict(), current.state_dict()
        assert a['param_groups'] == b['param_groups'] and a['state'].keys() == b['state'].keys()
        for key in a['state']:
            for field, value in a['state'][key].items():
                if isinstance(value, torch.Tensor): assert torch.equal(value, b['state'][key][field])
                else: assert value == b['state'][key][field]
    restored.anchor = new.anchor
    assert torch.equal(new.agent.act(previous[0], True), restored.agent.act(previous[0], True))
    experience = checkpoint.parent/'staged_goal_experience.pt'
    state = torch.load(experience, weights_only=True)
    state['goal_contract']['body_correction_radius'] = .15
    torch.save(state, experience)
    with pytest.raises(ValueError, match='replay context'):
        ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'mixed', stage, checkpoint=checkpoint)


def test_old_controller_resume_and_contaminated_anchor_are_rejected(tmp_path):
    source, _, warm, physical, stage, checkpoint = learned_source(tmp_path)
    with pytest.raises(ValueError, match='Old observation/control'):
        ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'old_resume', stage, checkpoint=checkpoint)
    with pytest.raises(ValueError, match='Correction controller'):
        ActualFlapResidualSACPilot(warm, physical, tmp_path/'old_radius', stage,
            body_anchor_state=source.body_anchor_state, fixed_prior_radius=.30)
    snapshot = actual_actor_anchor_state(torch.load(checkpoint, weights_only=True))
    contaminated = deepcopy(snapshot)
    contaminated['model']['q1.fake'] = torch.zeros(1)
    with pytest.raises(ValueError, match='actor-only snapshot'):
        ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'bad_anchor', stage, body_anchor_state=contaminated)
    snapshot['source_goal_contract']['body_correction_radius'] = .30
    with pytest.raises(ValueError, match='Reanchored source'):
        ReanchoredActualFlapSACPilot(warm, physical, tmp_path/'wrong_source', stage, body_anchor_state=snapshot)
