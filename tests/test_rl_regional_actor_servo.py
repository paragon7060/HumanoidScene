"""Regional actor continuity, real SAC gradients, and Adam isolation."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.algorithms.common import optimize
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import (
    RegionalActorMemorySACPilot, RegionRoutedNetwork, install_regional_actor,
    regional_actor_contract, validate_regional_actor_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetainedCorrectionSAC
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS
from test_rl_servo_critic import decoder_fixture


def agent():
    result = ServoRetainedCorrectionSAC(518, 4, 21,
        SACConfig(hidden=16, actor_lr=1e-3, entropy_backup=False,
                  freeze_actor_normalizer=True), action_projector=AbsoluteGoalJawProjector())
    result.correction_radius = .3
    result.executed_body_anchor = lambda raw: raw.new_zeros(len(raw), 19)
    return result


def tokens(n=16):
    raw, encoder = decoder_fixture(n)
    raw[:, 94:98] = torch.eye(4)[torch.arange(n) % 4]
    return raw, encoder


def test_different_component_networks_preserve_exact_full_batch_outputs_and_routing():
    a = agent(); raw, _ = tokens()
    # A rare region's standardized token is clipped by the legacy normalizer.
    a.actor_normalizer.mean[94:98] = torch.tensor([.5, .49, .009, .001])
    a.actor_normalizer.var[94:98] = torch.tensor([.25, .2499, .0089, .000999])
    normal = a.actor_normalizer(raw)
    install_regional_actor(a)
    with torch.no_grad():
        for i, head in enumerate(a.actor.network.heads):
            head[-1].bias.add_(i*.1)
        expected = torch.stack([h(normal) for h in a.actor.network.heads])[
            torch.arange(len(raw)) % 4, torch.arange(len(raw))]
        assert torch.equal(a.actor.network(normal), expected)
        assert torch.equal(a.actor.network.region_indices(normal), raw[:, 94:98].argmax(-1))
    assert staged_policy_class(RegionalActorMemorySACPilot.artifact_type) is RegionalActorMemorySACPilot


def test_absent_heads_keep_weights_and_existing_Adam_momentum_bit_identical():
    a = agent(); raw, _ = tokens(); install_regional_actor(a)
    normal = a.actor_normalizer(raw)
    optimize(a.actor_optimizer, a.actor.network(normal).square().mean(), a.actor.parameters())
    before = deepcopy(a.actor.network.state_dict())
    moments = {p:deepcopy(a.actor_optimizer.state[p]) for h in a.actor.network.heads[1:] for p in h.parameters()}
    single = raw[:4].clone(); single[:, 94:98] = 0; single[:, 94] = 1
    optimize(a.actor_optimizer, (a.actor.network(a.actor_normalizer(single))-3).square().mean(), a.actor.parameters())
    assert any(not torch.equal(v, a.actor.network.state_dict()[k]) for k,v in before.items() if k.startswith('heads.0.'))
    for index in range(1, 4):
        for k,v in before.items():
            if k.startswith(f'heads.{index}.'):
                assert torch.equal(v, a.actor.network.state_dict()[k])
        for p in a.actor.network.heads[index].parameters():
            assert p.grad is None
            assert all(torch.equal(v, a.actor_optimizer.state[p][k]) for k,v in moments[p].items())


def test_real_SAC_update_preserves_Q_when_only_actor_architecture_changes():
    ordinary, regional = agent(), agent(); regional.load_state_dict(ordinary.state_dict())
    raw, encoder = tokens(); ordinary.goal_servo_critic_encoder = regional.goal_servo_critic_encoder = encoder
    install_regional_actor(regional)
    assert torch.equal(ordinary.act(raw, True), regional.act(raw, True))
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(len(raw), 4),
        action=ordinary.act(raw, True), next_actor_obs=raw.clone(),
        next_critic_obs=torch.zeros(len(raw), 4), reward=torch.zeros(len(raw)),
        terminated=torch.zeros(len(raw), dtype=torch.bool))
    reports = []
    before = deepcopy(regional.actor.network.state_dict())
    for a in (ordinary, regional):
        torch.manual_seed(933)
        reports.append(a.update(batch))
    assert reports[0]['q_loss'] == reports[1]['q_loss']
    assert reports[0]['actor_loss'] == reports[1]['actor_loss']
    for name in ('q1', 'q2', 'target1', 'target2'):
        assert structure_sha256(getattr(ordinary, name).state_dict()) == structure_sha256(getattr(regional, name).state_dict())
    for index in range(4):
        assert any(not torch.equal(v, regional.actor.network.state_dict()[k])
            for k,v in before.items() if k.startswith(f'heads.{index}.'))
    assert all(torch.isfinite(v).all() for v in regional.state_dict().values())


def state():
    a = agent(); install_regional_actor(a)
    model = a.state_dict(); anchor = dict(frozen='same_coordinates')
    components = {r:dict(checkpoint_SHA256=str(i)*64, network_SHA256=structure_sha256(
        a.actor.network.heads[i].state_dict())) for i,r in enumerate(REGIONS)}
    return dict(actor_updates=0, model=model, body_anchor_state=anchor,
        config=dict(freeze_actor_normalizer=True, actor_feature_mode='flat'),
        goal_contract=dict(regional_actor=regional_actor_contract()),
        regional_actor_initialization=dict(kind='compatible_actor_only_region_components_v1',
            components=components, frozen_body_anchor_SHA256=structure_sha256(anchor),
            actor_normalizer_SHA256=structure_sha256({k:v for k,v in model.items() if k.startswith('actor_normalizer.')})))


@pytest.mark.parametrize('fault', ['head', 'region_order', 'anchor', 'normalizer', 'route', 'old_origin'])
def test_unstarted_checkpoint_rejects_wrong_region_or_frozen_coordinates(fault):
    s = state(); validate_regional_actor_state(s)
    if fault == 'head': s['model']['actor.network.heads.0.0.bias'].add_(.1)
    elif fault == 'region_order': s['goal_contract']['regional_actor']['regions'].reverse()
    elif fault == 'anchor': s['body_anchor_state']['frozen'] = 'different'
    elif fault == 'normalizer': s['model']['actor_normalizer.mean'][94] += .1
    elif fault == 'route': s['model']['actor.network.region_scale'][0] += .1
    else: s['actor_memory_initialization'] = dict(kind='whole_actor_fit')
    with pytest.raises(ValueError): validate_regional_actor_state(s)


def test_learned_heads_can_resume_but_routing_and_component_history_stay_fixed():
    s = state(); s['actor_updates'] = 1
    s['model']['actor.network.heads.0.0.bias'].add_(.1)
    validate_regional_actor_state(s)
    s['model']['actor.network.region_mean'][0] += .1
    with pytest.raises(ValueError): validate_regional_actor_state(s)
