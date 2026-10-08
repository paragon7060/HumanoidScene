"""New coordinate/decoder contracts; synthetic states are not grasp evidence."""
from copy import deepcopy

import pytest
import torch

from kuavo_isaaclab_scene.rl.multi_box.experiments.urdf_regional_goal_sac import (
    SOURCE_FORMAT, URDFRegionalGoalSACPilot, convert_original_body_goals,
    frozen_regional_source, original_regional_body_goal, urdf_goal_coordinates,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import RegionalActorMemorySACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_critic import BodyServoCriticEncoder
from kuavo_isaaclab_scene.rl.multi_box.experiments.tensor_arm_kinematics import TensorArmKinematics
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from test_rl_regional_actor_servo import agent, tokens, install_regional_actor
from test_rl_servo_critic import decoder_fixture


def snapshot():
    a = agent()
    install_regional_actor(a)
    return dict(format=SOURCE_FORMAT, source_artifact_type=RegionalActorMemorySACPilot.artifact_type,
        source_checkpoint_SHA256='a' * 64, source_actor_updates=5,
        source_goal_contract=dict(fixed_prior_radius=.30), source_hidden=16,
        source_frozen_body_anchor_SHA256='b' * 64,
        source_Q_replay_reward_entropy_and_optimizers_imported=False,
        model={k: v.detach().clone() for k, v in a.state_dict().items()
            if k.startswith(('actor.', 'actor_normalizer.'))})


def test_known_URDF_targets_keep_non_arm_normalization_and_original_dispatch():
    center, scale = torch.arange(21).float() / 100, torch.full((21,), .1)
    c, s = urdf_goal_coordinates(center, scale)
    k = TensorArmKinematics()
    assert torch.allclose(c[1:15] - s[1:15], k.lower.flatten() + .01, atol=2e-7)
    assert torch.allclose(c[1:15] + s[1:15], k.upper.flatten() - .01, atol=2e-7)
    non_arm = [0, *range(15, 21)]
    assert torch.equal(c[non_arm], center[non_arm]) and torch.equal(s[non_arm], scale[non_arm])
    assert staged_policy_class(URDFRegionalGoalSACPilot.artifact_type) is URDFRegionalGoalSACPilot
    assert staged_policy_class(RegionalActorMemorySACPilot.artifact_type) is RegionalActorMemorySACPilot


def test_new_goals_decode_to_same_real_servo_commands_with_pending_targets():
    raw, old = decoder_fixture(16)
    center, scale = urdf_goal_coordinates(old.center, old.scale)
    old.center[1:15] = center[1:15]
    new = BodyServoCriticEncoder(old.coordinates, center, scale)
    torch.manual_seed(73)
    goals = torch.rand(16, 21) * 1.6 - .8
    goals[:, 19:] = torch.tensor([1., -1.])
    body = convert_original_body_goals(goals[:, :19], old.center, old.scale, center, scale)
    converted = torch.cat((body, goals[:, 19:]), -1)
    assert converted.abs().max() <= 1
    assert torch.allclose(new(raw, converted), old(raw, goals), atol=2e-5, rtol=1e-5)
    assert torch.equal(body[:, [0, 15, 16, 17, 18]], goals[:, [0, 15, 16, 17, 18]])


def test_new_actor_can_request_more_arm_range_while_command_caps_stay_bounded():
    raw, old = decoder_fixture(1)
    center, scale = urdf_goal_coordinates(old.center, old.scale)
    old.center[1:15] = center[1:15]
    anchor = convert_original_body_goals(torch.zeros(1, 19), old.center, old.scale, center, scale)
    candidate = anchor.clone(); candidate[:, 1] += .3
    physical = center[:19] + scale[:19] * candidate
    assert physical[0, 1] > old.center[1] + .3 * old.scale[1]
    new = BodyServoCriticEncoder(old.coordinates, center, scale)
    commands = new(raw, torch.cat((candidate, raw.new_tensor([[1., -1.]])), -1))
    assert torch.isfinite(commands).all() and commands.abs().max() <= 1


def test_source_body_mapping_stays_original_and_frozen_during_actual_new_SAC_update():
    packet = snapshot(); model = frozen_regional_source(packet, 'cpu')
    raw, _ = tokens(8)
    nominal = torch.full((8, 19), .95)
    old = original_regional_body_goal(raw, nominal, packet, model)
    mean = model['actor'].network(model['actor_normalizer'](raw)).chunk(2, -1)[0][:, :19]
    assert torch.allclose(old, nominal + .05 * mean.tanh())
    a = agent(); install_regional_actor(a)
    _, encoder = decoder_fixture(8)
    center, scale = urdf_goal_coordinates(encoder.center, encoder.scale)
    a.goal_servo_critic_encoder = BodyServoCriticEncoder(encoder.coordinates, center, scale)
    a.executed_body_anchor = lambda r: r.new_zeros(len(r), 19)
    action = a.act(raw, True)
    before = deepcopy(model.state_dict())
    result = a.update(dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=action,
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8, 4),
        reward=torch.zeros(8), terminated=torch.zeros(8, dtype=torch.bool)))
    assert result['actor_updated'] and all(torch.isfinite(v).all() for v in a.state_dict().values())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())


@pytest.mark.parametrize('fault', ['Q', 'nan', 'routing', 'permission'])
def test_frozen_source_rejects_Q_nonfinite_or_mismatched_routing(fault):
    packet = snapshot()
    if fault == 'Q': packet['model']['q1.fake'] = torch.zeros(1)
    elif fault == 'nan': packet['model']['actor.network.heads.0.0.bias'][0] = torch.nan
    elif fault == 'routing': packet['model']['actor.network.region_scale'][0] += 1
    else: packet['source_Q_replay_reward_entropy_and_optimizers_imported'] = True
    with pytest.raises(ValueError): frozen_regional_source(packet, 'cpu')
