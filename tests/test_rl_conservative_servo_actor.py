"""Smaller real SAC actor steps, unchanged critics, and pristine migration guards."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/rl'))
from prepare_actual_success_actor_tail import identical
from prepare_conservative_servo_actor import prepare
from test_rl_pristine_greedy_collection import inputs
from test_rl_servo_critic import decoder_fixture
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import (
    ServoRetainedCorrectionSAC, ServoRetentionGentleSACPilot,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.conservative_servo_retention import (
    ACTOR_LR, BASE_ACTOR_LR, CRITIC_ALPHA_LR, ConservativeServoRetentionSACPilot,
    conservative_actor_contract, validate_conservative_actor_state,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class


def pristine():
    state, replay = inputs()
    state['config'].update(actor_lr=BASE_ACTOR_LR, lr=CRITIC_ALPHA_LR)
    for optimizer, rate in zip(state['optimizers'], (BASE_ACTOR_LR, *([CRITIC_ALPHA_LR] * 3))):
        optimizer['param_groups'] = [dict(params=[0], lr=rate)]
    return state, replay


def test_fresh_variant_changes_only_actor_rate_and_explicit_resume_identity():
    state, replay = pristine()
    before, old_replay = deepcopy(state), deepcopy(replay)
    new, experience = prepare(state, replay)
    assert identical(state, before) and identical(replay, old_replay)
    assert identical(new['model'], state['model'])
    assert identical(new['optimizers'][1:], state['optimizers'][1:])
    assert identical(new['hybrid_contract'], state['hybrid_contract'])
    assert new['config'] == state['config'] | dict(actor_lr=ACTOR_LR)
    assert new['optimizers'][0]['param_groups'][0]['lr'] == ACTOR_LR
    assert experience['goal_contract'] == new['goal_contract']
    assert new['goal_contract'] == state['goal_contract'] | dict(
        name=ConservativeServoRetentionSACPilot.artifact_type, actor_update_step=conservative_actor_contract())
    assert staged_policy_class(new['artifact_type']) is ConservativeServoRetentionSACPilot
    for key in ('executed_goal_transitions', 'successful_train_transitions', 'measured_train_credit_bank'):
        assert identical(experience[key], replay[key])
    validate_conservative_actor_state(new)


@pytest.mark.parametrize('fault', ['actor', 'Q', 'optimizer', 'replay', 'credit', 'collection', 'rate'])
def test_conservative_initialization_rejects_learning_and_collection_contamination(fault):
    state, replay = pristine()
    if fault == 'actor': state['actor_updates'] = 1
    elif fault == 'Q': state['critic_updates'] = 1
    elif fault == 'optimizer': state['optimizers'][0]['state'] = {0: dict(step=torch.tensor(1))}
    elif fault == 'replay': replay['executed_goal_transitions']['reward'] = torch.ones(1)
    elif fault == 'credit': replay['measured_train_credit_bank']['episodes']['middle_left'] = [{}]
    elif fault == 'collection': state['body_behavior_statistics']['episodes_drawn'] = 1
    else: state['config']['actor_lr'] = 3e-4
    with pytest.raises(ValueError): prepare(state, replay)


@pytest.mark.parametrize('fault', ['profile', 'config', 'optimizer', 'artifact'])
def test_saved_conservative_rate_cannot_silently_restore_another_optimizer(fault):
    state, replay = pristine()
    new, _ = prepare(state, replay)
    if fault == 'profile': new['goal_contract']['actor_update_step'] = {}
    elif fault == 'config': new['config']['actor_lr'] = BASE_ACTOR_LR
    elif fault == 'optimizer': new['optimizers'][0]['param_groups'][0]['lr'] = BASE_ACTOR_LR
    else: new['artifact_type'] = ServoRetentionGentleSACPilot.artifact_type
    with pytest.raises(ValueError): validate_conservative_actor_state(new)


def test_real_hybrid_actor_step_is_smaller_and_critic_updates_are_identical():
    raw, encoder = decoder_fixture(8)
    agents = []
    for cls in (ServoRetentionGentleSACPilot, ConservativeServoRetentionSACPilot):
        config = object.__new__(cls).learning_config(SACConfig(hidden=16, entropy_backup=False))
        agent = ServoRetainedCorrectionSAC(518, 4, 21, config, action_projector=AbsoluteGoalJawProjector())
        agent.correction_radius = .3
        agent.executed_body_anchor = lambda obs: obs.new_zeros(len(obs), 19)
        agent.goal_servo_critic_encoder = encoder
        agents.append(agent)
    original, conservative = agents
    conservative.load_state_dict(original.state_dict())
    actor_before = deepcopy(original.actor.state_dict())
    labels = torch.zeros(8, 21); labels[:, 1] = .2; labels[:, 19:] = 1
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=original.act(raw, True),
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8, 4),
        reward=torch.zeros(8), terminated=torch.zeros(8, dtype=torch.bool))
    reports = []
    for agent in agents:
        torch.manual_seed(9307)
        reports.append(agent.update(batch, successful_train=dict(actor_obs=raw, action=labels),
            success_goal_weight=1., success_jaw_weight=.1))
    assert reports[0]['actor_loss'] == reports[1]['actor_loss']
    assert reports[0]['q_loss'] == reports[1]['q_loss']
    for prefix in ('q1', 'q2', 'target1', 'target2'):
        assert identical(getattr(original, prefix).state_dict(), getattr(conservative, prefix).state_dict())
    distances = [torch.cat([(v - actor_before[k]).double().flatten()
        for k, v in a.actor.state_dict().items()]).norm().item() for a in agents]
    assert 0 < distances[1] < distances[0]
    assert distances[1] / distances[0] == pytest.approx(.1, rel=.04)
    assert original.actor_optimizer.param_groups[0]['lr'] == BASE_ACTOR_LR
    assert conservative.actor_optimizer.param_groups[0]['lr'] == ACTOR_LR
    assert all(a.q_optimizer.param_groups[0]['lr'] == CRITIC_ALPHA_LR for a in agents)
    saved = conservative.checkpoint()
    restored = ServoRetainedCorrectionSAC(518, 4, 21, conservative.config,
        action_projector=AbsoluteGoalJawProjector())
    restored.correction_radius = .3
    restored.executed_body_anchor = conservative.executed_body_anchor
    restored.goal_servo_critic_encoder = encoder
    restored.restore(saved)
    assert restored.actor_optimizer.param_groups[0]['lr'] == ACTOR_LR
    assert identical(restored.actor_optimizer.state_dict(), conservative.actor_optimizer.state_dict())
