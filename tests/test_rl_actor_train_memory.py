"""Data isolation and actual SAC updates; fixtures are not physical successes."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/rl'))
from prepare_actor_memory_servo import prepare
from prepare_conservative_servo_actor import prepare as conservative_prepare
from prepare_actual_success_actor_tail import identical
from test_rl_conservative_servo_actor import pristine
from test_rl_staged_train_success import episode
from test_rl_servo_critic import decoder_fixture
from kuavo_isaaclab_scene.rl.algorithms.sac import SACConfig
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import (
    FORMAT, ActorTrainMemory, actor_memory_contract, compatibility_contract, structure_sha256,
)
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_memory_servo_retention import ActorMemoryServoRetentionSACPilot
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.servo_success_retention import ServoRetainedCorrectionSAC
from kuavo_isaaclab_scene.rl.multi_box.experiments.actual_flap_residual_sac import AbsoluteGoalJawProjector


def fixtures(n=100):
    initial, replay = conservative_prepare(*pristine())
    compatible = compatibility_contract(initial['goal_contract'])
    anchor = structure_sha256(initial['body_anchor_state'])
    entries = []
    for i, region in enumerate(REGIONS):
        _, outcome = episode(region, n=n, env=i)
        outcome['layout']['split'] = 'train'
        raw, _ = decoder_fixture(n)
        raw[:, 94:98] = 0; raw[:, 94+i] = 1
        raw[:, 450] = torch.arange(n)
        labels = torch.zeros(n, 21); labels[:, 1] = .2; labels[:, 19:] = -1
        entries.append(dict(identity=f'closed_SAC/wave1/env{i}/seed{100+i}',
                            outcome=outcome, actor_obs=raw, action=labels))
    memory = dict(format=FORMAT, actor_dim=518, episodes=entries,
        source_checkpoint_SHA256='f'*64, compatibility=compatible, frozen_body_anchor_SHA256=anchor)
    proof = {k: True for k in ('writer_and_supervisor_closed_exit0_final_Drive_verified',
        'all27_successful_TRAIN_paths_and12476_rows_own_HDF_and_manifest_exact',
        'actor_observation_and_goal_and_frozen_anchor_and_physics_success_control_compatible',
        'current_stricter_reset_relative_drop10cm_checked_entire_attempt',
        'all_labels_fit_current_actor_bounds_and_jawgate',
        'all_absolute_goals_decode_to_actual_recorded_body_and_jaw_commands',
        'source_TRAIN_and_new_TRAIN_and_all_DEV_FINAL_seeds_disjoint',
        'only_actor_obs_and_action_and_success_provenance_exported',
        'no_critic_reward_next_state_or_terminal_label_exported', 'no_active_HDF_or_replay_read')}
    proof.update(actor_memory_compatibility=compatible, frozen_body_anchor_SHA256=anchor,
                 source_checkpoint_SHA256='f'*64)
    return initial, replay, memory, proof


def restore(data):
    return ActorTrainMemory(data, compatibility=data['compatibility'],
                            frozen_anchor_SHA256=data['frozen_body_anchor_SHA256'])


def test_old_actor_memory_has_only_actor_labels_and_balances_all_regions_and_tail():
    _, _, data, _ = fixtures(); bank = restore(data)
    torch.manual_seed(36)
    batch, stats = bank.sample_actor(4096, 'cpu', None)
    assert set(batch) == {'actor_obs', 'action'}
    assert batch['actor_obs'].shape == (4096, 518)
    assert batch['actor_obs'][:, 94:98].argmax(-1).bincount().tolist() == [1024]*4
    assert int((batch['actor_obs'][:, 450] >= 36).sum()) >= 2048
    assert stats['successful_train_actor_designated_tail_rows'] == 2048
    assert stats['previous_success_actor_memory_rows'] == 4096
    assert stats['previous_success_memory_rows_in_Q_batch'] == 0
    assert bank.report()['critic_rows'] == bank.report()['reward_rows'] == 0
    restored = restore(deepcopy(bank.state()))
    assert identical(restored.state(), bank.state())
    data['episodes'][0]['action'].fill_(123)
    assert bank.state()['episodes'][0]['action'].abs().max() <= 1


def test_new_successes_share_actor_sampling_but_old_paths_never_supply_Q_fields():
    _, _, data, _ = fixtures(); bank = restore(data)
    current = SimpleNamespace(episodes={r: [] for r in REGIONS})
    for e in data['episodes']:
        rows = {k: e[k].clone() for k in ('actor_obs', 'action')}
        rows['actor_obs'][:, 451] = 7
        rows['reward'] = torch.full((100,), 999.)
        current.episodes[e['outcome']['layout']['target_region']].append(dict(rows=rows))
    torch.manual_seed(17)
    batch, stats = bank.sample_actor(512, 'cpu', current)
    assert set(batch) == {'actor_obs', 'action'}
    assert 0 < stats['new_success_actor_memory_rows'] < 512
    assert stats['new_success_actor_memory_rows'] == int(batch['actor_obs'][:, 451].eq(7).sum())
    assert batch['actor_obs'][:, 94:98].argmax(-1).bincount().tolist() == [128]*4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA restore regression')
@pytest.mark.parametrize('learner_device', ['cpu', 'cuda:0'])
def test_cuda_checkpoint_memory_and_online_CPU_successes_sample_on_learner_device(learner_device):
    _, _, data, _ = fixtures()
    expected_hash = structure_sha256(data)
    for e in data['episodes']:
        for key in ('actor_obs', 'action'):
            e[key] = e[key].to('cuda:0')
    bank = restore(data)
    assert structure_sha256(bank.state()) == expected_hash
    assert all(e[key].device.type == 'cpu' for e in bank.state()['episodes']
               for key in ('actor_obs', 'action'))
    # add_episode() and restore() keep the online TrainSuccessBank on CPU.
    current = SimpleNamespace(episodes={r: [] for r in REGIONS})
    for e in bank.state()['episodes']:
        rows = {key: e[key].clone() for key in ('actor_obs', 'action')}
        rows['actor_obs'][:, 451] = 7
        current.episodes[e['outcome']['layout']['target_region']].append(dict(rows=rows))
    torch.manual_seed(17)
    batch, stats = bank.sample_actor(64, learner_device, current)
    assert 0 < stats['new_success_actor_memory_rows'] < 64
    assert stats['new_success_actor_memory_rows'] == int(batch['actor_obs'][:, 451].eq(7).sum())
    assert stats['previous_success_memory_rows_in_Q_batch'] == 0
    assert all(v.device == torch.device(learner_device) for v in batch.values())
    assert batch['actor_obs'][:, 94:98].argmax(-1).bincount().tolist() == [16]*4


@pytest.mark.parametrize('fault', ['reward', 'critic', 'DEV', 'unsafe', 'drop',
                                 'region', 'approach', 'jaw', 'nan', 'duplicate', 'missing'])
def test_memory_rejects_reward_fields_or_unverified_success_rows(fault):
    _, _, data, _ = fixtures(); e = data['episodes'][0]
    if fault == 'reward': e['reward'] = torch.zeros(100)
    elif fault == 'critic': data['critic_obs'] = torch.zeros(100, 578)
    elif fault == 'DEV': e['outcome']['split'] = 'validation'
    elif fault == 'unsafe': e['outcome']['result']['unsafe'] = True
    elif fault == 'drop': e['outcome']['result']['unsafe_causes']['box_drop'] = True
    elif fault == 'region': e['actor_obs'][:, 94:98] = 0
    elif fault == 'approach': e['actor_obs'][0, -6] = 0
    elif fault == 'jaw': e['action'][0, 19] = 0
    elif fault == 'nan': e['actor_obs'][0, 0] = float('nan')
    elif fault == 'duplicate': data['episodes'].append(deepcopy(e))
    else: data['episodes'].pop()
    with pytest.raises(ValueError): restore(data)


def test_memory_requires_same_coordinates_safety_and_frozen_body_anchor():
    _, _, data, _ = fixtures()
    with pytest.raises(ValueError):
        ActorTrainMemory(data, compatibility={}, frozen_anchor_SHA256=data['frozen_body_anchor_SHA256'])
    with pytest.raises(ValueError):
        ActorTrainMemory(data, compatibility=data['compatibility'], frozen_anchor_SHA256='0'*64)
    assert structure_sha256({'b': torch.tensor([2.]), 'a': 1}) == structure_sha256({'a': 1, 'b': torch.tensor([2.])})
    assert structure_sha256({'a': torch.tensor([1.])}) != structure_sha256({'a': torch.tensor([2.])})


def test_initialization_keeps_all_models_optimizers_and_reward_banks_and_only_adds_actor_memory():
    initial, replay, memory, proof = fixtures()
    before, old_replay = deepcopy(initial), deepcopy(replay)
    state, experience = prepare(initial, replay, memory, proof)
    assert identical(before, initial) and identical(old_replay, replay)
    for k in ('model', 'config', 'optimizers', 'hybrid_contract', 'successful_train_transitions'):
        assert identical(state[k], initial[k])
    for k in ('executed_goal_transitions', 'successful_train_transitions', 'measured_train_credit_bank'):
        assert identical(experience[k], replay[k])
    assert state['goal_contract'] == initial['goal_contract'] | dict(
        name=ActorMemoryServoRetentionSACPilot.artifact_type, actor_training_memory=actor_memory_contract())
    assert identical(state['actor_training_memory'], memory)
    assert staged_policy_class(state['artifact_type']) is ActorMemoryServoRetentionSACPilot


@pytest.mark.parametrize('fault', ['audit', 'anchor', 'contract', 'Q', 'replay', 'optimizer', 'collection'])
def test_actor_memory_fork_rejects_stale_audit_and_started_training(fault):
    initial, replay, memory, proof = fixtures()
    if fault == 'audit': proof['current_stricter_reset_relative_drop10cm_checked_entire_attempt'] = False
    elif fault == 'anchor': proof['frozen_body_anchor_SHA256'] = '0'*64
    elif fault == 'contract': proof['actor_memory_compatibility'] = {}
    elif fault == 'Q': initial['critic_updates'] = 1
    elif fault == 'replay': replay['executed_goal_transitions']['reward'] = torch.ones(1)
    elif fault == 'optimizer': initial['optimizers'][0]['state'] = {0: dict(step=torch.tensor(1))}
    else: initial['body_behavior_statistics']['episodes_drawn'] = 1
    with pytest.raises(ValueError): prepare(initial, replay, memory, proof)


def test_actual_hybrid_update_keeps_Q_identical_and_applies_actor_memory_gradient():
    raw, encoder = decoder_fixture(8)
    agents = []
    for _ in range(2):
        agent = ServoRetainedCorrectionSAC(518, 4, 21,
            SACConfig(hidden=16, actor_lr=.001, entropy_backup=False),
            action_projector=AbsoluteGoalJawProjector())
        agent.correction_radius = .3
        agent.executed_body_anchor = lambda obs: obs.new_zeros(len(obs), 19)
        agent.goal_servo_critic_encoder = encoder
        agents.append(agent)
    ordinary, memory = agents; memory.load_state_dict(ordinary.state_dict())
    labels = ordinary.act(raw, True); labels[:, 1] = .2
    batch = dict(actor_obs=raw, critic_obs=torch.zeros(8, 4), action=ordinary.act(raw, True),
        next_actor_obs=raw.clone(), next_critic_obs=torch.zeros(8, 4),
        reward=torch.zeros(8), terminated=torch.zeros(8, dtype=torch.bool))
    reports = []
    for i, agent in enumerate(agents):
        torch.manual_seed(5001)
        options = dict(successful_train=dict(actor_obs=raw, action=labels),
            success_goal_weight=50., success_jaw_weight=.05) if i else {}
        reports.append(agent.update(batch, **options))
    assert reports[0]['q_loss'] == reports[1]['q_loss']
    for k in ('q1', 'q2', 'target1', 'target2'):
        assert identical(getattr(ordinary, k).state_dict(), getattr(memory, k).state_dict())
    assert not identical(ordinary.actor.state_dict(), memory.actor.state_dict())
    assert reports[1]['success_goal_weight'] == 50.
    assert reports[1]['success_servo_interval_coefficient'] == .01
    with torch.no_grad():
        assert (memory.act(raw, True)[:, 1]-.2).square().mean() < (ordinary.act(raw, True)[:, 1]-.2).square().mean()
