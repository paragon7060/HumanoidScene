"""Pure migration metadata/restore boundaries; not physical task success."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from test_rl_regional_guard_metric_cache import regional_fixture
from test_rl_success_update_guard import equal
from kuavo_isaaclab_scene.rl.multi_box.experiments import region_size_sac as split
from kuavo_isaaclab_scene.rl.multi_box.experiments.region_size_actor import split_region_size_actor_and_Adam
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_policy import staged_policy_class
import prepare_region_size_resume as command


def source_fixture(monkeypatch):
    # Full physical contract is tested separately on actual closed checkpoints.
    # This fixture isolates migration/serialization without real task data.
    monkeypatch.setattr(split, 'validate_independent_hand_close_state', lambda state: None)
    agent = regional_fixture()
    for i, p in enumerate(agent.actor.parameters()):
        agent.actor_optimizer.state[p] = dict(step=torch.tensor(float(i+1)),
            exp_avg=torch.full_like(p, .01), exp_avg_sq=torch.full_like(p, .02))
    source = agent.checkpoint() | dict(artifact_type=split.SOURCE_ARTIFACT,
        actor_updates=32, critic_updates=100, success_guard_statistics={},
        goal_contract=dict(name=split.SOURCE_ARTIFACT, regional_actor='independent_perceived_rack_region_heads_unchanged_routing_v1',
                           physical_contract={'fixture': True}, replay_capacity=1024),
        successful_train_transitions={'synthetic_protocol_fixture': True})
    return agent, source


def test_serialized_migration_matches_independent_live_Adam_clone_without_source_or_RNG_change(monkeypatch):
    agent, source = source_fixture(monkeypatch); before = deepcopy(source); rng = torch.get_rng_state().clone()
    saved = split.migrate_region_size_checkpoint(source, 'a'*64)
    split_region_size_actor_and_Adam(agent)
    assert equal(agent.state_dict(), saved['model'])
    assert equal([o.state_dict() for o in agent.optimizers], saved['optimizers'])
    assert equal(source, before) and torch.equal(rng, torch.get_rng_state())
    assert equal(source['success_guard_memory'], saved['success_guard_memory'])
    assert saved['actor_updates'] == 32 and saved['critic_updates'] == 100
    assert equal(source['goal_contract']['physical_contract'], saved['goal_contract']['physical_contract'])
    for a, b in ((0,24), (6,30)):
        assert saved['optimizers'][0]['state'][a]['step'].data_ptr() != saved['optimizers'][0]['state'][b]['step'].data_ptr()
    assert staged_policy_class(split.ARTIFACT) is split.RegionSizeIndependentHandCloseSACPilot


@pytest.mark.parametrize('fault', ('pattern', 'lookup', 'head_shape', 'head_value', 'step', 'moment', 'provenance', 'counter'))
def test_malformed_new_checkpoint_cannot_restore(monkeypatch, fault):
    _, source = source_fixture(monkeypatch); saved = split.migrate_region_size_checkpoint(source, 'a'*64)
    if fault == 'pattern': saved['model']['actor.network.type_patterns'][0,0] += 1
    elif fault == 'lookup': saved['model']['actor.network.head_lookup'][0,0] = 1
    elif fault == 'head_shape': saved['model']['actor.network.heads.4.0.weight'] = torch.zeros(1)
    elif fault == 'head_value': saved['model']['actor.network.heads.4.4.bias'][0] += 1
    elif fault == 'step': saved['optimizers'][0]['state'][0]['step'] = torch.ones(2)
    elif fault == 'moment': saved['optimizers'][0]['state'][0]['exp_avg'][0,0] = float('nan')
    elif fault == 'provenance': saved['region_size_initialization']['source_actor_updates'] = -1
    else: saved['actor_updates'] = 31
    with pytest.raises(ValueError): split.validate_region_size_state(saved)


def test_resume_retags_only_metadata_without_relabeling_rows_or_mutating_input(monkeypatch):
    _, source = source_fixture(monkeypatch); saved = split.migrate_region_size_checkpoint(source, 'a'*64)
    rows = {'actor_obs': torch.randn(2,518), 'action': torch.randn(2,21)}
    experience = dict(goal_contract=deepcopy(source['goal_contract']), executed_goal_transitions=rows,
                      successful_train_transitions=source['successful_train_transitions'], extra_provenance='kept')
    before = deepcopy(experience)
    new = split.migrate_region_size_resume_input(experience, source, saved)
    assert equal(experience, before) and new['executed_goal_transitions'] is rows
    assert new['goal_contract'] == saved['goal_contract'] and new['extra_provenance'] == 'kept'
    for key in rows: assert torch.equal(rows[key], before['executed_goal_transitions'][key])
    bad = deepcopy(experience); bad['successful_train_transitions']['synthetic_protocol_fixture'] = False
    with pytest.raises(ValueError): split.migrate_region_size_resume_input(bad, source, saved)
    with pytest.raises(ValueError): split.migrate_region_size_resume_input(experience, source | {'model': {}}, saved)


def test_terminal_success_candidate_cannot_masquerade_as_full_training_resume(monkeypatch):
    _, source = source_fixture(monkeypatch); source['TRAIN_success_candidate'] = {'fixed_DEV_success_verified': False}
    with pytest.raises(ValueError, match='full checkpoint'): split.migrate_region_size_checkpoint(source, 'a'*64)


def test_live_dependency_gate_precedes_torch_import_raw_file_access_and_output_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '')
    monkeypatch.setattr(command, 'verified_dependency', lambda *args: None)
    args = SimpleNamespace(unit='owned-live-unit',launch_record=tmp_path/'missing.json',
                           checkpoint=tmp_path/'absent.pt',destination=tmp_path/'new')
    with pytest.raises(RuntimeError, match='before Torch or raw replay'): command.prepare(args)
    assert not args.destination.exists()


def test_new_policy_cannot_silently_start_fresh_without_migration():
    with pytest.raises(ValueError, match='explicit compatible'): split.RegionSizeIndependentHandCloseSACPilot()
