"""Pristine model/data migration only; synthetic fixtures are not physical success."""
from copy import deepcopy

import pytest
import torch

from test_rl_actor_train_memory import fixtures
from test_rl_size_workplaces import prepared
from prepare_actor_memory_servo import prepare as memory_prepare
from prepare_size_workplace_actor import prepare
from prepare_actual_success_actor_tail import identical
from kuavo_isaaclab_scene.rl.multi_box.experiments.actor_train_memory import compatibility_contract,structure_sha256
from kuavo_isaaclab_scene.rl.multi_box.experiments.staged_train_success import REGIONS
from kuavo_isaaclab_scene.rl.multi_box.experiments.regional_actor_servo import (
    RegionalActorMemorySACPilot,install_regional_actor,regional_actor_contract)
from test_rl_regional_actor_servo import agent as regional_agent


def source():
    initial,replay,memory,proof=fixtures(n=4)
    old,waypoints=prepared()
    for value in (initial,replay):
        value['goal_contract']['shelf_templates']=deepcopy(old['region_workplaces'])
        value['goal_contract']['context_order']=['held_phase','held_x_rack_m','held_y_rack_m',
            'sin_held_yaw','cos_held_yaw','policy_radius']
    memory['compatibility']=compatibility_contract(initial['goal_contract'])
    proof['actor_memory_compatibility']=deepcopy(memory['compatibility'])
    initial,replay=memory_prepare(initial,replay,memory,proof)
    # Build only the synthetic checkpoint architecture; the actual full
    # trainer/asset restoration is a separate prerequisite for a real run.
    agent=regional_agent();install_regional_actor(agent);routed=agent.actor.network
    initial['model']=agent.state_dict()
    initial['config'].update(hidden=agent.config.hidden,freeze_actor_normalizer=True,actor_feature_mode='flat')
    initial['optimizers'][0]=torch.optim.Adam(routed.parameters(),lr=initial['config']['actor_lr']).state_dict()
    origin=dict(kind='compatible_actor_only_region_components_v1',components={r:dict(
        checkpoint_SHA256='d'*64,network_SHA256=structure_sha256(routed.heads[i].state_dict()))
        for i,r in enumerate(REGIONS)},frozen_body_anchor_SHA256=structure_sha256(initial['body_anchor_state']),
        actor_normalizer_SHA256=structure_sha256({k:v for k,v in initial['model'].items() if k.startswith('actor_normalizer.')}))
    for value in (initial,replay):
        value['artifact_type']=RegionalActorMemorySACPilot.artifact_type
        value['goal_contract'].update(name=RegionalActorMemorySACPilot.artifact_type,regional_actor=regional_actor_contract())
        value['regional_actor_initialization']=deepcopy(origin)
    return initial,replay,waypoints


def test_size_targets_do_not_import_learned_Q_or_relabel_old_actor_memory():
    old,replay,waypoints=source();original=deepcopy(old);old_replay=deepcopy(replay)
    changed,experience=prepare(old,replay,waypoints,source_checkpoint_SHA256='c'*64)
    assert identical(old,original) and identical(replay,old_replay)
    assert changed['goal_contract']['shelf_templates']==waypoints['size_workplaces']
    assert changed['goal_contract']==experience['goal_contract']
    for key in ('model','optimizers','actor_training_memory','body_anchor_state','frozen_actor_prior'):
        assert identical(changed[key],old[key])
    assert changed['actor_updates']==changed['critic_updates']==0
    assert compatibility_contract(changed['goal_contract'])==compatibility_contract(old['goal_contract'])
    assert changed['actor_training_memory']['compatibility']['actor']['shelf_templates']==old['goal_contract']['shelf_templates']
    assert not any(len(v) for v in experience['executed_goal_transitions'].values())


@pytest.mark.parametrize('wrong',['actor_updated','Q_updated','optimizer_used','wrong_checkpoint',
    'old_memory_relabelled','wrong_context','wrong_source_targets'])
def test_changed_learning_or_reference_contract_cannot_silently_migrate(wrong):
    initial,replay,waypoints=source();sha='c'*64
    if wrong=='actor_updated':initial['actor_updates']=1
    elif wrong=='Q_updated':initial['critic_updates']=1
    elif wrong=='optimizer_used':initial['optimizers'][0]['state'][0]={'step':1}
    elif wrong=='wrong_checkpoint':sha='f'*64
    elif wrong=='old_memory_relabelled':
        for value in (initial,replay):
            value['actor_training_memory']['compatibility']['actor']['shelf_templates']=deepcopy(waypoints['size_workplaces'])
    elif wrong=='wrong_context':
        for value in (initial,replay):value['goal_contract']['context_order']=['hidden_unobserved_base_target']
    else:
        for value in (initial,replay):value['goal_contract']['shelf_templates']['regions']['shelf_2_left']['template']['base_yaw_rack_rad']+=.01
    with pytest.raises(ValueError):prepare(initial,replay,waypoints,source_checkpoint_SHA256=sha)
