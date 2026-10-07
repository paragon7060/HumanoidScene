"""Collection forks must never import or relabel learned reward-bearing data."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/rl'))

from prepare_actual_success_actor_tail import identical
from prepare_greedy_collection_actor import prepare
from test_rl_episode_return_credit import actor_inputs
from prepare_episode_return_actor import prepare as return_prepare
from kuavo_isaaclab_scene.rl.multi_box.experiments.body_behavior_exploration import (
    body_behavior_config, body_behavior_statistics, VARIANT, GREEDY_REST_VARIANT,
)


def inputs():
    initial, experience, actor = actor_inputs()
    state,replay=return_prepare(initial,experience,actor)
    for value in (state,replay):
        value.update(body_behavior=body_behavior_config(VARIANT),
            body_behavior_statistics=body_behavior_statistics())
    return state,replay


def test_return_servo_fork_keeps_all_models_optimizers_labels_and_contracts():
    state,replay=inputs();before,exp=deepcopy(state),deepcopy(replay)
    new,experience=prepare(state,replay,source_checkpoint='closed_pristine_fixture.pt')
    assert identical(state,before) and identical(replay,exp)
    for key in ('model','optimizers','goal_contract','config','hybrid_contract',
                'successful_train_transitions','measured_train_credit'):
        assert identical(new[key],state[key])
    assert identical(experience['executed_goal_transitions'],replay['executed_goal_transitions'])
    assert identical(experience['measured_train_credit_bank'],replay['measured_train_credit_bank'])
    assert new['body_behavior']==experience['body_behavior']==body_behavior_config(GREEDY_REST_VARIANT)
    assert new['actor_updates']==new['critic_updates']==0


@pytest.mark.parametrize('bad',[
    'critic_updated','optimizer_updated','normalizer_updated','online_rows',
    'success_rows','credit_rows','credit_contract','collection_started','nonfinite_model',
])
def test_collection_fork_rejects_trained_or_incompatible_inputs(bad):
    state,replay=inputs()
    if bad=='critic_updated':state['critic_updates']=1
    elif bad=='optimizer_updated':state['optimizers'][0]['state']={0:{'step':torch.tensor(1)}}
    elif bad=='normalizer_updated':state['model']['critic_normalizer.count']=torch.tensor(1.)
    elif bad=='online_rows':replay['executed_goal_transitions']['reward']=torch.ones(1)
    elif bad=='success_rows':replay['successful_train_transitions']['episodes']['middle_left']=[{}]
    elif bad=='credit_rows':replay['measured_train_credit_bank']['episodes']['middle_left']=[{}]
    elif bad=='credit_contract':replay['measured_train_credit_bank']['config']={}
    elif bad=='collection_started':state['body_behavior_statistics']['episodes_drawn']=1
    elif bad=='nonfinite_model':state['model']['q1.weight'][0]=float('nan')
    with pytest.raises(ValueError):prepare(state,replay,source_checkpoint='fixture.pt')
